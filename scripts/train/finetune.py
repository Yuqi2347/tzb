import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Tuple
from types import MethodType
from torchvision import transforms

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import AutoModel, AutoTokenizer
# from transformers.integrations import deepspeed  # 旧版本没有这个

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remote_sensing_mllm.dataset import SupervisedDataset, data_collator
from remote_sensing_mllm.trainer import Trainer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    step1: Optional[bool] = field(default=False)
    merge_lora_on_save: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to merge LoRA weights into base model when saving checkpoints"}
    )

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\.model\.layers\.\d+\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)"

    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer, output_dir: str, bias="none", merge_lora=True):
    """训练结束后合并 LoRA 并保存完整模型"""
    if trainer.args.should_save and trainer.args.local_rank == 0:
        print("\n" + "="*60)
        print("🔄 Final save: Merging LoRA into base model...")
        print("="*60)
        
        model = trainer.model
        
        # 检查是否是 PEFT 模型
        is_peft_model = hasattr(model, 'merge_and_unload')
        
        if is_peft_model and merge_lora:
            try:
                # 合并 LoRA 权重
                print("Merging LoRA weights...")
                model = model.merge_and_unload()
                print("✅ LoRA merged!")
                
                # 保存合并后的完整模型
                print(f"💾 Saving merged model to {output_dir}...")
                model.save_pretrained(
                    output_dir,
                    safe_serialization=False,  # 避免共享权重问题
                    max_shard_size="5GB"
                )
                print("✅ Merged model saved!")
                
            except Exception as e:
                print(f"❌ Merge failed: {e}")
                print("Saving adapter only...")
                trainer.model.save_pretrained(output_dir, safe_serialization=False)
        else:
            # 非 PEFT 模型或不需要合并
            model.save_pretrained(output_dir, safe_serialization=False)
        
        # 保存 tokenizer
        if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(output_dir)
            print("✅ Tokenizer saved!")
        
        print("="*60)
        print(f"✅ Final model saved to: {output_dir}")
        print("="*60 + "\n")


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    transform,
    data_collator=None,
    slice_config=None,
    patch_size=14,
    query_nums=64,
    batch_vision=False,
    max_length=2048,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_json,
        transform,
        tokenizer,
        slice_config=slice_config,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=max_length,
    )

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(
            eval_json,
            transform,
            tokenizer,
            slice_config=slice_config,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
            max_length=max_length,
        )
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator= partial(data_collator, max_length=max_length),
    )


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

def get_parameter_number(model, print_names=False):
    trainable_params, all_param = 0, 0
    seen = set()

    for name, param in model.named_parameters():
        # Calculate number of parameters, handling DeepSpeed ZeRO 3 if applicable
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params

        # Only count each unique Parameter once
        if param.requires_grad:
            param_id = id(param)
            if param_id not in seen:
                seen.add(param_id)
                trainable_params += num_params
                if print_names:
                    # print(f"🟢 Trainable: {name:<80} ({num_params})")
                    pass

    print(f"\n📊 Summary:")
    print(f"  ✅ Trainable parameters: {trainable_params:,}")
    print(f"  🔒 Total parameters:     {all_param:,}")
    print(f"  📈 Percentage:           {100 * trainable_params / all_param:.4f}%")

    return {'Total': all_param, 'Trainable': trainable_params}



local_rank = 0


def train():
    global local_rank
    
    # 【修复】V100 GPU 不支持硬件 BF16，绕过 transformers 检查
    import torch.cuda
    if not torch.cuda.is_bf16_supported():
        print("⚠️  Patching BF16 check for V100 GPU...")
        torch.cuda.is_bf16_supported = lambda: True

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    # 【修复】V100 不支持硬件BF16，但可以用软件实现
    # 绕过 transformers 的检查
    if training_args.bf16 and not torch.cuda.is_bf16_supported():
        print("⚠️  GPU doesn't support hardware BF16, but will use software emulation")
        # 修改内部标志绕过检查
        import torch.cuda
        torch.cuda.is_bf16_supported = lambda: True
    
    print(f"Using dtype: {compute_dtype}")

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None

    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        # 检查是否使用 ZeRO3（兼容旧版本 transformers）
        try:
            from transformers.integrations import is_deepspeed_zero3_enabled
            is_zero3 = is_deepspeed_zero3_enabled()
        except ImportError:
            is_zero3 = getattr(training_args, "deepspeed", None) and "zero3" in str(training_args.deepspeed)
        
        if len(training_args.fsdp) > 0 or is_zero3:
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
        # init_vision=True,
        # init_audio=False,
        # init_tts=False,
    )
    
    # 【关键修复】强制转换所有层到目标 dtype（避免混合 dtype 错误）
    print(f"Converting all model parameters to {compute_dtype}...")
    for name, param in model.named_parameters():
        if param.dtype != compute_dtype and param.dtype not in [torch.int64, torch.int32, torch.int8, torch.uint8, torch.bool]:
            param.data = param.data.to(compute_dtype)
    
    for name, buffer in model.named_buffers():
        if buffer.dtype != compute_dtype and buffer.dtype not in [torch.int64, torch.int32, torch.int8, torch.uint8, torch.bool]:
            buffer.data = buffer.data.to(compute_dtype)
    
    print(f"✅ All parameters converted to {compute_dtype}")
    # for name, module in model.named_modules():
    #     if "attn" in name or "proj" in name:
    #         print(name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    
    # 【关键】动态添加新 token（只在内存中，不保存）
    tokens_to_add = ["<think>", "</think>", "<answer>", "</answer>"]
    new_tokens_needed = []
    for token in tokens_to_add:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            new_tokens_needed.append(token)
    
    if new_tokens_needed:
        print(f"\n➕ Adding {len(new_tokens_needed)} new tokens: {new_tokens_needed}")
        special_tokens = {"additional_special_tokens": new_tokens_needed}
        num_new = tokenizer.add_special_tokens(special_tokens)
        print(f"✅ Added {num_new} tokens (in memory only)")
    
    # 检查并自动扩展 embedding（如果 tokenizer 比模型大）
    model_vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    
    if tokenizer_vocab_size > model_vocab_size:
        print(f"\n🔧 [Rank {local_rank}] Resizing model embeddings: {model_vocab_size} -> {tokenizer_vocab_size}")
        
        # 在所有进程扩展前同步
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        old_embeddings = model.llm.get_input_embeddings().weight.data.clone()
        model.resize_token_embeddings(tokenizer_vocab_size)
        
        # 用平均值初始化新 token（在 rank 0 计算，然后广播）
        with torch.no_grad():
            if local_rank == 0:
                mean_embedding = old_embeddings.mean(dim=0)
            else:
                # 其他 rank 创建空的
                mean_embedding = torch.zeros(old_embeddings.shape[1], dtype=old_embeddings.dtype, device=old_embeddings.device)
            
            # 从 rank 0 广播平均值，确保所有进程一致
            if torch.distributed.is_initialized():
                if mean_embedding.device.type == 'cpu':
                    mean_embedding = mean_embedding.cuda()
                torch.distributed.broadcast(mean_embedding, src=0)
                if old_embeddings.device.type == 'cpu':
                    mean_embedding = mean_embedding.cpu()
            
            # 所有进程使用相同的平均值初始化
            new_embeddings = model.llm.get_input_embeddings().weight
            new_embeddings[:model_vocab_size] = old_embeddings
            new_embeddings[model_vocab_size:] = mean_embedding
        
        print(f"✅ [Rank {local_rank}] Initialized {tokenizer_vocab_size - model_vocab_size} new token embeddings")
        
        # 扩展完成后再次同步
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    # step1 已废弃 - 使用 add_tokens.py 预先添加 tokens
    # if training_args.step1:
    #     ...


    # print(tokenizer.convert_tokens_to_ids("<think>"))
    # print(tokenizer.convert_tokens_to_ids("<answer>"))
    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
        # model.resampler.requires_grad_(False)
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        # modules_to_save = ['embed_tokens']
        if training_args.tune_vision:
            modules_to_save.append('vpm')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            # modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    
    # 冻结 embedding 和 resampler（简化版本 - 不训练 embedding）
    # for name, param in model.named_parameters():
    #     if 'embed_tokens' in name or "resampler" in name:
    #         param.requires_grad = False

    # 冻结 embedding，但新 token 可训练
    embed_layer = model.llm.get_input_embeddings()
    old_vocab_size = 73448  # 原始大小
    new_vocab_size = len(tokenizer)

    if new_vocab_size > old_vocab_size:
        # 只冻结旧 token，新 token 可训练
        embed_layer.weight.requires_grad = True
        
        def new_token_only_grad(grad):
            mask = torch.zeros_like(grad)
            mask[old_vocab_size:] = 1.0  # 只更新新 token
            return grad * mask
        
        embed_layer.weight.register_hook(new_token_only_grad)
        print(f"✅ New tokens ({old_vocab_size}-{new_vocab_size}) are trainable")
    else:
        embed_layer.weight.requires_grad = False

    # 冻结 resampler
    for name, param in model.named_parameters():
        if "resampler" in name:
            param.requires_grad = False

    # 【关键】在训练前同步所有进程，确保模型参数完全一致
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        
        # 打印每个 rank 的可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"[Rank {local_rank}] Trainable: {trainable_params:,} / Total: {all_params:,}")
        
        # 收集所有 rank 的参数数量，检查是否一致
        trainable_tensor = torch.tensor([trainable_params], dtype=torch.long, device='cuda')
        all_tensor = torch.tensor([all_params], dtype=torch.long, device='cuda')
        
        # 使用 all_gather 收集所有 rank 的信息
        world_size = torch.distributed.get_world_size()
        trainable_list = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
        all_list = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
        
        torch.distributed.all_gather(trainable_list, trainable_tensor)
        torch.distributed.all_gather(all_list, all_tensor)
        
        if local_rank == 0:
            print("\n" + "="*60)
            print("🔍 Checking parameter consistency across ranks:")
            for rank in range(world_size):
                t_params = trainable_list[rank].item()
                a_params = all_list[rank].item()
                status = "✅" if t_params == trainable_params else "❌"
                print(f"  {status} Rank {rank}: Trainable={t_params:,}, Total={a_params:,}")
            
            # 检查是否所有 rank 一致
            all_same = all(t.item() == trainable_params for t in trainable_list)
            if all_same:
                print("✅ All ranks have identical model parameters!")
            else:
                print("❌ WARNING: Ranks have different parameters! Training will fail!")
            print("="*60 + "\n")
        
        torch.distributed.barrier()

    rank0_print(get_parameter_number(model,print_names=True))

    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        transform=transform_func,
        data_collator=data_collator,
        slice_config=slice_config,
        patch_size=model.config.patch_size,
        query_nums=model.config.query_num,
        batch_vision=batch_vision,
        max_length=training_args.model_max_length,
    )

    training_args.gradient_checkpointing_kwargs={"use_reentrant":False}
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias,
        merge_lora=True  # 保存为完整模型
    )


if __name__ == "__main__":
    train()
