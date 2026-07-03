#!/usr/bin/env python
# train_grpo.py (DDP-ready)

import argparse, json, os, re, time
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
import requests
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remote_sensing_mllm.compute_reward import compute_reward, compute_format_reward
from remote_sensing_mllm.conversation import conv_templates

DEFAULT_IMAGE_TOKEN = "(<image>./</image>)"

def extract_tag_content(text: str, tag: str) -> str:
    """Return the inner text of the first <tag>...</tag> block in `text`."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def split_think_answer(text: str) -> Tuple[str, str]:
    """Split a model response into (<think>, <answer>) text segments."""
    return extract_tag_content(text, "think"), extract_tag_content(text, "answer")


def build_rgrpo_user_prompt(base_prompt: str, feedbacks: List[str]) -> str:
    """Append accumulated feedback to the base prompt so the model can revise its reasoning."""
    if not feedbacks:
        return base_prompt
    feedback_lines = "\n".join(
        f"[Feedback {idx + 1}]: {fb.strip()}" for idx, fb in enumerate(feedbacks) if fb
    )
    guidance = (
        "\n\nPlease incorporate every feedback item above. "
        "Re-evaluate your reasoning step-by-step in <think>...</think> before giving the final <answer>."
    )
    return f"{base_prompt}\n\nPrevious feedback:\n{feedback_lines}{guidance}"


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    """Move all tensor values in a processor output dict onto the target device."""
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device, non_blocking=True)
    # 处理嵌套的像素/尺寸信息
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        if isinstance(pv, torch.Tensor):
            inputs["pixel_values"] = pv.to(device, non_blocking=True)
        elif isinstance(pv, (list, tuple)):
            inputs["pixel_values"] = [
                item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                for item in pv
            ]
    if "tgt_sizes" in inputs:
        ts = inputs["tgt_sizes"]
        if isinstance(ts, torch.Tensor):
            inputs["tgt_sizes"] = ts.to(device)
        elif isinstance(ts, (list, tuple)):
            inputs["tgt_sizes"] = [
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in ts
            ]
    if "image_bound" in inputs and inputs["image_bound"] is not None:
        ib = inputs["image_bound"]
        if isinstance(ib, torch.Tensor):
            inputs["image_bound"] = ib.to(device)
        elif isinstance(ib, (list, tuple)):
            inputs["image_bound"] = [
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in ib
            ]
    return inputs


def call_rgrpo_feedback(
    api_base: str,
    model_name: str,
    api_key: Optional[str],
    query: str,
    think_text: str,
    answer_text: str,
    ground_truth: Optional[str],
    iteration_index: int,
    temperature: float,
    top_p: float,
    retry_limit: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call the external GPT-5 chat completion API to critique the current reasoning trace.

    Returns:
        feedback_text: A non-empty string when the call succeeds, otherwise None.
        error_message: Populated with a human readable error when the call fails.
    """
    if not api_key:
        return None, "Missing API key"

    endpoint = api_base.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    user_payload = (
        f"Query:\n{query}\n\n"
        f"Model reasoning (iteration {iteration_index}):\n{think_text}\n\n"
        f"Model answer:\n{answer_text}\n"
    )
    if ground_truth is not None:
        user_payload += f"\nReference answer:\n{ground_truth}\n"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert reasoning critic. "
                    "Identify specific logical or factual errors in the model's thinking steps. "
                    "Respond with concise bullet points describing mistakes and improvement guidance."
                ),
            },
            {
                "role": "user",
                "content": user_payload,
            },
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": 512,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error = None
    for attempt in range(retry_limit + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", None)
                )
                if content:
                    return content.strip(), None
                last_error = "Empty feedback content"
            else:
                try:
                    error_json = response.json()
                    detail = error_json.get("error", {}).get("message") or response.text
                except Exception:
                    detail = response.text
                last_error = f"HTTP {response.status_code}: {detail}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < retry_limit:
            time.sleep(1.5 * (attempt + 1))

    return None, last_error

def parse_args():
    p = argparse.ArgumentParser()
    # 基本配置
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--prompts_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    # 数据与训练
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--group_size", type=int, default=4, help="GRPO: 每个prompt采样的response数量")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    # 日志与保存
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--logging_dir", type=str, default=None)
    # Reward 权重
    p.add_argument("--think_r", type=float, default=0.5)
    p.add_argument("--answer_r", type=float, default=0.5)
    p.add_argument("--vqa_r", type=float, default=1.0)
    p.add_argument("--caption_r", type=float, default=0.5)
    p.add_argument("--grounding_r", type=float, default=1.0)
    p.add_argument("--penalty_unit", type=float, default=1.2)
    # 分阶段训练 & reward 调度
    p.add_argument("--warmup_steps", type=int, default=600)
    p.add_argument("--schedule_steps", type=int, default=600)
    p.add_argument("--format_only_steps", type=int, default=200)
    p.add_argument("--max_reward", type=float, default=1.0)
    p.add_argument("--max_adv", type=float, default=1.0)
    p.add_argument("--baseline_init", type=float, default=0.0)
    p.add_argument("--baseline_decay", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--beta_entropy", type=float, default=1e-3)
    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj")
    p.add_argument("--q_lora", action="store_true")
    # 兼容参数
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", type=str, default=None)
    # 图像根目录
    p.add_argument("--image_path", type=str, required=True)
    
    # 优化策略选择
    p.add_argument("--optimization_strategy", type=str, default="grpo", 
                   choices=["grpo", "ppo", "rgrpo"], help="优化策略: grpo, ppo 或 rgrpo")
    
    # PPO 特定参数
    p.add_argument("--ppo_clip_ratio", type=float, default=0.2, help="PPO裁剪比率")
    p.add_argument("--ppo_value_loss_coef", type=float, default=0.5, help="PPO价值损失系数")
    p.add_argument("--ppo_entropy_coef", type=float, default=0.01, help="PPO熵系数")
    p.add_argument("--ppo_max_grad_norm", type=float, default=0.5, help="PPO最大梯度范数")
    p.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="PPO GAE lambda参数")
    p.add_argument("--ppo_num_epochs", type=int, default=4, help="PPO内部epoch数")
    p.add_argument("--ppo_batch_size", type=int, default=4, help="PPO批次大小")
    p.add_argument("--ppo_mini_batch_size", type=int, default=1, help="PPO小批次大小")
    
    # RGRPO 特定参数
    p.add_argument("--rgrpo_max_iterations", type=int, default=4, help="RGRPO：最大迭代次数")
    p.add_argument("--rgrpo_success_threshold", type=float, default=1.0, help="RGRPO：判定正确的reward阈值")
    p.add_argument("--rgrpo_api_base", type=str, default="https://xiaoai.plus/v1", help="RGRPO：GPT-5 API base URL")
    p.add_argument("--rgrpo_api_model", type=str, default="gpt-5", help="RGRPO：GPT-5 模型名称")
    p.add_argument("--rgrpo_api_key_env", type=str, default="RGRPO_API_KEY", help="RGRPO：存放API KEY的环境变量名")
    p.add_argument("--rgrpo_feedback_temperature", type=float, default=0.2, help="RGRPO：反馈生成温度")
    p.add_argument("--rgrpo_feedback_top_p", type=float, default=0.9, help="RGRPO：反馈生成top_p")
    p.add_argument("--rgrpo_retry_limit", type=int, default=2, help="RGRPO：反馈生成最大重试次数")
    
    return p.parse_args()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process():
    return get_rank() == 0

def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

def collate_fn(batch):
    return batch

class VRSDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, prompt, eval_type):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.prompt = prompt
        self.eval_type = eval_type

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = os.path.join(self.image_folder, line["image"]) \
            if not line["image"].startswith(self.image_folder) else line["image"]
        qs = line["conversations"][0]["value"].replace("<image>", "").replace("</p>", "</p> in the form of normalized coordinates.")
        qs = DEFAULT_IMAGE_TOKEN + qs
        if "[caption]" in qs:
            qs += (
            "Please describe the content of the following remote sensing image in detail, reply in plain text paragraphs without structured output. "
            "First, give an overall one sentence summary of the geographical and functional types (e.g., city, forest, airport, farmland, port, etc.). "
            "Then divide the image conceptually into nine parts (from top-left to bottom-right) and describe the main objects, structures, "
            "and their colors, densities, and spatial layouts in each part in one sentence separately. "
            "Finally, summarize the overall spatial composition and provide high-level reasoning, including possible environmental, "
            "functional, or socioeconomic implications that can be inferred from the image in one sentence(e.g., human activity, traffic importance, "
            "urban planning, ecological balance)."
        )
        gt = line["conversations"][1]["value"]
        
        # 处理TIFF/TIF格式图片
        image = None
        if image_file.lower().endswith(('.tif', '.tiff')):
            # 方法1: 尝试使用tifffile库（专门处理科学/遥感TIFF）
            try:
                import tifffile
                import numpy as np
                img_array = tifffile.imread(image_file)
                # 处理多通道情况
                if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                    img_array = img_array[:, :, :3]  # 取前3个通道
                elif len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)  # 灰度转RGB
                # 归一化到0-255
                if img_array.max() > 255:
                    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                image = Image.fromarray(img_array, mode='RGB')
            except ImportError:
                pass  # tifffile未安装，尝试下一个方法
            except Exception as e:
                print(f"tifffile failed for {image_file}: {e}")
            
            # 方法2: 尝试使用cv2
            if image is None:
                try:
                    import cv2
                    import numpy as np
                    img_array = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
                    if img_array is not None:
                        # 处理多通道
                        if len(img_array.shape) == 3:
                            if img_array.shape[2] >= 3:
                                img_array = img_array[:, :, :3]
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        elif len(img_array.shape) == 2:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        # 归一化
                        if img_array.max() > 255:
                            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                        image = Image.fromarray(img_array)
                except Exception as e:
                    print(f"cv2 failed for {image_file}: {e}")
            
            # 方法3: 尝试PIL with relaxed error handling
            if image is None:
                try:
                    from PIL import TiffImagePlugin
                    import warnings
                    warnings.filterwarnings('ignore', category=UserWarning)
                    TiffImagePlugin.READ_LIBTIFF = False  # 使用PIL内置解码器
                    with Image.open(image_file) as img:
                        img.load()
                        image = img.convert("RGB")
                except Exception as e:
                    print(f"PIL relaxed mode failed for {image_file}: {e}")
        
        # 对于非TIFF格式或TIFF所有方法都失败的情况
        if image is None:
            try:
                image = Image.open(image_file).convert("RGB")
            except Exception as e:
                print(f"Warning: All methods failed to load {image_file}: {e}")
                print(f"Creating blank placeholder image")
                # 创建一个空白图片作为fallback
                image = Image.new('RGB', (224, 224), color='gray')
        
        return qs, image, gt

    def __len__(self):
        return len(self.questions)

def create_data_loader(questions, data_name, image_folder, tokenizer, image_processor, prompt, eval_type,
                       batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    if data_name == 'vrs':
        dataset = VRSDataset(questions, image_folder, tokenizer, image_processor, prompt, eval_type)
    else:
        raise ValueError("dataset not exist!!")
    sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False) if get_world_size() > 1 else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    return data_loader, sampler

def format_time(seconds):
    seconds = float(max(0.0, seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def main():
    args = parse_args()
    from datetime import timedelta

    # ===== DDP 初始化 =====
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=1800)
        )
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 输出目录与日志
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logging_dir or os.path.join(args.output_dir, "logs"))
    else:
        writer = None

    # ==== 加载模型 ====
    if args.resume_from_checkpoint:
        if is_main_process(): print(f"🔄 从 checkpoint 恢复: {args.resume_from_checkpoint}")
        base_model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None))
        model = PeftModel.from_pretrained(base_model, args.resume_from_checkpoint)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from_checkpoint, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.resume_from_checkpoint, trust_remote_code=True)
        # ========== 添加 special tokens（和第一阶段保持一致）==========
        tokens_to_add = ["<think>", "</think>", "<answer>", "</answer>"]
        new_tokens_needed = []
        for token in tokens_to_add:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id == tokenizer.unk_token_id:
                new_tokens_needed.append(token)
        
        if new_tokens_needed:
            if is_main_process():
                print(f"⚠️ 需要添加 {len(new_tokens_needed)} 个 special tokens: {new_tokens_needed}")
            special_tokens = {"additional_special_tokens": new_tokens_needed}
            num_new = tokenizer.add_special_tokens(special_tokens)
            if is_main_process():
                print(f"✅ Added {num_new} tokens to tokenizer")
            
            # 检查并扩展 embedding
            model_vocab_size = model.llm.get_input_embeddings().weight.shape[0] if hasattr(model, 'llm') else model.get_input_embeddings().weight.shape[0]
            tokenizer_vocab_size = len(tokenizer)
            if tokenizer_vocab_size > model_vocab_size:
                if is_main_process():
                    print(f"🔧 扩展模型 embedding: {model_vocab_size} → {tokenizer_vocab_size}")
                if hasattr(model, 'llm'):
                    model.llm.resize_token_embeddings(tokenizer_vocab_size)
                else:
                    model.resize_token_embeddings(tokenizer_vocab_size)
        else:
            if is_main_process():
                print(f"✅ Special tokens 已存在，无需添加")
        # ============================================================
        
    else:
        if is_main_process(): print(f"🔧 新训练: {args.model_name_or_path}")
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None))
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

        # ========== 添加 special tokens（和第一阶段保持一致）==========
        tokens_to_add = ["<think>", "</think>", "<answer>", "</answer>"]
        new_tokens_needed = []
        for token in tokens_to_add:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id == tokenizer.unk_token_id:
                new_tokens_needed.append(token)
        
        if new_tokens_needed:
            if is_main_process():
                print(f"⚠️ 需要添加 {len(new_tokens_needed)} 个 special tokens: {new_tokens_needed}")
            special_tokens = {"additional_special_tokens": new_tokens_needed}
            num_new = tokenizer.add_special_tokens(special_tokens)
            if is_main_process():
                print(f"✅ Added {num_new} tokens to tokenizer")
            
            # 检查并扩展 embedding
            model_vocab_size = model.llm.get_input_embeddings().weight.shape[0] if hasattr(model, 'llm') else model.get_input_embeddings().weight.shape[0]
            tokenizer_vocab_size = len(tokenizer)
            if tokenizer_vocab_size > model_vocab_size:
                if is_main_process():
                    print(f"🔧 扩展模型 embedding: {model_vocab_size} → {tokenizer_vocab_size}")
                if hasattr(model, 'llm'):
                    model.llm.resize_token_embeddings(tokenizer_vocab_size)
                else:
                    model.resize_token_embeddings(tokenizer_vocab_size)
        else:
            if is_main_process():
                print(f"✅ Special tokens 已存在，无需添加")
        # ============================================================

    # LoRA
    if args.use_lora and not args.resume_from_checkpoint:
        if hasattr(model, "llm"):
            for _, param in model.llm.named_parameters(): param.requires_grad=False
        target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                 target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none")
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model = model.to(device)
    model.train()
    if is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.2f}%)")

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ==== 优化器 & step ====
    optimizer = AdamW((p for p in model.parameters() if getattr(p,"requires_grad",False)), lr=args.lr)
    start_step = 0
    if args.resume_from_checkpoint:
        opt_file = os.path.join(args.resume_from_checkpoint, 'optimizer.pt')
        step_file = os.path.join(args.resume_from_checkpoint, 'step.txt')
        if os.path.exists(opt_file) and os.path.exists(step_file):
            optimizer.load_state_dict(torch.load(opt_file, map_location=device))
            start_step = int(open(step_file).read())
            if is_main_process(): print(f"🔄 恢复 optimizer, 从 step {start_step} 开始")

    # ==== DataLoader ====
    questions = json.load(open(args.prompts_path,'r'))
    prompt_template = """You are a remote sensing assistant. Think step by step using <think>...</think> and give your answer in <answer>...</answer>."""
    data_loader, sampler = create_data_loader(questions, 'vrs', args.image_path, tokenizer, processor,
                                              prompt=prompt_template, eval_type='vqa', batch_size=args.batch_size)

    warmup_steps = args.warmup_steps
    schedule_steps = args.schedule_steps
    beta = args.beta_entropy
    baseline = args.baseline_init
    decay = args.baseline_decay
    step = start_step
    start_time = time.time()
    last_log_time = start_time
    length = len(data_loader)
    
    # 梯度累积相关
    gradient_accumulation_steps = args.gradient_accumulation_steps
    accumulation_counter = 0

    def unwrap(m):
        return m.module if hasattr(m, "module") else m

    # ==== 策略选择 ====
    if is_main_process():
        print(f"🚀 使用优化策略: {args.optimization_strategy.upper()}")
        if args.optimization_strategy == "ppo":
            print(f"   PPO参数: clip_ratio={args.ppo_clip_ratio}, num_epochs={args.ppo_num_epochs}")
        elif args.optimization_strategy == "grpo":
            print(f"   GRPO参数: group_size={args.group_size}, warmup_steps={args.warmup_steps}")
        elif args.optimization_strategy == "rgrpo":
            print(
                f"   RGRPO参数: group_size={args.group_size}, max_iterations={args.rgrpo_max_iterations}, "
                f"success_threshold={args.rgrpo_success_threshold}"
            )

    # ==== 训练循环 ====
    for epoch in range(args.epochs):
        if sampler is not None: sampler.set_epoch(epoch)
        for idx, batch in enumerate(data_loader):
            prompt_text, image, gt = batch[0]
            base_user_prompt = prompt_text
            msgs = [{"role":"user","content":prompt_text},{"role":"system","content":prompt_template}]
            prompt_input = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt_input, images=image, return_tensors='pt')
            inputs = move_inputs_to_device(inputs, device)
            
            # 🔥 清理临时变量
            del msgs
            torch.cuda.synchronize()

            # ========== 根据策略选择训练方法 ==========
            # 初始化变量用于日志记录
            total_policy_loss_scalar = 0.0
            total_entropy_scalar = 0.0
            resp = ""
            r = 0.0
            
            if args.optimization_strategy == "rgrpo":
                del inputs
            
            if args.optimization_strategy == "grpo":
                # ========== GRPO: 组采样 ==========
                # 对同一个prompt生成K个response
                group_responses = []  # 存储 (gen_ids, resp, gen_part)
                group_rewards = []
                
                for k in range(args.group_size):
                    # 生成response（使用不同温度增加多样性）
                    temperature = 0.7 + k * 0.05  # 0.7, 0.75, 0.8, 0.85
                    with torch.no_grad():
                        gen_ids = unwrap(model).generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            pixel_values=inputs.get('pixel_values'),
                            tgt_sizes=inputs.get('tgt_sizes'),
                            image_bound=inputs.get('image_bound'),
                            tokenizer=tokenizer,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True, 
                            temperature=temperature
                        )
                    resp = tokenizer.batch_decode(gen_ids,skip_special_tokens=False)[0].replace("<|im_end|>", "").replace("<|im_start|>", "")
                    
                    # 计算奖励
                    fmt_r = compute_format_reward(resp, perfect_r=1.0, penalty_unit=args.penalty_unit)
                    task_r = 0.0 if (epoch==0 and idx<=args.format_only_steps) else compute_reward(prompt_input, resp, gt,
                                                                                                vqa_r=args.vqa_r,
                                                                                                caption_r=args.caption_r,
                                                                                                grounding_r=args.grounding_r)
                    if step < warmup_steps:
                        r = fmt_r
                    else:
                        mix = min(1.0, (step-warmup_steps)/max(1,schedule_steps))
                        r = fmt_r + mix*task_r
                    r = max(min(r, args.max_reward), -args.max_reward)
                    
                    # 保存
                    gen_part = gen_ids[:, inputs['input_ids'].shape[1]:].clone()
                    group_responses.append((gen_ids, resp, gen_part, fmt_r, task_r))
                    group_rewards.append(r)
                    
                    # 🔥 每次生成后立即清理显存
                    del gen_ids
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            
                # 计算组内baseline和advantages
                group_rewards = torch.tensor(group_rewards, device=device)
                baseline = group_rewards.mean()  # 组内平均奖励
                advantages = group_rewards - baseline
                
                # 可选：归一化advantages（减少方差）
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Clip advantages
                advantages = torch.clamp(advantages, -args.max_adv, args.max_adv)

                # ========== 对每个response计算策略梯度loss（逐个反向传播）==========
                # 🔥 关键改进：每个response计算后立即backward，避免多个计算图同时占用显存
                total_policy_loss_scalar = 0.0  # 用于记录（不保留计算图）
                total_entropy_scalar = 0.0      # 用于记录（不保留计算图）
                
                for response_idx, (gen_ids, resp, gen_part, fmt_r, task_r) in enumerate(group_responses):
                    adv = advantages[response_idx]
                    
                    # 拼接完整输入
                    input_ids = inputs['input_ids']
                    full_input = torch.cat([input_ids, gen_part], dim=1)
                    
                    # 准备inputs副本（避免修改原始inputs）
                    forward_inputs = {
                        'input_ids': full_input,
                        'attention_mask': torch.ones_like(full_input, device=full_input.device),
                        'position_ids': torch.arange(full_input.size(1), device=full_input.device).unsqueeze(0).expand(full_input.shape)
                    }
                    
                    # 复制其他必要的输入
                    if 'pixel_values' in inputs:
                        forward_inputs['pixel_values'] = inputs['pixel_values']
                    if 'tgt_sizes' in inputs:
                        forward_inputs['tgt_sizes'] = inputs['tgt_sizes']
                    if 'image_bound' in inputs:
                        forward_inputs['image_bound'] = inputs['image_bound']
                    
                    # Forward pass
                    outputs = model(data=forward_inputs, return_dict=True)
                    logits = outputs.logits[:, -gen_part.size(1):, :]
                    logp = torch.nn.functional.log_softmax(logits, dim=-1)
                    p = logp.exp()
                    
                    # 熵正则化
                    entropy = -(p*logp).sum(dim=-1).mean(dim=1).mean()
                    
                    # 选择实际生成的token的log概率
                    logp_sel = logp.gather(2, gen_part.unsqueeze(-1)).squeeze(-1)
                    logp_sum = logp_sel.sum(dim=1)
                    
                    # 策略梯度loss
                    policy_loss = -((logp_sum+args.epsilon)*adv).mean()
                    
                    # 🔥 计算当前response的总loss（已经考虑组平均和梯度累积）
                    loss = (policy_loss - beta * entropy) / args.group_size / gradient_accumulation_steps
                    
                    # 🔥 立即backward（梯度会累积到参数上）
                    loss.backward()
                    
                    # 记录用于日志（detach以避免计算图）
                    total_policy_loss_scalar += policy_loss.detach().item()
                    total_entropy_scalar += entropy.detach().item()
                    
                    # 🔥 立即清理当前response的所有变量和计算图
                    del outputs, logits, logp, p, logp_sel, logp_sum, policy_loss, entropy, loss, forward_inputs, full_input
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            
                # 计算平均值用于日志
                total_policy_loss_scalar = total_policy_loss_scalar / args.group_size
                total_entropy_scalar = total_entropy_scalar / args.group_size
                
            elif args.optimization_strategy == "rgrpo":
                group_responses = []
                group_rewards = []
                api_key = os.environ.get(args.rgrpo_api_key_env, "")
                if not api_key and is_main_process():
                    print(f"⚠️ RGRPO: 环境变量 {args.rgrpo_api_key_env} 未设置，反馈将退化为占位提示")
                for sample_idx in range(args.group_size):
                    feedback_history: List[str] = []
                    iteration_records: List[dict] = []
                    base_temperature = 0.7 + sample_idx * 0.05
                    final_prompt_input = prompt_input
                    final_resp = ""
                    final_gen_part = None
                    final_fmt_r = 0.0
                    final_task_r = 0.0
                    # 逐步迭代
                    for iter_idx in range(args.rgrpo_max_iterations):
                        iter_user_content = build_rgrpo_user_prompt(base_user_prompt, feedback_history)
                        iter_msgs = [
                            {"role": "user", "content": iter_user_content},
                            {"role": "system", "content": prompt_template},
                        ]
                        iter_prompt_input = processor.tokenizer.apply_chat_template(
                            iter_msgs, tokenize=False, add_generation_prompt=True
                        )
                        iter_inputs = processor(text=iter_prompt_input, images=image, return_tensors='pt')
                        iter_inputs = move_inputs_to_device(iter_inputs, device)
                        
                        temperature = base_temperature + 0.05 * iter_idx
                        with torch.no_grad():
                            gen_ids = unwrap(model).generate(
                                input_ids=iter_inputs['input_ids'],
                                attention_mask=iter_inputs['attention_mask'],
                                pixel_values=iter_inputs.get('pixel_values'),
                                tgt_sizes=iter_inputs.get('tgt_sizes'),
                                image_bound=iter_inputs.get('image_bound'),
                                tokenizer=tokenizer,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=True,
                                temperature=temperature
                            )
                        resp = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0].replace("<|im_end|>", "").replace("<|im_start|>", "")
                        
                        fmt_r = compute_format_reward(resp, perfect_r=1.0, penalty_unit=args.penalty_unit)
                        task_r = compute_reward(
                            iter_prompt_input,
                            resp,
                            gt,
                            vqa_r=args.vqa_r,
                            caption_r=args.caption_r,
                            grounding_r=args.grounding_r
                        )
                        think_text, answer_text = split_think_answer(resp)
                        success = task_r >= args.rgrpo_success_threshold
                        
                        input_len = iter_inputs['input_ids'].shape[1]
                        gen_part = gen_ids[:, input_len:].clone().cpu()
                        
                        record = {
                            "iteration": iter_idx + 1,
                            "temperature": temperature,
                            "task_reward": task_r,
                            "format_reward": fmt_r,
                            "success": success,
                            "think": think_text,
                            "answer": answer_text,
                        }
                        iteration_records.append(record)
                        
                        final_prompt_input = iter_prompt_input
                        final_resp = resp
                        final_gen_part = gen_part
                        final_fmt_r = fmt_r
                        final_task_r = task_r
                        
                        del iter_inputs
                        del gen_ids
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        
                        if success or iter_idx + 1 == args.rgrpo_max_iterations:
                            del gen_part
                            break
                        
                        feedback, error = call_rgrpo_feedback(
                            api_base=args.rgrpo_api_base,
                            model_name=args.rgrpo_api_model,
                            api_key=api_key,
                            query=base_user_prompt,
                            think_text=think_text,
                            answer_text=answer_text,
                            ground_truth=str(gt),
                            iteration_index=iter_idx + 1,
                            temperature=args.rgrpo_feedback_temperature,
                            top_p=args.rgrpo_feedback_top_p,
                            retry_limit=args.rgrpo_retry_limit,
                        )
                        if feedback is None:
                            feedback = f"Feedback request failed: {error}"
                        iteration_records[-1]["feedback"] = feedback
                        feedback_history.append(feedback)
                        del gen_part
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    
                    r = final_fmt_r if step < warmup_steps else final_fmt_r + min(1.0, (step - warmup_steps) / max(1, schedule_steps)) * final_task_r
                    r = max(min(r, args.max_reward), -args.max_reward)
                    group_rewards.append(r)
                    group_responses.append({
                        "prompt_input": final_prompt_input,
                        "gen_part": final_gen_part,
                        "resp": final_resp,
                        "fmt_r": final_fmt_r,
                        "task_r": final_task_r,
                        "feedback_history": feedback_history,
                        "iterations": iteration_records,
                    })
                
                group_rewards = torch.tensor(group_rewards, device=device)
                baseline = group_rewards.mean()
                advantages = group_rewards - baseline
                
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = torch.clamp(advantages, -args.max_adv, args.max_adv)
                
                total_policy_loss_scalar = 0.0
                total_entropy_scalar = 0.0
                
                for response_idx, entry in enumerate(group_responses):
                    adv = advantages[response_idx]
                    prompt_input_text = entry["prompt_input"]
                    gen_part = entry["gen_part"].to(device)
                    
                    base_inputs = processor(text=prompt_input_text, images=image, return_tensors='pt')
                    base_inputs = move_inputs_to_device(base_inputs, device)
                    input_ids = base_inputs['input_ids']
                    full_input = torch.cat([input_ids, gen_part], dim=1)
                    
                    forward_inputs = {
                        'input_ids': full_input,
                        'attention_mask': torch.ones_like(full_input, device=full_input.device),
                        'position_ids': torch.arange(full_input.size(1), device=full_input.device).unsqueeze(0).expand(full_input.shape)
                    }
                    if 'pixel_values' in base_inputs:
                        forward_inputs['pixel_values'] = base_inputs['pixel_values']
                    if 'tgt_sizes' in base_inputs:
                        forward_inputs['tgt_sizes'] = base_inputs['tgt_sizes']
                    if 'image_bound' in base_inputs and base_inputs['image_bound'] is not None:
                        forward_inputs['image_bound'] = base_inputs['image_bound']
                    
                    outputs = model(data=forward_inputs, return_dict=True)
                    logits = outputs.logits[:, -gen_part.size(1):, :]
                    logp = torch.nn.functional.log_softmax(logits, dim=-1)
                    p = logp.exp()
                    
                    entropy = -(p * logp).sum(dim=-1).mean(dim=1).mean()
                    logp_sel = logp.gather(2, gen_part.unsqueeze(-1)).squeeze(-1)
                    logp_sum = logp_sel.sum(dim=1)
                    
                    policy_loss = -((logp_sum + args.epsilon) * adv).mean()
                    loss = (policy_loss - beta * entropy) / args.group_size / gradient_accumulation_steps
                    loss.backward()
                    
                    total_policy_loss_scalar += policy_loss.detach().item()
                    total_entropy_scalar += entropy.detach().item()
                    
                    del outputs, logits, logp, p, logp_sel, logp_sum, policy_loss, entropy, loss, base_inputs, forward_inputs, full_input, gen_part
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                total_policy_loss_scalar = total_policy_loss_scalar / max(args.group_size, 1)
                total_entropy_scalar = total_entropy_scalar / max(args.group_size, 1)

            elif args.optimization_strategy == "ppo":
                # ========== PPO: 近端策略优化 ==========
                # 生成单个response
                with torch.no_grad():
                    gen_ids = unwrap(model).generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs.get('pixel_values'),
                        tgt_sizes=inputs.get('tgt_sizes'),
                        image_bound=inputs.get('image_bound'),
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True, 
                        temperature=0.7
                    )
                
                resp = tokenizer.batch_decode(gen_ids,skip_special_tokens=False)[0].replace("<|im_end|>", "").replace("<|im_start|>", "")
                
                # 计算奖励
                fmt_r = compute_format_reward(resp, perfect_r=1.0, penalty_unit=args.penalty_unit)
                task_r = 0.0 if (epoch==0 and idx<=args.format_only_steps) else compute_reward(prompt_input, resp, gt,
                                                                                                vqa_r=args.vqa_r,
                                                                                                caption_r=args.caption_r,
                                                                                                grounding_r=args.grounding_r)
                if step < warmup_steps:
                    r = fmt_r
                else:
                    mix = min(1.0, (step-warmup_steps)/max(1,schedule_steps))
                    r = fmt_r + mix*task_r
                r = max(min(r, args.max_reward), -args.max_reward)
                
                # PPO训练逻辑（简化版本）
                gen_part = gen_ids[:, inputs['input_ids'].shape[1]:].clone()
                
                # 拼接完整输入
                input_ids = inputs['input_ids']
                full_input = torch.cat([input_ids, gen_part], dim=1)
                
                # 准备inputs副本
                forward_inputs = {
                    'input_ids': full_input,
                    'attention_mask': torch.ones_like(full_input, device=full_input.device),
                    'position_ids': torch.arange(full_input.size(1), device=full_input.device).unsqueeze(0).expand(full_input.shape)
                }
                
                # 复制其他必要的输入
                if 'pixel_values' in inputs:
                    forward_inputs['pixel_values'] = inputs['pixel_values']
                if 'tgt_sizes' in inputs:
                    forward_inputs['tgt_sizes'] = inputs['tgt_sizes']
                if 'image_bound' in inputs:
                    forward_inputs['image_bound'] = inputs['image_bound']
                
                # Forward pass
                outputs = model(data=forward_inputs, return_dict=True)
                logits = outputs.logits[:, -gen_part.size(1):, :]
                logp = torch.nn.functional.log_softmax(logits, dim=-1)
                p = logp.exp()
                
                # 熵正则化
                entropy = -(p*logp).sum(dim=-1).mean(dim=1).mean()
                
                # 选择实际生成的token的log概率
                logp_sel = logp.gather(2, gen_part.unsqueeze(-1)).squeeze(-1)
                logp_sum = logp_sel.sum(dim=1)
                
                # PPO损失（简化版本，实际需要更复杂的实现）
                policy_loss = -(logp_sum * r).mean()
                
                # 总损失
                total_policy_loss_scalar = policy_loss.item()
                total_entropy_scalar = entropy.item()
                
                loss = (policy_loss - args.ppo_entropy_coef * entropy) / gradient_accumulation_steps
                loss.backward()
                
                # 清理（保留resp和r用于日志）
                del outputs, logits, logp, p, logp_sel, logp_sum, policy_loss, entropy, loss, forward_inputs, full_input, gen_ids, gen_part
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            accumulation_counter += 1
            
            # 只有在累积了足够步数后才更新参数
            if accumulation_counter % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # logging
            if is_main_process() and (step % args.log_interval==0):
                now = time.time()
                dt = now - last_log_time
                elapsed = now - start_time
                
                # 显示真实的loss（已经是平均后的标量）
                if args.optimization_strategy == "grpo":
                    true_loss = total_policy_loss_scalar - beta * total_entropy_scalar
                    # 计算组内统计
                    group_rewards_np = group_rewards.cpu().numpy()
                    r_mean = baseline.item()  # 组内平均奖励
                    r_std = group_rewards.std().item()
                    r_max = group_rewards.max().item()
                    r_min = group_rewards.min().item()
                    
                    # 找到最佳response
                    best_idx = group_rewards.argmax().item()
                    best_resp = group_responses[best_idx][1]  # 获取response文本
                    best_fmt_r = group_responses[best_idx][3]
                    best_task_r = group_responses[best_idx][4]
                    
                    # 打印组内信息
                    print(f"\n{'='*100}")
                    print(f"[R{get_rank()}] GRPO Step {step} | loss {true_loss:.4f} | baseline {r_mean:.4f} | ent {total_entropy_scalar:.4f}")
                    print(f"  组内奖励: mean={r_mean:.4f} std={r_std:.4f} max={r_max:.4f} min={r_min:.4f}")
                    print(f"  各response奖励: {[f'{r:.3f}' for r in group_rewards_np]}")
                    print(f"  各response优势: {[f'{adv:.3f}' for adv in advantages.cpu().numpy()]}")
                    
                    print(f"  最佳响应 (idx={best_idx}, r={r_max:.4f}, fmt={best_fmt_r:.3f}, task={best_task_r:.3f}):")
                    print(f"    {best_resp[:200]}...")  # 只显示前200字符
                    
                elif args.optimization_strategy == "rgrpo":
                    true_loss = total_policy_loss_scalar - beta * total_entropy_scalar
                    group_rewards_np = group_rewards.detach().cpu().numpy()
                    r_mean = baseline.item()
                    r_std = group_rewards.std().item()
                    r_max = group_rewards.max().item()
                    r_min = group_rewards.min().item()
                    
                    print(f"\n{'='*100}")
                    print(f"[R{get_rank()}] RGRPO Step {step} | loss {true_loss:.4f} | baseline {r_mean:.4f} | ent {total_entropy_scalar:.4f}")
                    print(f"  组内奖励: mean={r_mean:.4f} std={r_std:.4f} max={r_max:.4f} min={r_min:.4f}")
                    print(f"  各response奖励: {[f'{val:.3f}' for val in group_rewards_np]}")
                    print(f"  各response优势: {[f'{adv:.3f}' for adv in advantages.detach().cpu().numpy()]}")
                    for ridx, entry in enumerate(group_responses):
                        iterations = entry.get("iterations", [])
                        summary = "; ".join(
                            f"it{it['iteration']}: task={it['task_reward']:.3f}, fmt={it['format_reward']:.3f}, ok={it['success']}"
                            for it in iterations
                        )
                        print(f"  样本{ridx}: {summary}")
                        if iterations and "feedback" in iterations[-1]:
                            fb = iterations[-1]["feedback"]
                            print(f"    最新反馈: {fb[:200]}{'...' if len(fb) > 200 else ''}")
                    best_idx = group_rewards.argmax().item()
                    print(f"  最终使用响应 idx={best_idx}: {group_responses[best_idx]['resp'][:200]}...")
                    
                elif args.optimization_strategy == "ppo":
                    true_loss = total_policy_loss_scalar - args.ppo_entropy_coef * total_entropy_scalar
                    
                    # 打印PPO信息
                    print(f"\n{'='*100}")
                    print(f"[R{get_rank()}] PPO Step {step} | loss {true_loss:.4f} | reward {r:.4f} | ent {total_entropy_scalar:.4f}")
                    print(f"  响应: {resp[:]}")  # 只显示前200字符
                
                update_flag = "✓" if accumulation_counter % gradient_accumulation_steps == 0 else "○"
                
                if step>0:
                    avg_time_per_step = elapsed/step
                    estimated_total_time = avg_time_per_step*length
                    remaining_time = estimated_total_time - elapsed
                    print(f"  进度: [{step}/{length}] | Step时间: {dt:.2f}s | 已用: {elapsed:.1f}s | 预计剩余: {format_time(remaining_time)}")
                else:
                    print(f"  进度: [{step}/{length}] | Step时间: {dt:.2f}s | 已用: {elapsed:.1f}s")
                
                print(f"{'='*100}\n")
                
                last_log_time = now
                
                # TensorBoard记录
                if writer is not None:
                    writer.add_scalar('loss/total', true_loss, step)
                    writer.add_scalar('loss/policy', total_policy_loss_scalar * gradient_accumulation_steps, step)
                    writer.add_scalar('loss/entropy', total_entropy_scalar, step)
                    
                    if args.optimization_strategy == "grpo":
                        writer.add_scalar('reward/mean', r_mean, step)
                        writer.add_scalar('reward/std', r_std, step)
                        writer.add_scalar('reward/max', r_max, step)
                        writer.add_scalar('reward/min', r_min, step)
                        writer.add_scalar('reward/format_best', best_fmt_r, step)
                        writer.add_scalar('reward/task_best', best_task_r, step)
                    elif args.optimization_strategy == "ppo":
                        writer.add_scalar('reward/current', r, step)

            # checkpoint
            if is_main_process() and step and step % args.save_interval==0:
                ckpt = os.path.join(args.output_dir, f'step_{step}')
                os.makedirs(ckpt, exist_ok=True)
                unwrap(model).save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                torch.save(optimizer.state_dict(), os.path.join(ckpt,'optimizer.pt'))
                with open(os.path.join(ckpt,'step.txt'),'w') as f: f.write(str(step))
            
            # 清理数据，释放显存
            # 🔥 清理所有step相关变量
            if args.optimization_strategy == "grpo":
                del group_responses, group_rewards, advantages
            elif args.optimization_strategy == "rgrpo":
                del group_responses, group_rewards, advantages
            elif args.optimization_strategy == "ppo":
                # PPO的清理在训练循环中已经完成
                pass
            
            # 🔥 强制同步并清理显存
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # 🔥 每个step结束后，如果是DDP，同步所有进程
            if use_ddp:
                torch.distributed.barrier()
            
            step += 1

    barrier()
    if is_main_process():
        final_ckpt = os.path.join(args.output_dir, 'final')
        os.makedirs(final_ckpt, exist_ok=True)
        unwrap(model).save_pretrained(final_ckpt)
        tokenizer.save_pretrained(final_ckpt)
        if writer: writer.close()
        print(f"🎉 Training done! Saved to {final_ckpt}")

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
