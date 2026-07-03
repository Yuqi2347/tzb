
import torch
import torch.nn as nn
import deepspeed
from typing import Dict, Union, Optional, List, Tuple, Any
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer import *
# from transformers.integrations import is_deepspeed_zero3_enabled  # 旧版本没有
# 兼容性处理
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    def is_deepspeed_zero3_enabled():
        return False


class Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "target" in inputs:
            labels = inputs.pop("target")
            inputs["labels"] = labels
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if not self.args.use_lora:
            outputs = self.model(data=inputs, use_cache=False)
        else:
            with self.model._enable_peft_forward_hooks(**inputs):
                outputs = self.model.base_model(data=inputs, use_cache=False)

        if labels is not None:
            logits = outputs.logits  # [B, T, V]
            vocab_size = self.model.config.vocab_size
            batch_size, seq_len, _ = logits.size()
            # print(f"labels:{labels}")
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            logits = logits.view(-1, vocab_size)               # [B*T, V]
            labels = labels.view(-1).long().to(logits.device)  # [B*T]
            loss = loss_fct(logits, labels)                    # [B*T]

            # mask掉 -100 的位置
            label_mask = (labels != -100).float()
            loss = loss * label_mask  # [B*T]

            # reshape 到 [B, T]
            loss = loss.view(batch_size, -1)
            label_mask = label_mask.view(batch_size, -1)
            loss_per_sample = loss.sum(dim=1) / label_mask.sum(dim=1).clamp(min=1.0)  # [B]

            # 对 CoT 样本加权
            if "is_cot_sample" in inputs:
                cot_weight = torch.tensor(inputs["is_cot_sample"]).float().to(loss.device)  # [B]
                loss_per_sample = loss_per_sample * (1.0 + cot_weight * 1.0)  # 可调权重系数

            loss = loss_per_sample.mean()  # -> scalar

        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                   for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                        print(f"🔍 Loss: {loss.item():.4f}")
                        print(f"Sample labels: {inputs['labels'][0][:50]}")
                        print(f"Logits shape: {outputs.logits.shape}")

                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            num_items_in_batch (`int`, *optional*):
                The number of items in the batch (for newer transformers versions).

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    # 【关键修复】禁用自定义保存，使用默认保存逻辑
    # 这样可以避免 merge LoRA 导致的模型状态不一致
    def _save_disabled(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # 【已禁用】这个方法不再使用
        model_to_save = self.model
        is_peft_model = hasattr(model_to_save, 'merge_and_unload')
        
        # 只有最终保存才 merge（checkpoint 保存不 merge）
        is_checkpoint = "checkpoint" in output_dir
        merge_lora_weights = False if is_checkpoint else getattr(self.args, 'merge_lora_on_save', True)
        
        if is_peft_model and merge_lora_weights:
            logger.info("🔄 Merging LoRA weights into base model for checkpoint...")
            # 合并LoRA权重（注意：这会创建一个新的模型对象）
            model_to_save = self.model.merge_and_unload()
            logger.info("✅ LoRA weights merged!")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model_to_save, supported_classes):
            if state_dict is None:
                state_dict = model_to_save.state_dict()

            if isinstance(unwrap_model(model_to_save), supported_classes):
                unwrap_model(model_to_save).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model_to_save.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors, max_shard_size="5GB"
            )

        # 保存 tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info("✅ Tokenizer saved!")

        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        logger.info("✅ Training args saved!")
        
        # 清理合并后的模型（如果创建了新对象）
        if is_peft_model and merge_lora_weights and model_to_save is not self.model:
            del model_to_save
            torch.cuda.empty_cache()


  
    # def _save(self, output_dir: Optional[str] = None, state_dict=None):
    #     from peft import PeftModel
    #     from collections import OrderedDict
    #     import torch, os
    #     from transformers import AutoModelForCausalLM

    #     print("=== DEBUG: custom _save is running ===", flush=True)
    #     output_dir = output_dir or self.args.output_dir

    #     # 只在 rank=0 保存
    #     if not self.is_world_process_zero():
    #         print("当前 rank 非 0，跳过保存", flush=True)
    #         return

    #     os.makedirs(output_dir, exist_ok=True)
    #     print(f"Saving model checkpoint to {output_dir}", flush=True)

    #     model_to_save = self.model

    #     # LoRA merge
    #     if isinstance(self.model, PeftModel):
    #         print("检测到 PeftModel，正在合并 LoRA 权重...", flush=True)
    #         merged_model = self.model.merge_and_unload()
    #         self.model = merged_model  # 替换内部模型
    #         model_to_save = merged_model

    #         if self.tokenizer is not None:
    #             model_to_save.resize_token_embeddings(len(self.tokenizer))
    #             print(f"Token embeddings 已扩展到 {len(self.tokenizer)} 个 token", flush=True)

    #     # 获取合并后的权重
    #     if state_dict is None:
    #         state_dict = model_to_save.state_dict()

    #     # ===== 去掉指定前缀 =====
    #     def strip_prefix_if_present(sd, prefix):
    #         return OrderedDict(
    #             (k[len(prefix):] if k.startswith(prefix) else k, v)
    #             for k, v in sd.items()
    #         )

    #     prefix_to_remove = "base_model.model."
    #     state_dict = strip_prefix_if_present(state_dict, prefix_to_remove)
    #     print(f"已移除权重前缀 '{prefix_to_remove}'（若存在）", flush=True)

    #     # 保存模型权重
    #     model_to_save.save_pretrained(
    #         output_dir,
    #         state_dict=state_dict,
    #         safe_serialization=False  # 避免 shared tensor 报错
    #     )
    #     print("模型已保存 ✅", flush=True)

    #     # 保存 tokenizer
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir, safe_serialization=False)
    #         print("Tokenizer 已保存 ✅", flush=True)

    #     # 保存训练参数
    #     torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    #     print("训练参数已保存 ✅", flush=True)

    #     # ===== 验证保存是否正确 =====
    #     try:
    #         print("🔍 开始验证保存的模型能否正常加载...")
    #         _ = AutoModelForCausalLM.from_pretrained(output_dir,trust_remote_code=True)
    #         print("✅ 验证成功：模型加载无报错")
    #     except Exception as e:
    #         print(f"❌ 验证失败：{e}")




