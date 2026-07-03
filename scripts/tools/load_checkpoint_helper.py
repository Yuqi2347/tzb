#!/usr/bin/env python3
"""
辅助脚本：正确加载训练的 checkpoint
处理 token 添加和版本兼容问题
"""
import torch
import argparse
from transformers import AutoModel, AutoTokenizer

def load_trained_model(checkpoint_path, base_model_path=None):
    """
    加载训练后的模型，自动处理 token 添加
    
    Args:
        checkpoint_path: checkpoint 路径
        base_model_path: 基础模型路径（如果提供，会从这里加载 tokenizer）
    """
    # 如果提供了基础模型路径，从那里加载 tokenizer（避免版本问题）
    if base_model_path:
        print(f"Loading tokenizer from base model: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    else:
        print(f"Loading tokenizer from checkpoint: {checkpoint_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️  Failed to load tokenizer from checkpoint: {e}")
            print("Please provide base_model_path")
            raise
    
    # 动态添加新 token（和训练时相同）
    tokens_to_add = ["<think>", "</think>", "<answer>", "</answer>"]
    new_tokens_needed = []
    for token in tokens_to_add:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            new_tokens_needed.append(token)
    
    if new_tokens_needed:
        print(f"➕ Adding {len(new_tokens_needed)} tokens: {new_tokens_needed}")
        special_tokens = {"additional_special_tokens": new_tokens_needed}
        tokenizer.add_special_tokens(special_tokens)
    
    # 加载模型
    print(f"Loading model from: {checkpoint_path}")
    model = AutoModel.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # 确保所有层都是 FP16
    for name, param in model.named_parameters():
        if param.dtype not in [torch.float16, torch.int64, torch.int32, torch.int8]:
            param.data = param.data.to(torch.float16)
    
    model = model.eval().cuda()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a trained checkpoint and verify special token IDs.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument("--base-model", default=None, help="Optional base model directory for tokenizer loading.")
    args = parser.parse_args()

    model, tokenizer = load_trained_model(args.checkpoint, args.base_model)
    
    print(f"\nTokenizer vocabulary size: {len(tokenizer)}")
    print(f"Model embedding size: {model.llm.get_input_embeddings().weight.shape[0]}")
    
    # 验证 token IDs
    for token in ["<think>", "</think>", "<answer>", "</answer>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token:<10} -> {token_id}")

