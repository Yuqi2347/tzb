#!/usr/bin/env python3
"""
合并 LoRA adapter 到基础模型，生成完整模型
"""
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import argparse
import os

def merge_lora_checkpoint(base_model_path, checkpoint_path, output_path):
    """
    合并 LoRA checkpoint
    
    Args:
        base_model_path: 基础模型路径
        checkpoint_path: LoRA adapter checkpoint 路径
        output_path: 输出合并后的模型路径
    """
    print("="*60)
    print("🔄 Merging LoRA adapter into base model")
    print("="*60)
    
    # 加载基础模型
    print(f"\n1️⃣  Loading base model from: {base_model_path}")
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    print(f"   Base model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 加载 tokenizer（从基础模型）
    print(f"\n2️⃣  Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 动态添加新 token（和训练时一样）
    tokens_to_add = ["<think>", "</think>", "<answer>", "</answer>"]
    new_tokens_needed = []
    for token in tokens_to_add:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            new_tokens_needed.append(token)
    
    if new_tokens_needed:
        print(f"   Adding {len(new_tokens_needed)} tokens: {new_tokens_needed}")
        special_tokens = {"additional_special_tokens": new_tokens_needed}
        num_new = tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"   ✅ Tokenizer expanded to {len(tokenizer)} tokens")
    
    # 加载 LoRA adapter
    print(f"\n3️⃣  Loading LoRA adapter from: {checkpoint_path}")
    if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print("   ✅ LoRA adapter loaded")
        
        # 合并权重
        print(f"\n4️⃣  Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("   ✅ LoRA merged!")
    else:
        print("   ⚠️  No adapter found, using base model")
    
    # 保存合并后的模型
    print(f"\n5️⃣  Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # 【关键】使用 safetensors 格式，文件名为 model-*.safetensors
    try:
        model.save_pretrained(
            output_path,
            safe_serialization=True,  # 使用 safetensors 格式
            max_shard_size="5GB"
        )
        print("   ✅ Saved as safetensors format")
    except Exception as e:
        print(f"   ⚠️  Failed to save as safetensors: {e}")
        print("   Trying .bin format...")
        model.save_pretrained(
            output_path,
            safe_serialization=False,
            max_shard_size="5GB"
        )
    
    tokenizer.save_pretrained(output_path)
    
    print("\n" + "="*60)
    print(f"✅ Merged model saved successfully!")
    print(f"📁 Output: {output_path}")
    print("="*60)
    
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base Hugging Face model.")
    parser.add_argument("--base-model", required=True, help="Base model directory.")
    parser.add_argument("--checkpoint", required=True, help="LoRA adapter checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for the merged model.")
    args = parser.parse_args()

    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}\n")

    model, tokenizer = merge_lora_checkpoint(args.base_model, args.checkpoint, args.output_dir)
    # 复制文件
    import shutil

    # 复制推理需要的额外文件
    files_to_copy = [
        'image_processing_fm9gv.py',
        'processing_fm9gv.py', 
        'preprocessor_config.json'
    ]

    for filename in files_to_copy:
        src = os.path.join(args.base_model, filename)
        dst = os.path.join(args.output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"   Copied {filename}")
    print("\n使用方法:")
    print(f"  model = AutoModel.from_pretrained('{args.output_dir}', trust_remote_code=True)")

