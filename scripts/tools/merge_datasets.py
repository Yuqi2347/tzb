#!/usr/bin/env python3
"""
合并并打散数据集脚本
将 xxwdata.json 和 dotav2.json 合并成一个打散的数据集
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any

def merge_and_shuffle_datasets(
    dataset1_path: str, 
    dataset2_path: str, 
    output_path: str, 
    seed: int = 42
) -> None:
    """
    合并并打散两个数据集
    
    Args:
        dataset1_path: 第一个数据集路径
        dataset2_path: 第二个数据集路径  
        output_path: 输出文件路径
        seed: 随机种子
    """
    print(f"📂 读取数据集1: {dataset1_path}")
    with open(dataset1_path, 'r', encoding='utf-8') as f:
        dataset1 = json.load(f)
    
    print(f"📂 读取数据集2: {dataset2_path}")
    with open(dataset2_path, 'r', encoding='utf-8') as f:
        dataset2 = json.load(f)
    
    print(f"📊 数据集1大小: {len(dataset1)} 条记录")
    print(f"📊 数据集2大小: {len(dataset2)} 条记录")
    
    # 合并数据集
    print("🔄 合并数据集...")
    merged_data = dataset1 + dataset2
    print(f"📊 合并后大小: {len(merged_data)} 条记录")
    
    # 设置随机种子
    random.seed(seed)
    
    # 打散数据
    print("🔀 打散数据集...")
    shuffled_data = merged_data.copy()
    random.shuffle(shuffled_data)
    
    # 保存结果
    print(f"💾 保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(shuffled_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 数据集合并和打散完成!")
    print(f"📁 输出文件: {output_path}")
    print(f"🎲 使用随机种子: {seed}")

def verify_merge(original1_path: str, original2_path: str, merged_path: str) -> None:
    """
    验证合并结果
    
    Args:
        original1_path: 原始数据集1路径
        original2_path: 原始数据集2路径
        merged_path: 合并后数据集路径
    """
    print("\n🔍 验证合并结果...")
    
    with open(original1_path, 'r', encoding='utf-8') as f:
        original1 = json.load(f)
    
    with open(original2_path, 'r', encoding='utf-8') as f:
        original2 = json.load(f)
    
    with open(merged_path, 'r', encoding='utf-8') as f:
        merged = json.load(f)
    
    # 检查数据量
    expected_size = len(original1) + len(original2)
    if len(merged) != expected_size:
        print(f"❌ 数据量不一致! 期望: {expected_size}, 实际: {len(merged)}")
        return
    
    print(f"✅ 数据量正确: {len(merged)} 条记录")
    
    # 检查是否真的被打散了
    first_original1_id = original1[0].get('id', 'unknown')
    first_original2_id = original2[0].get('id', 'unknown')
    first_merged_id = merged[0].get('id', 'unknown')
    
    print(f"📋 原始数据集1第一个样本ID: {first_original1_id}")
    print(f"📋 原始数据集2第一个样本ID: {first_original2_id}")
    print(f"📋 合并后第一个样本ID: {first_merged_id}")
    
    if first_merged_id in [first_original1_id, first_original2_id]:
        print("⚠️  警告: 第一个样本来自原始数据集，可能没有完全打散")
    else:
        print("✅ 数据已成功打散")

def main():
    parser = argparse.ArgumentParser(description='合并并打散两个JSON数据集')
    parser.add_argument('--dataset1', '-d1', required=True, help='第一个数据集路径')
    parser.add_argument('--dataset2', '-d2', required=True, help='第二个数据集路径')
    parser.add_argument('--output', '-o', help='输出文件路径 (默认为merged_dataset.json)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--verify', '-v', action='store_true', help='验证合并结果')
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output is None:
        args.output = "merged_dataset.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(args.dataset1):
        print(f"❌ 错误: 数据集1不存在: {args.dataset1}")
        return
    
    if not os.path.exists(args.dataset2):
        print(f"❌ 错误: 数据集2不存在: {args.dataset2}")
        return
    
    # 执行合并和打散
    merge_and_shuffle_datasets(args.dataset1, args.dataset2, args.output, args.seed)
    
    # 验证结果
    if args.verify:
        verify_merge(args.dataset1, args.dataset2, args.output)

if __name__ == "__main__":
    main()

