#!/usr/bin/env python3
"""
数据集打乱脚本
用于打乱JSON格式的训练数据集
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any

def shuffle_dataset(input_file: str, output_file: str, seed: int = 42) -> None:
    """
    打乱数据集文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        seed: 随机种子
    """
    print(f"正在读取数据集: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集大小: {len(data)} 条记录")
    
    # 设置随机种子以确保可重现性
    random.seed(seed)
    
    # 打乱数据
    print("正在打乱数据...")
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 保存打乱后的数据
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shuffled_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 数据集打乱完成!")
    print(f"原始文件: {input_file}")
    print(f"打乱后文件: {output_file}")
    print(f"使用随机种子: {seed}")

def verify_shuffle(original_file: str, shuffled_file: str) -> None:
    """
    验证打乱结果
    
    Args:
        original_file: 原始文件路径
        shuffled_file: 打乱后文件路径
    """
    print("\n🔍 验证打乱结果...")
    
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(shuffled_file, 'r', encoding='utf-8') as f:
        shuffled_data = json.load(f)
    
    # 检查数据量是否一致
    if len(original_data) != len(shuffled_data):
        print("❌ 数据量不一致!")
        return
    
    # 检查是否真的被打乱了
    first_original_id = original_data[0].get('id', 'unknown')
    first_shuffled_id = shuffled_data[0].get('id', 'unknown')
    
    if first_original_id == first_shuffled_id:
        print("⚠️  警告: 第一个样本相同，可能没有被打乱")
    else:
        print("✅ 数据已成功打乱")
    
    print(f"原始数据第一个样本ID: {first_original_id}")
    print(f"打乱后第一个样本ID: {first_shuffled_id}")

def main():
    parser = argparse.ArgumentParser(description='打乱JSON格式的数据集')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径 (默认为输入文件名_shuffled.json)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--verify', '-v', action='store_true', help='验证打乱结果')
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_shuffled.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 执行打乱
    shuffle_dataset(args.input, args.output, args.seed)
    
    # 验证结果
    if args.verify:
        verify_shuffle(args.input, args.output)

if __name__ == "__main__":
    main()

