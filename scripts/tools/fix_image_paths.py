#!/usr/bin/env python3
"""
修复数据集中的图像路径问题
"""

import json
import os
import argparse
from typing import Dict, List, Any

def find_correct_image_path(image_name: str, search_dirs: List[str]) -> str:
    """
    在指定目录中查找图像文件
    
    Args:
        image_name: 图像文件名
        search_dirs: 搜索目录列表
        
    Returns:
        找到的图像完整路径，如果未找到返回None
    """
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # 在目录中递归查找文件
            for root, dirs, files in os.walk(search_dir):
                if image_name in files:
                    return os.path.join(root, image_name)
    return None

def fix_image_paths(input_file: str, output_file: str, search_dirs: List[str]) -> None:
    """
    修复数据集中的图像路径
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        search_dirs: 图像搜索目录列表
    """
    print(f"正在读取数据集: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集大小: {len(data)} 条记录")
    
    fixed_count = 0
    not_found_count = 0
    already_correct_count = 0
    
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(data)}")
            
        image_path = item.get('image', '')
        
        # 检查路径是否已经正确
        if os.path.exists(image_path):
            already_correct_count += 1
            continue
            
        # 提取文件名
        image_name = os.path.basename(image_path)
        
        # 查找正确的路径
        correct_path = find_correct_image_path(image_name, search_dirs)
        
        if correct_path:
            item['image'] = correct_path
            fixed_count += 1
        else:
            print(f"⚠️  未找到图像: {image_name}")
            not_found_count += 1
    
    # 保存修复后的数据
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n📊 修复统计:")
    print(f"✅ 已修复: {fixed_count}")
    print(f"✅ 原本正确: {already_correct_count}")
    print(f"❌ 未找到: {not_found_count}")
    print(f"📁 总记录数: {len(data)}")

def main():
    parser = argparse.ArgumentParser(description='修复数据集中的图像路径')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径 (默认为输入文件名_fixed.json)')
    parser.add_argument('--search-dirs', '-d', nargs='+', 
                       default=['datasets/', 'benchmarks/'],
                       help='图像搜索目录列表')
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_fixed.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 检查搜索目录
    print("🔍 搜索目录:")
    for search_dir in args.search_dirs:
        if os.path.exists(search_dir):
            print(f"✅ {search_dir}")
        else:
            print(f"❌ {search_dir} (不存在)")
    
    # 执行修复
    fix_image_paths(args.input, args.output, args.search_dirs)

if __name__ == "__main__":
    main()

