#!/usr/bin/env python3
"""
数据集调试分析脚本
详细分析数据集中的问题，包括图像路径、文件存在性等
"""

import json
import os
import random
from PIL import Image
from collections import defaultdict
import argparse

def analyze_dataset(json_file: str, sample_size: int = 100) -> None:
    """
    分析数据集中的问题
    
    Args:
        json_file: JSON文件路径
        sample_size: 分析样本数量
    """
    print(f"🔍 分析数据集: {json_file}")
    print("=" * 60)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 数据集总大小: {len(data)} 条记录")
    
    # 随机采样进行分析
    if len(data) > sample_size:
        sample_data = random.sample(data, sample_size)
        print(f"🎲 随机采样 {sample_size} 条记录进行分析")
    else:
        sample_data = data
        print(f"📋 分析全部 {len(sample_data)} 条记录")
    
    # 统计信息
    stats = {
        'total_samples': len(sample_data),
        'file_exists': 0,
        'file_not_exists': 0,
        'image_load_success': 0,
        'image_load_failed': 0,
        'path_patterns': defaultdict(int),
        'error_types': defaultdict(int)
    }
    
    # 详细分析每个样本
    print("\n📋 详细分析结果:")
    print("-" * 60)
    
    for i, item in enumerate(sample_data):
        try:
            image_path = item.get('image', '')
            item_id = item.get('id', f'sample_{i}')
            
            print(f"\n样本 {i+1}: {item_id}")
            print(f"  图像路径: {image_path}")
            
            # 检查文件是否存在
            if os.path.exists(image_path):
                stats['file_exists'] += 1
                print(f"  ✅ 文件存在")
                
                # 尝试加载图像
                try:
                    image = Image.open(image_path).convert("RGB")
                    stats['image_load_success'] += 1
                    print(f"  ✅ 图像加载成功: {image.size}")
                except Exception as e:
                    stats['image_load_failed'] += 1
                    stats['error_types'][f'image_load_error: {str(e)[:50]}'] += 1
                    print(f"  ❌ 图像加载失败: {e}")
            else:
                stats['file_not_exists'] += 1
                print(f"  ❌ 文件不存在")
                
                # 分析路径模式
                if '/images_png/' in image_path:
                    stats['path_patterns']['images_png'] += 1
                elif '/dataset/' in image_path:
                    stats['path_patterns']['dataset'] += 1
                elif '/images_png_small/' in image_path:
                    stats['path_patterns']['images_png_small'] += 1
                else:
                    stats['path_patterns']['other'] += 1
                
                # 尝试找到可能的正确路径
                image_name = os.path.basename(image_path)
                possible_paths = find_possible_paths(image_name)
                if possible_paths:
                    print(f"  🔍 可能的正确路径:")
                    for path in possible_paths[:3]:  # 只显示前3个
                        print(f"    - {path}")
                else:
                    print(f"  ❌ 未找到可能的正确路径")
            
        except Exception as e:
            stats['error_types'][f'processing_error: {str(e)[:50]}'] += 1
            print(f"  ❌ 处理样本失败: {e}")
    
    # 输出统计结果
    print("\n" + "=" * 60)
    print("📊 统计结果:")
    print("-" * 60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"文件存在: {stats['file_exists']} ({stats['file_exists']/stats['total_samples']*100:.1f}%)")
    print(f"文件不存在: {stats['file_not_exists']} ({stats['file_not_exists']/stats['total_samples']*100:.1f}%)")
    print(f"图像加载成功: {stats['image_load_success']} ({stats['image_load_success']/stats['total_samples']*100:.1f}%)")
    print(f"图像加载失败: {stats['image_load_failed']} ({stats['image_load_failed']/stats['total_samples']*100:.1f}%)")
    
    print(f"\n📁 路径模式统计:")
    for pattern, count in stats['path_patterns'].items():
        print(f"  {pattern}: {count}")
    
    print(f"\n❌ 错误类型统计:")
    for error_type, count in stats['error_types'].items():
        print(f"  {error_type}: {count}")
    
    # 提供修复建议
    print(f"\n💡 修复建议:")
    if stats['file_not_exists'] > 0:
        print(f"  - 有 {stats['file_not_exists']} 个文件不存在，需要修复路径")
    if stats['image_load_failed'] > 0:
        print(f"  - 有 {stats['image_load_failed']} 个图像加载失败，可能是文件损坏")
    if stats['path_patterns']['images_png'] > 0:
        print(f"  - 有 {stats['path_patterns']['images_png']} 个路径指向 /images_png/，需要检查该目录")
    if stats['path_patterns']['images_png_small'] > 0:
        print(f"  - 有 {stats['path_patterns']['images_png_small']} 个路径指向 /images_png_small/，需要检查该目录")

def find_possible_paths(image_name: str) -> list:
    """
    查找可能的正确路径
    
    Args:
        image_name: 图像文件名
        
    Returns:
        可能的路径列表
    """
    possible_paths = []
    search_dirs = [
        'datasets/',
        'benchmarks/',
        'examples/images/',
        'outputs/images/'
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                if image_name in files:
                    possible_paths.append(os.path.join(root, image_name))
    
    return possible_paths

def check_directory_structure() -> None:
    """
    检查目录结构
    """
    print("\n📁 目录结构检查:")
    print("-" * 60)
    
    directories_to_check = [
        'datasets/',
        'benchmarks/',
        'examples/images/',
        'datasets/RSVQA/',
        'datasets/dota2.0/',
        'datasets/dota2.0/test-dev/'
    ]
    
    for directory in directories_to_check:
        if os.path.exists(directory):
            file_count = len([f for f in os.listdir(directory) if f.endswith('.png')])
            print(f"✅ {directory} (存在, {file_count} 个PNG文件)")
        else:
            print(f"❌ {directory} (不存在)")

def main():
    parser = argparse.ArgumentParser(description='调试分析数据集')
    parser.add_argument('--dataset', '-d', required=True, help='数据集文件路径')
    parser.add_argument('--sample-size', '-s', type=int, default=100, help='分析样本数量')
    parser.add_argument('--check-dirs', action='store_true', help='检查目录结构')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ 错误: 数据集文件不存在: {args.dataset}")
        return
    
    # 检查目录结构
    if args.check_dirs:
        check_directory_structure()
    
    # 分析数据集
    analyze_dataset(args.dataset, args.sample_size)

if __name__ == "__main__":
    main()

