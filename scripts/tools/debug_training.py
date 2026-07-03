#!/usr/bin/env python3
"""
训练过程调试脚本
模拟训练过程中的数据加载，找出具体的错误原因
"""

import json
import os
import random
from PIL import Image
import traceback

def simulate_training_data_loading(json_file: str, num_samples: int = 20) -> None:
    """
    模拟训练过程中的数据加载
    
    Args:
        json_file: JSON文件路径
        num_samples: 测试样本数量
    """
    print(f"🔍 模拟训练数据加载: {json_file}")
    print("=" * 60)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 数据集大小: {len(data)} 条记录")
    
    # 模拟训练过程中的数据加载
    success_count = 0
    error_count = 0
    error_details = []
    
    # 模拟transform（简化版本）
    def simple_transform(image):
        # 这里只是模拟，实际训练时会使用torchvision.transforms
        return image
    
    for i in range(min(num_samples, len(data))):
        try:
            print(f"\n🔄 处理样本 {i+1}/{num_samples}")
            
            # 模拟SupervisedDataset.__getitem__方法
            item = data[i]
            image_path = item.get('image', '')
            conversations = item.get('conversations', [])
            
            print(f"  📁 图像路径: {image_path}")
            print(f"  💬 对话数量: {len(conversations)}")
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            # 加载图像
            print(f"  🖼️  加载图像...")
            image = Image.open(image_path).convert("RGB")
            print(f"  ✅ 图像加载成功: {image.size}")
            
            # 应用transform
            print(f"  🔄 应用transform...")
            transformed_image = simple_transform(image)
            print(f"  ✅ Transform成功: {transformed_image.size}")
            
            # 模拟conversation处理
            print(f"  💬 处理对话...")
            for j, conv in enumerate(conversations):
                role = conv.get('from', '')
                value = conv.get('value', '')
                print(f"    {j+1}. {role}: {value[:50]}...")
            
            success_count += 1
            print(f"  ✅ 样本 {i+1} 处理成功")
            
        except Exception as e:
            error_count += 1
            error_msg = f"样本 {i+1} 处理失败: {str(e)}"
            error_details.append(error_msg)
            print(f"  ❌ {error_msg}")
            print(f"  📋 错误详情: {traceback.format_exc()}")
    
    # 输出统计结果
    print("\n" + "=" * 60)
    print("📊 模拟训练结果:")
    print("-" * 60)
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {error_count}")
    print(f"📁 总样本: {num_samples}")
    
    if error_count > 0:
        print(f"\n❌ 错误详情:")
        for error in error_details:
            print(f"  - {error}")
    else:
        print(f"\n🎉 所有样本都处理成功！")

def check_memory_usage() -> None:
    """
    检查内存使用情况
    """
    print("\n💾 内存使用情况:")
    print("-" * 60)
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"当前进程内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
        print(f"虚拟内存使用: {memory_info.vms / 1024 / 1024:.2f} MB")
    except ImportError:
        print("psutil未安装，无法检查内存使用情况")

def check_cuda_availability() -> None:
    """
    检查CUDA可用性
    """
    print("\n🖥️  CUDA检查:")
    print("-" * 60)
    
    try:
        import torch
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"CUDA设备名称: {torch.cuda.get_device_name()}")
            print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB")
    except ImportError:
        print("torch未安装，无法检查CUDA")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='调试训练过程')
    parser.add_argument('--dataset', '-d', required=True, help='数据集文件路径')
    parser.add_argument('--samples', '-s', type=int, default=20, help='测试样本数量')
    parser.add_argument('--check-memory', action='store_true', help='检查内存使用')
    parser.add_argument('--check-cuda', action='store_true', help='检查CUDA')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ 错误: 数据集文件不存在: {args.dataset}")
        return
    
    # 检查系统资源
    if args.check_memory:
        check_memory_usage()
    
    if args.check_cuda:
        check_cuda_availability()
    
    # 模拟训练数据加载
    simulate_training_data_loading(args.dataset, args.samples)

if __name__ == "__main__":
    main()
