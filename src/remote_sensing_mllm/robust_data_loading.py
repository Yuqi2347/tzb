#!/usr/bin/env python3
"""
健壮的数据加载配置
为训练过程提供更稳定的数据加载
"""

import json
import os
import random
import time
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustDataLoader:
    """
    健壮的数据加载器
    """
    
    def __init__(self, json_file: str, max_retries: int = 3, retry_delay: float = 0.1):
        """
        初始化数据加载器
        
        Args:
            json_file: JSON文件路径
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.json_file = json_file
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.data = self._load_data()
        
    def _load_data(self):
        """加载数据"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载数据集: {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def get_sample(self, index: int):
        """
        获取样本，带重试机制
        
        Args:
            index: 样本索引
            
        Returns:
            样本数据
        """
        for attempt in range(self.max_retries):
            try:
                if index >= len(self.data):
                    # 如果索引超出范围，随机选择一个样本
                    index = random.randint(0, len(self.data) - 1)
                
                item = self.data[index]
                image_path = item.get('image', '')
                
                # 验证图像文件
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"图像文件不存在: {image_path}")
                
                # 尝试加载图像
                image = Image.open(image_path).convert("RGB")
                
                return {
                    'id': item.get('id', f'sample_{index}'),
                    'image': image,
                    'conversations': item.get('conversations', []),
                    'image_path': image_path
                }
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"获取样本 {index} 失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                    # 随机选择另一个样本
                    index = random.randint(0, len(self.data) - 1)
                else:
                    logger.error(f"获取样本 {index} 最终失败: {e}")
                    # 返回一个默认样本
                    return self._get_fallback_sample()
    
    def _get_fallback_sample(self):
        """获取备用样本"""
        # 返回第一个样本作为备用
        try:
            item = self.data[0]
            image_path = item.get('image', '')
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                return {
                    'id': item.get('id', 'fallback_sample'),
                    'image': image,
                    'conversations': item.get('conversations', []),
                    'image_path': image_path
                }
        except:
            pass
        
        # 如果连备用样本都失败，返回空样本
        return {
            'id': 'empty_sample',
            'image': None,
            'conversations': [],
            'image_path': ''
        }
    
    def get_random_sample(self):
        """获取随机样本"""
        index = random.randint(0, len(self.data) - 1)
        return self.get_sample(index)
    
    def validate_dataset(self, sample_size: int = 100):
        """
        验证数据集
        
        Args:
            sample_size: 验证样本数量
            
        Returns:
            验证结果
        """
        logger.info(f"开始验证数据集，样本数量: {sample_size}")
        
        success_count = 0
        error_count = 0
        error_samples = []
        
        # 随机选择样本进行验证
        indices = random.sample(range(len(self.data)), min(sample_size, len(self.data)))
        
        for i, index in enumerate(indices):
            try:
                sample = self.get_sample(index)
                if sample['image'] is not None:
                    success_count += 1
                else:
                    error_count += 1
                    error_samples.append(index)
            except Exception as e:
                error_count += 1
                error_samples.append(index)
                logger.warning(f"验证样本 {index} 失败: {e}")
        
        result = {
            'total_samples': len(indices),
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_count / len(indices) * 100,
            'error_samples': error_samples
        }
        
        logger.info(f"验证完成: 成功 {success_count}/{len(indices)} ({result['success_rate']:.1f}%)")
        
        return result

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='健壮的数据加载测试')
    parser.add_argument('--dataset', '-d', required=True, help='数据集文件路径')
    parser.add_argument('--validate', '-v', action='store_true', help='验证数据集')
    parser.add_argument('--samples', '-s', type=int, default=100, help='验证样本数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ 错误: 数据集文件不存在: {args.dataset}")
        return
    
    # 创建数据加载器
    loader = RobustDataLoader(args.dataset)
    
    if args.validate:
        # 验证数据集
        result = loader.validate_dataset(args.samples)
        
        print(f"\n📊 验证结果:")
        print(f"总样本数: {result['total_samples']}")
        print(f"成功: {result['success_count']}")
        print(f"失败: {result['error_count']}")
        print(f"成功率: {result['success_rate']:.1f}%")
        
        if result['error_samples']:
            print(f"失败的样本索引: {result['error_samples'][:10]}...")
    else:
        # 测试随机样本获取
        print("🎲 测试随机样本获取:")
        for i in range(5):
            sample = loader.get_random_sample()
            print(f"样本 {i+1}: {sample['id']} - {sample['image_path']}")

if __name__ == "__main__":
    main()

