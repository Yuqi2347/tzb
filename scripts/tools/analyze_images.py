#!/usr/bin/env python3
"""
图像数据集分析工具
分析遥感图像数据集目录中的图像数据
支持多种格式：PNG, TIF, JPG等
"""

import os
import sys
import glob
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImageAnalyzer:
    def __init__(self, base_path="datasets/valid/images"):
        self.base_path = Path(base_path)
        self.results = {}
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
        
    def scan_directory_structure(self):
        """扫描目录结构"""
        print("🔍 扫描目录结构...")
        structure = {}
        
        for dataset_dir in self.base_path.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                structure[dataset_name] = {
                    'path': str(dataset_dir),
                    'files': [],
                    'subdirs': []
                }
                
                # 扫描文件
                for file_path in dataset_dir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                        structure[dataset_name]['files'].append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'size': file_path.stat().st_size,
                            'extension': file_path.suffix.lower()
                        })
                    elif file_path.is_dir():
                        structure[dataset_name]['subdirs'].append(file_path.name)
        
        self.results['structure'] = structure
        return structure
    
    def analyze_image_properties(self, max_images_per_dataset=50):
        """分析图像属性"""
        print("📊 分析图像属性...")
        analysis = {}
        
        for dataset_name, dataset_info in self.results['structure'].items():
            print(f"  分析数据集: {dataset_name}")
            analysis[dataset_name] = {
                'total_files': len(dataset_info['files']),
                'image_properties': [],
                'size_stats': {},
                'format_distribution': Counter(),
                'errors': []
            }
            
            # 随机采样分析（避免分析过多图像）
            files_to_analyze = dataset_info['files'][:max_images_per_dataset]
            
            sizes = []
            formats = []
            
            for i, file_info in enumerate(files_to_analyze):
                try:
                    with Image.open(file_info['path']) as img:
                        properties = {
                            'filename': file_info['name'],
                            'size': file_info['size'],
                            'width': img.width,
                            'height': img.height,
                            'format': img.format,
                            'mode': img.mode,
                            'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                        }
                        
                        # 计算统计信息
                        if img.mode == 'RGB':
                            stat = ImageStat.Stat(img)
                            properties.update({
                                'mean_r': stat.mean[0],
                                'mean_g': stat.mean[1], 
                                'mean_b': stat.mean[2],
                                'std_r': stat.stddev[0],
                                'std_g': stat.stddev[1],
                                'std_b': stat.stddev[2]
                            })
                        
                        analysis[dataset_name]['image_properties'].append(properties)
                        sizes.append((img.width, img.height))
                        formats.append(img.format)
                        
                except Exception as e:
                    analysis[dataset_name]['errors'].append({
                        'filename': file_info['name'],
                        'error': str(e)
                    })
            
            # 计算统计信息
            if sizes:
                widths, heights = zip(*sizes)
                analysis[dataset_name]['size_stats'] = {
                    'width': {'min': min(widths), 'max': max(widths), 'mean': np.mean(widths)},
                    'height': {'min': min(heights), 'max': max(heights), 'mean': np.mean(heights)},
                    'aspect_ratios': [w/h for w, h in sizes]
                }
            
            analysis[dataset_name]['format_distribution'] = Counter(formats)
        
        self.results['analysis'] = analysis
        return analysis
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("📋 生成总结报告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'datasets': {}
        }
        
        for dataset_name, analysis in self.results['analysis'].items():
            report['datasets'][dataset_name] = {
                'total_files': analysis['total_files'],
                'analyzed_files': len(analysis['image_properties']),
                'errors': len(analysis['errors']),
                'formats': dict(analysis['format_distribution']),
                'size_statistics': analysis['size_stats']
            }
        
        self.results['summary'] = report
        return report
    
    def create_visualizations(self, output_dir="outputs/image_analysis"):
        """创建可视化图表"""
        print("📈 创建可视化图表...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 数据集文件数量对比
        self._plot_dataset_comparison(output_path)
        
        # 2. 图像尺寸分布
        self._plot_size_distribution(output_path)
        
        # 3. 格式分布
        self._plot_format_distribution(output_path)
        
        # 4. 样本图像展示
        self._plot_sample_images(output_path)
        
        print(f"📁 可视化结果保存到: {output_path}")
        return output_path
    
    def _plot_dataset_comparison(self, output_path):
        """绘制数据集对比图"""
        datasets = list(self.results['analysis'].keys())
        file_counts = [self.results['analysis'][d]['total_files'] for d in datasets]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(datasets, file_counts, color='skyblue', alpha=0.7)
        plt.title('各数据集图像文件数量对比', fontsize=16, fontweight='bold')
        plt.xlabel('数据集', fontsize=12)
        plt.ylabel('文件数量', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # 在柱子上添加数值标签
        for bar, count in zip(bars, file_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(file_counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_size_distribution(self, output_path):
        """绘制图像尺寸分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (dataset_name, analysis) in enumerate(self.results['analysis'].items()):
            if i >= 4:  # 最多显示4个数据集
                break
                
            if not analysis['image_properties']:
                continue
                
            ax = axes[i]
            widths = [img['width'] for img in analysis['image_properties']]
            heights = [img['height'] for img in analysis['image_properties']]
            
            ax.scatter(widths, heights, alpha=0.6, s=20)
            ax.set_title(f'{dataset_name} 图像尺寸分布')
            ax.set_xlabel('宽度 (像素)')
            ax.set_ylabel('高度 (像素)')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for j in range(i+1, 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_format_distribution(self, output_path):
        """绘制格式分布饼图"""
        all_formats = Counter()
        for analysis in self.results['analysis'].values():
            all_formats.update(analysis['format_distribution'])
        
        if all_formats:
            plt.figure(figsize=(10, 8))
            labels = list(all_formats.keys())
            sizes = list(all_formats.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                             startangle=90, textprops={'fontsize': 12})
            
            plt.title('图像格式分布', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(output_path / 'format_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_sample_images(self, output_path):
        """展示样本图像"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        sample_count = 0
        for dataset_name, analysis in self.results['analysis'].items():
            if sample_count >= 6:  # 最多显示6个样本
                break
                
            if analysis['image_properties']:
                # 选择第一个图像作为样本
                sample_img = analysis['image_properties'][0]
                try:
                    img = Image.open(sample_img['filename'])
                    axes[sample_count].imshow(img)
                    axes[sample_count].set_title(f'{dataset_name}\n{sample_img["width"]}x{sample_img["height"]}')
                    axes[sample_count].axis('off')
                    sample_count += 1
                except:
                    continue
        
        # 隐藏多余的子图
        for j in range(sample_count, 6):
            axes[j].set_visible(False)
        
        plt.suptitle('样本图像展示', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_file="outputs/image_analysis/results.json"):
        """保存分析结果到JSON文件"""
        print(f"💾 保存结果到: {output_file}")
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换所有numpy类型
        def clean_results(results):
            if isinstance(results, dict):
                return {k: clean_results(v) for k, v in results.items()}
            elif isinstance(results, list):
                return [clean_results(item) for item in results]
            else:
                return convert_numpy(results)
        
        cleaned_results = clean_results(self.results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        """打印分析总结"""
        print("\n" + "="*60)
        print("📊 图像数据集分析总结")
        print("="*60)
        
        total_files = sum(analysis['total_files'] for analysis in self.results['analysis'].values())
        total_analyzed = sum(len(analysis['image_properties']) for analysis in self.results['analysis'].values())
        total_errors = sum(len(analysis['errors']) for analysis in self.results['analysis'].values())
        
        print(f"📁 基础路径: {self.base_path}")
        print(f"📈 总文件数: {total_files}")
        print(f"🔍 已分析: {total_analyzed}")
        print(f"❌ 错误数: {total_errors}")
        print()
        
        for dataset_name, analysis in self.results['analysis'].items():
            print(f"📂 {dataset_name}:")
            print(f"   - 文件数量: {analysis['total_files']}")
            print(f"   - 已分析: {len(analysis['image_properties'])}")
            print(f"   - 错误数: {len(analysis['errors'])}")
            
            if analysis['size_stats']:
                stats = analysis['size_stats']
                print(f"   - 尺寸范围: {stats['width']['min']}x{stats['height']['min']} ~ {stats['width']['max']}x{stats['height']['max']}")
                print(f"   - 平均尺寸: {stats['width']['mean']:.0f}x{stats['height']['mean']:.0f}")
            
            if analysis['format_distribution']:
                formats = dict(analysis['format_distribution'])
                print(f"   - 格式分布: {formats}")
            print()

def main():
    """主函数"""
    print("🚀 开始图像数据集分析...")
    
    # 创建分析器
    analyzer = ImageAnalyzer()
    
    # 执行分析步骤
    analyzer.scan_directory_structure()
    analyzer.analyze_image_properties()
    analyzer.generate_summary_report()
    
    # 创建可视化
    output_dir = analyzer.create_visualizations()
    
    # 保存结果
    analyzer.save_results()
    
    # 打印总结
    analyzer.print_summary()
    
    print(f"\n✅ 分析完成！结果保存在:")
    print("   - JSON报告: outputs/image_analysis/results.json")
    print(f"   - 可视化图表: {output_dir}")

if __name__ == "__main__":
    main()
