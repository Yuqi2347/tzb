#!/usr/bin/env python3
"""
修复JSON数据中的类型问题
将所有value字段转换为字符串类型
"""

import json
import argparse
import os
from typing import Any, Dict, List

def convert_value_to_string(obj: Any) -> Any:
    """
    递归地将所有value字段转换为字符串
    
    Args:
        obj: 要处理的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key == "value" and isinstance(value, (int, float)):
                # 将value字段的数值转换为字符串
                result[key] = str(value)
            else:
                # 递归处理嵌套对象
                result[key] = convert_value_to_string(value)
        return result
    elif isinstance(obj, list):
        # 递归处理列表中的每个元素
        return [convert_value_to_string(item) for item in obj]
    else:
        # 其他类型保持不变
        return obj

def fix_json_types(input_file: str, output_file: str) -> None:
    """
    修复JSON文件中的类型问题
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"🔍 读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 原始数据大小: {len(data)} 条记录")
    
    # 统计需要修复的记录
    int_value_count = 0
    float_value_count = 0
    
    for item in data:
        if "conversations" in item:
            for conv in item["conversations"]:
                if "value" in conv:
                    if isinstance(conv["value"], int):
                        int_value_count += 1
                    elif isinstance(conv["value"], float):
                        float_value_count += 1
    
    print(f"📋 发现 {int_value_count} 个整数值需要转换")
    print(f"📋 发现 {float_value_count} 个浮点数值需要转换")
    
    # 转换数据类型
    print("🔄 开始转换数据类型...")
    fixed_data = convert_value_to_string(data)
    
    # 验证转换结果
    print("✅ 验证转换结果...")
    remaining_int_count = 0
    remaining_float_count = 0
    
    for item in fixed_data:
        if "conversations" in item:
            for conv in item["conversations"]:
                if "value" in conv:
                    if isinstance(conv["value"], int):
                        remaining_int_count += 1
                    elif isinstance(conv["value"], float):
                        remaining_float_count += 1
    
    if remaining_int_count == 0 and remaining_float_count == 0:
        print("✅ 所有数值类型已成功转换为字符串")
    else:
        print(f"⚠️  仍有 {remaining_int_count} 个整数和 {remaining_float_count} 个浮点数未转换")
    
    # 保存修复后的数据
    print(f"💾 保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    print("🎉 数据类型修复完成!")

def verify_fix(input_file: str, output_file: str) -> None:
    """
    验证修复结果
    
    Args:
        input_file: 原始文件路径
        output_file: 修复后文件路径
    """
    print("\n🔍 验证修复结果...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        fixed_data = json.load(f)
    
    # 检查数据量
    if len(original_data) != len(fixed_data):
        print(f"❌ 数据量不一致! 原始: {len(original_data)}, 修复后: {len(fixed_data)}")
        return
    
    print(f"✅ 数据量一致: {len(fixed_data)} 条记录")
    
    # 检查类型转换
    int_values_found = 0
    for item in fixed_data:
        if "conversations" in item:
            for conv in item["conversations"]:
                if "value" in conv and isinstance(conv["value"], (int, float)):
                    int_values_found += 1
    
    if int_values_found == 0:
        print("✅ 所有value字段都是字符串类型")
    else:
        print(f"⚠️  仍有 {int_values_found} 个value字段不是字符串类型")

def main():
    parser = argparse.ArgumentParser(description='修复JSON数据中的类型问题')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径 (默认为输入文件名_fixed.json)')
    parser.add_argument('--verify', '-v', action='store_true', help='验证修复结果')
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_fixed.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    # 执行修复
    fix_json_types(args.input, args.output)
    
    # 验证结果
    if args.verify:
        verify_fix(args.input, args.output)

if __name__ == "__main__":
    main()

