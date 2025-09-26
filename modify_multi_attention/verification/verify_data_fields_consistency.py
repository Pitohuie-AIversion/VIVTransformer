#!/usr/bin/env python3
"""
验证所有相关脚本的数据字段一致性
检查可能存在的字段名不匹配问题
"""

import os
import re
from pathlib import Path

def find_field_usage_patterns():
    """查找所有脚本中的字段使用模式"""
    
    # 要检查的字段模式
    field_patterns = {
        'test_mse': [r"test_mse", r"final_test_mse"],
        'test_r2': [r"test_r2", r"final_test_r2"],
        'parameters': [r"num_parameters", r"parameters"],
        'training_time': [r"train_time", r"training_time"]
    }
    
    # 要检查的文件
    python_files = []
    for root, dirs, files in os.walk('.'):
        # 跳过某些目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'test_visualization_output']]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(os.path.join(root, file))
    
    print("🔍 检查数据字段使用一致性...")
    print(f"📁 检查文件数量: {len(python_files)}")
    
    field_usage = {}
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_fields = {}
            
            # 检查每个字段模式
            for field_group, patterns in field_patterns.items():
                found_patterns = []
                for pattern in patterns:
                    # 查找字段访问模式，如 ['test_mse'] 或 .get('test_mse')
                    matches = re.findall(rf"['\"]({pattern})['\"]", content)
                    if matches:
                        found_patterns.extend(matches)
                
                if found_patterns:
                    file_fields[field_group] = list(set(found_patterns))
            
            if file_fields:
                field_usage[file_path] = file_fields
                
        except Exception as e:
            print(f"❌ 读取文件失败: {file_path} - {e}")
    
    return field_usage

def analyze_field_consistency(field_usage):
    """分析字段使用的一致性"""
    
    print("\n📊 字段使用分析结果:")
    print("=" * 80)
    
    # 统计每个字段组的使用情况
    field_stats = {}
    
    for file_path, file_fields in field_usage.items():
        for field_group, used_fields in file_fields.items():
            if field_group not in field_stats:
                field_stats[field_group] = {}
            
            for field in used_fields:
                if field not in field_stats[field_group]:
                    field_stats[field_group][field] = []
                field_stats[field_group][field].append(file_path)
    
    # 显示统计结果
    inconsistencies = []
    
    for field_group, field_usage_stats in field_stats.items():
        print(f"\n🔸 {field_group.upper()} 字段组:")
        
        if len(field_usage_stats) > 1:
            print("  ⚠️  发现不一致使用:")
            inconsistencies.append(field_group)
            
            for field, files in field_usage_stats.items():
                print(f"    - '{field}' 使用在 {len(files)} 个文件中:")
                for file_path in files[:3]:  # 只显示前3个文件
                    print(f"      • {file_path}")
                if len(files) > 3:
                    print(f"      • ... 还有 {len(files) - 3} 个文件")
        else:
            field_name = list(field_usage_stats.keys())[0]
            file_count = len(field_usage_stats[field_name])
            print(f"  ✅ 一致使用 '{field_name}' 在 {file_count} 个文件中")
    
    return inconsistencies

def suggest_fixes(inconsistencies):
    """建议修复方案"""
    
    if not inconsistencies:
        print("\n🎉 所有字段使用都是一致的！")
        return
    
    print(f"\n🔧 发现 {len(inconsistencies)} 个不一致的字段组，建议修复:")
    print("=" * 80)
    
    fix_suggestions = {
        'test_mse': {
            'preferred': 'final_test_mse',
            'reason': '与fixed_models_test_results.json数据结构一致'
        },
        'test_r2': {
            'preferred': 'final_test_r2', 
            'reason': '与fixed_models_test_results.json数据结构一致'
        },
        'parameters': {
            'preferred': 'parameters',
            'reason': '与fixed_models_test_results.json数据结构一致'
        },
        'training_time': {
            'preferred': 'training_time',
            'reason': '更清晰的字段名'
        }
    }
    
    for field_group in inconsistencies:
        if field_group in fix_suggestions:
            suggestion = fix_suggestions[field_group]
            print(f"\n🔸 {field_group.upper()}:")
            print(f"  推荐使用: '{suggestion['preferred']}'")
            print(f"  原因: {suggestion['reason']}")
            print(f"  修复方法: 使用 .get('{suggestion['preferred']}', data.get('备选字段名', 默认值)) 模式")

def main():
    """主函数"""
    print("🔍 数据字段一致性验证工具")
    print("=" * 80)
    
    # 查找字段使用模式
    field_usage = find_field_usage_patterns()
    
    if not field_usage:
        print("❌ 没有找到任何字段使用")
        return
    
    print(f"✅ 在 {len(field_usage)} 个文件中找到字段使用")
    
    # 分析一致性
    inconsistencies = analyze_field_consistency(field_usage)
    
    # 建议修复方案
    suggest_fixes(inconsistencies)
    
    print("\n" + "=" * 80)
    print("✅ 字段一致性验证完成")
    
    if inconsistencies:
        print("⚠️  建议根据上述建议统一字段使用")
    else:
        print("🎉 所有字段使用都是一致的")

if __name__ == "__main__":
    main()