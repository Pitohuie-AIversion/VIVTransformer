#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成标准化对比报告
收集所有模型的训练、验证、测试损失数据，输出为CSV格式
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def collect_loss_data(base_dir="attention_results"):
    """
    收集所有模型的损失数据
    """
    results = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"警告: 结果目录 {base_dir} 不存在")
        return results
    
    # 遍历所有实验目录（包括debug_unet_finite等）
    for experiment_dir in base_path.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        experiment_name = experiment_dir.name
        print(f"检查实验目录: {experiment_name}")
        
        # 遍历loss配置目录
        for loss_config_dir in experiment_dir.iterdir():
            if not loss_config_dir.is_dir() or not loss_config_dir.name.startswith('loss_config_'):
                continue
                
            loss_config_name = loss_config_dir.name
            
            # 遍历每个loss配置下的注意力机制目录
            for attention_dir in loss_config_dir.iterdir():
                if not attention_dir.is_dir():
                    continue
                    
                attention_type = attention_dir.name
                loss_log_path = attention_dir / "loss_logs" / "loss_log.txt"
            
            if loss_log_path.exists():
                try:
                    # 读取损失日志
                    with open(loss_log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    if len(lines) > 1:  # 至少有标题行和一行数据
                        # 解析最后一行数据（最终epoch的结果）
                        last_line = lines[-1].strip()
                        if last_line:
                            parts = last_line.split(', ')
                            if len(parts) >= 4:
                                epoch = int(parts[0])
                                train_loss = float(parts[1])
                                valid_loss = float(parts[2])
                                test_loss = float(parts[3])
                                
                                results.append({
                                    'experiment': experiment_name,
                                    'loss_config': loss_config_name,
                                    'attention_type': attention_type,
                                    'final_epoch': epoch,
                                    'train_loss': train_loss,
                                    'valid_loss': valid_loss,
                                    'test_loss': test_loss,
                                    'log_path': str(loss_log_path)
                                })
                                print(f"✓ 收集到数据: {experiment_name}/{loss_config_name}/{attention_type}")
                            else:
                                print(f"⚠ 数据格式错误: {loss_log_path}")
                        else:
                            print(f"⚠ 空数据行: {loss_log_path}")
                    else:
                        print(f"⚠ 数据不足: {loss_log_path}")
                        
                except Exception as e:
                    print(f"✗ 读取失败 {loss_log_path}: {e}")
            else:
                print(f"⚠ 日志文件不存在: {loss_log_path}")
    
    return results

def generate_csv_report(results, output_path="model_comparison_report.csv"):
    """
    生成CSV格式的对比报告
    """
    if not results:
        print("没有找到任何有效数据，无法生成报告")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按实验、损失配置和注意力类型排序
    df = df.sort_values(['experiment', 'loss_config', 'attention_type'])
    
    # 保存为CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV报告已生成: {output_path}")
    
    # 显示统计信息
    print(f"\n=== 报告统计 ===")
    print(f"总计模型配置: {len(df)}")
    print(f"损失配置数量: {df['loss_config'].nunique()}")
    print(f"注意力机制数量: {df['attention_type'].nunique()}")
    
    # 显示最佳性能
    if len(df) > 0:
        best_train = df.loc[df['train_loss'].idxmin()]
        best_valid = df.loc[df['valid_loss'].idxmin()]
        best_test = df.loc[df['test_loss'].idxmin()]
        
        print(f"\n=== 最佳性能 ===")
        print(f"最低训练损失: {best_train['train_loss']:.6f} ({best_train['experiment']}/{best_train['loss_config']}/{best_train['attention_type']})")
        print(f"最低验证损失: {best_valid['valid_loss']:.6f} ({best_valid['experiment']}/{best_valid['loss_config']}/{best_valid['attention_type']})")
        print(f"最低测试损失: {best_test['test_loss']:.6f} ({best_test['experiment']}/{best_test['loss_config']}/{best_test['attention_type']})")
    
    return df

def generate_summary_report(df, output_path="model_summary_report.txt"):
    """
    生成文本格式的摘要报告
    """
    if df is None or len(df) == 0:
        return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("模型对比分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 基本统计
        f.write("基本统计信息:\n")
        f.write("-" * 20 + "\n")
        f.write(f"总计模型配置: {len(df)}\n")
        f.write(f"损失配置数量: {df['loss_config'].nunique()}\n")
        f.write(f"注意力机制数量: {df['attention_type'].nunique()}\n\n")
        
        # 损失统计
        f.write("损失统计 (所有模型):\n")
        f.write("-" * 20 + "\n")
        for loss_type in ['train_loss', 'valid_loss', 'test_loss']:
            f.write(f"{loss_type}:\n")
            f.write(f"  平均值: {df[loss_type].mean():.6f}\n")
            f.write(f"  最小值: {df[loss_type].min():.6f}\n")
            f.write(f"  最大值: {df[loss_type].max():.6f}\n")
            f.write(f"  标准差: {df[loss_type].std():.6f}\n\n")
        
        # 最佳模型
        best_valid = df.loc[df['valid_loss'].idxmin()]
        f.write("最佳模型 (基于验证损失):\n")
        f.write("-" * 20 + "\n")
        f.write(f"配置: {best_valid['experiment']}/{best_valid['loss_config']}/{best_valid['attention_type']}\n")
        f.write(f"训练损失: {best_valid['train_loss']:.6f}\n")
        f.write(f"验证损失: {best_valid['valid_loss']:.6f}\n")
        f.write(f"测试损失: {best_valid['test_loss']:.6f}\n")
        f.write(f"训练轮数: {best_valid['final_epoch']}\n\n")
        
        # 按注意力机制分组统计
        f.write("按注意力机制分组统计:\n")
        f.write("-" * 20 + "\n")
        for attention_type in sorted(df['attention_type'].unique()):
            subset = df[df['attention_type'] == attention_type]
            f.write(f"{attention_type}:\n")
            f.write(f"  配置数量: {len(subset)}\n")
            f.write(f"  平均验证损失: {subset['valid_loss'].mean():.6f}\n")
            f.write(f"  最佳验证损失: {subset['valid_loss'].min():.6f}\n\n")
    
    print(f"✓ 摘要报告已生成: {output_path}")

def main():
    print("开始生成模型对比报告...")
    
    # 收集数据
    results = collect_loss_data()
    
    if not results:
        print("未找到任何有效的模型结果数据")
        return
    
    # 生成CSV报告
    df = generate_csv_report(results)
    
    # 生成摘要报告
    generate_summary_report(df)
    
    print("\n报告生成完成！")
    print("文件输出:")
    print("- model_comparison_report.csv (详细数据)")
    print("- model_summary_report.txt (摘要分析)")

if __name__ == "__main__":
    main()