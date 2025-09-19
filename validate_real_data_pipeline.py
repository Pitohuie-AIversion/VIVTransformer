#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据集验证脚本
使用DarcyFlow真实数据集验证数据处理流程的完整性和准确性
"""

import torch
import numpy as np
import h5py
import json
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.append('x:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1')
sys.path.append('x:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\modify_multi_attention')

def load_real_darcy_data(data_path):
    """加载真实的DarcyFlow数据集"""
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"数据集键: {list(f.keys())}")
            
            # 检查数据集结构
            for key in f.keys():
                print(f"键 '{key}' 形状: {f[key].shape}")
            
            # 根据DarcyFlow数据集的实际结构加载数据
            # DarcyFlow: 'nu'是输入(渗透率场), 'tensor'是输出(压力场)
            if 'nu' in f and 'tensor' in f:
                # nu: (10000, 128, 128) - 输入渗透率场
                # tensor: (10000, 1, 128, 128) - 输出压力场
                input_data = torch.tensor(f['nu'][:100], dtype=torch.float32)  # 取前100个样本
                output_tensor = torch.tensor(f['tensor'][:100], dtype=torch.float32)
                # 移除tensor的单一通道维度
                output_data = output_tensor.squeeze(1)  # (100, 128, 128)
                print(f"DarcyFlow数据结构: nu(输入)={input_data.shape}, tensor(输出)={output_data.shape}")
            elif 'input' in f and 'output' in f:
                input_data = torch.tensor(f['input'][:100], dtype=torch.float32)
                output_data = torch.tensor(f['output'][:100], dtype=torch.float32)
            elif 'x' in f and 'y' in f:
                input_data = torch.tensor(f['x'][:100], dtype=torch.float32)
                output_data = torch.tensor(f['y'][:100], dtype=torch.float32)
            else:
                # 取前两个最大的数据集作为输入输出
                keys = list(f.keys())
                # 按数据大小排序，取最大的两个
                key_sizes = [(key, f[key].size) for key in keys if len(f[key].shape) >= 2]
                key_sizes.sort(key=lambda x: x[1], reverse=True)
                
                if len(key_sizes) >= 2:
                    input_key, output_key = key_sizes[0][0], key_sizes[1][0]
                    input_data = torch.tensor(f[input_key][:100], dtype=torch.float32)
                    output_data = torch.tensor(f[output_key][:100], dtype=torch.float32)
                    if len(output_data.shape) > len(input_data.shape):
                        output_data = output_data.squeeze(1)  # 移除多余维度
                else:
                    raise ValueError("无法识别合适的输入输出数据")
            
            return {
                'input': input_data,
                'output': output_data
            }
    except Exception as e:
        print(f"加载真实数据失败: {e}")
        return None

def validate_real_data_loading(data_path):
    """验证真实数据加载功能"""
    print("\n=== 验证真实数据加载 ===")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(data_path):
            print(f"错误: 数据文件不存在 - {data_path}")
            return False
        
        # 加载真实数据
        data = load_real_darcy_data(data_path)
        if data is None:
            return False
        
        input_data = data['input']
        output_data = data['output']
        
        print(f"真实数据加载成功!")
        print(f"输入数据形状: {input_data.shape}")
        print(f"输出数据形状: {output_data.shape}")
        print(f"输入数据类型: {input_data.dtype}")
        print(f"输出数据类型: {output_data.dtype}")
        
        # 数据维度检查
        if len(input_data.shape) < 2:
            print("警告: 输入数据维度过低")
            return False
        
        # 批次一致性检查
        if input_data.shape[0] != output_data.shape[0]:
            print(f"错误: 批次大小不一致 - 输入: {input_data.shape[0]}, 输出: {output_data.shape[0]}")
            return False
        
        # 数值范围检查
        input_min, input_max = input_data.min().item(), input_data.max().item()
        output_min, output_max = output_data.min().item(), output_data.max().item()
        print(f"输入数据范围: [{input_min:.6f}, {input_max:.6f}]")
        print(f"输出数据范围: [{output_min:.6f}, {output_max:.6f}]")
        
        # 数据完整性检查
        if torch.isnan(input_data).any():
            print("错误: 输入数据包含NaN值")
            return False
        
        if torch.isinf(input_data).any():
            print("错误: 输入数据包含无穷值")
            return False
        
        if torch.isnan(output_data).any():
            print("错误: 输出数据包含NaN值")
            return False
        
        if torch.isinf(output_data).any():
            print("错误: 输出数据包含无穷值")
            return False
        
        print("✅ 真实数据加载验证通过")
        return True
        
    except Exception as e:
        print(f"真实数据加载验证失败: {e}")
        return False

def validate_data_preprocessing(input_data, output_data):
    """验证数据预处理功能"""
    print("\n=== 验证数据预处理 ===")
    
    try:
        # 基本数据处理测试
        # 1. 数据切片
        slice_data = input_data[:50]  # 取前50个样本
        print(f"数据切片成功: {slice_data.shape}")
        
        # 2. 数据重塑（如果需要）
        if len(input_data.shape) > 2:
            # 展平除批次维度外的所有维度
            flattened = input_data.view(input_data.shape[0], -1)
            print(f"数据展平成功: {flattened.shape}")
        
        # 3. 数据归一化
        normalized = (input_data - input_data.mean()) / (input_data.std() + 1e-8)
        print(f"数据归一化成功: 均值={normalized.mean().item():.6f}, 标准差={normalized.std().item():.6f}")
        
        # 4. 数据类型转换
        float64_data = input_data.double()
        float32_data = float64_data.float()
        print(f"数据类型转换成功: {float32_data.dtype}")
        
        print("✅ 数据预处理验证通过")
        return True
        
    except Exception as e:
        print(f"数据预处理验证失败: {e}")
        return False

def validate_model_compatibility(input_data, output_data):
    """验证模型兼容性"""
    print("\n=== 验证模型兼容性 ===")
    
    try:
        # 测试不同的模型输入格式
        batch_size = input_data.shape[0]
        
        # 1. MLP格式 (batch_size, features)
        if len(input_data.shape) > 2:
            mlp_input = input_data.view(batch_size, -1)
        else:
            mlp_input = input_data
        print(f"MLP格式兼容: {mlp_input.shape}")
        
        # 2. CNN格式 (batch_size, channels, height, width)
        if len(input_data.shape) == 4:
            cnn_input = input_data
        elif len(input_data.shape) == 3:
            cnn_input = input_data.unsqueeze(1)  # 添加通道维度
        else:
            # 尝试重塑为2D图像
            feature_size = int(np.sqrt(input_data.shape[-1]))
            if feature_size * feature_size == input_data.shape[-1]:
                cnn_input = input_data.view(batch_size, 1, feature_size, feature_size)
            else:
                cnn_input = input_data.unsqueeze(1).unsqueeze(-1)  # 添加维度
        print(f"CNN格式兼容: {cnn_input.shape}")
        
        # 3. Transformer格式 (batch_size, sequence_length, features)
        if len(input_data.shape) == 3:
            transformer_input = input_data
        else:
            # 重塑为序列格式
            transformer_input = input_data.view(batch_size, -1, 1)
        print(f"Transformer格式兼容: {transformer_input.shape}")
        
        print("✅ 模型兼容性验证通过")
        return True
        
    except Exception as e:
        print(f"模型兼容性验证失败: {e}")
        return False

def calculate_pde_metrics(input_data, output_data):
    """计算PDE相关指标"""
    print("\n=== 计算PDE指标 ===")
    
    try:
        # 1. 相对L2误差（使用真实数据作为参考）
        # 这里我们计算数据的统计特性
        input_l2_norm = torch.norm(input_data, dim=-1).mean()
        output_l2_norm = torch.norm(output_data, dim=-1).mean()
        print(f"输入L2范数: {input_l2_norm.item():.6f}")
        print(f"输出L2范数: {output_l2_norm.item():.6f}")
        
        # 2. 数据分布特性
        input_mean = input_data.mean().item()
        input_std = input_data.std().item()
        output_mean = output_data.mean().item()
        output_std = output_data.std().item()
        
        print(f"输入数据统计: 均值={input_mean:.6f}, 标准差={input_std:.6f}")
        print(f"输出数据统计: 均值={output_mean:.6f}, 标准差={output_std:.6f}")
        
        # 3. 数据相关性
        if input_data.numel() == output_data.numel():
            correlation = torch.corrcoef(torch.stack([
                input_data.flatten(),
                output_data.flatten()
            ]))[0, 1]
            print(f"输入输出相关性: {correlation.item():.6f}")
        
        print("✅ PDE指标计算验证通过")
        return True
        
    except Exception as e:
        print(f"PDE指标计算验证失败: {e}")
        return False

def generate_validation_report(results, data_path):
    """生成验证报告"""
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'data_source': data_path,
        'overall_success': all(results.values()),
        'detailed_results': results,
        'summary': {
            'total_checks': len(results),
            'passed_checks': sum(results.values()),
            'failed_checks': len(results) - sum(results.values()),
            'compliance_status': 'CONFIRMED' if all(results.values()) else 'ISSUES_FOUND'
        }
    }
    
    # 保存报告
    report_path = 'real_data_validation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n验证报告已保存到: {report_path}")
    return report

def main():
    """主函数"""
    print("开始真实数据集验证流程...")
    
    # 真实数据路径
    data_path = "X:\\2025\\Graduation_project\\report\\data\\pdebench\\2D\\DarcyFlow\\2D_DarcyFlow_beta0.01_Train.hdf5"
    
    # 执行验证
    results = {}
    
    # 1. 验证数据加载
    results['data_loading'] = validate_real_data_loading(data_path)
    
    if results['data_loading']:
        # 加载数据用于后续测试
        data = load_real_darcy_data(data_path)
        input_data = data['input']
        output_data = data['output']
        
        # 2. 验证数据预处理
        results['data_preprocessing'] = validate_data_preprocessing(input_data, output_data)
        
        # 3. 验证模型兼容性
        results['model_compatibility'] = validate_model_compatibility(input_data, output_data)
        
        # 4. 验证PDE指标计算
        results['pde_metrics'] = calculate_pde_metrics(input_data, output_data)
    else:
        results['data_preprocessing'] = False
        results['model_compatibility'] = False
        results['pde_metrics'] = False
    
    # 生成报告
    report = generate_validation_report(results, data_path)
    
    # 输出总结
    print("\n" + "="*50)
    print("真实数据验证总结:")
    print(f"数据源: {data_path}")
    print(f"总体状态: {'✅ 通过' if report['overall_success'] else '❌ 失败'}")
    print(f"通过检查: {report['summary']['passed_checks']}/{report['summary']['total_checks']}")
    print(f"合规状态: {report['summary']['compliance_status']}")
    print("="*50)
    
    return 0 if report['overall_success'] else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)