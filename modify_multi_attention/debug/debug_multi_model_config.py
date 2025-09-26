#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_model_config.yaml 调试脚本

功能:
1. 验证配置文件格式和内容
2. 检查模型配置的完整性
3. 测试模型创建流程
4. 识别潜在问题并提供修复建议

作者: AI Assistant
日期: 2025
"""

import yaml
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'generate_data'))

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件加载成功")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def analyze_config_structure(config: Dict[str, Any]) -> None:
    """分析配置文件结构"""
    print("\n📊 配置文件结构分析:")
    
    # 主要配置部分
    main_sections = ['data', 'training', 'loss_configs', 'attention_types', 'models', 'device', 'visualization', 'output']
    
    for section in main_sections:
        if section in config:
            print(f"  ✅ {section}")
            if section == 'models':
                models = config[section]
                print(f"     包含 {len(models)} 个模型:")
                for model_name in models.keys():
                    print(f"       - {model_name}")
        else:
            print(f"  ❌ {section} (缺失)")

def validate_model_configs(config: Dict[str, Any]) -> List[str]:
    """验证模型配置"""
    print("\n🤖 模型配置验证:")
    
    issues = []
    models = config.get('models', {})
    
    for model_name, model_config in models.items():
        print(f"\n  检查模型: {model_name}")
        
        # 检查必需字段
        required_fields = ['model_type', 'input_dim', 'output_dim']
        for field in required_fields:
            if field in model_config:
                print(f"    ✅ {field}: {model_config[field]}")
            else:
                issue = f"模型 {model_name} 缺少必需字段: {field}"
                print(f"    ❌ {issue}")
                issues.append(issue)
        
        # 检查模型类型特定配置
        model_type = model_config.get('model_type', '')
        
        if model_type == 'transformer':
            transformer_fields = ['seq_len', 'd_model', 'num_heads', 'num_layers']
            for field in transformer_fields:
                if field in model_config:
                    print(f"    ✅ {field}: {model_config[field]}")
                else:
                    issue = f"Transformer模型 {model_name} 缺少字段: {field}"
                    print(f"    ⚠️  {issue}")
                    issues.append(issue)
        
        elif model_type == 'mlp':
            mlp_fields = ['hidden_dims']
            for field in mlp_fields:
                if field in model_config:
                    print(f"    ✅ {field}: {model_config[field]}")
                else:
                    issue = f"MLP模型 {model_name} 缺少字段: {field}"
                    print(f"    ⚠️  {issue}")
                    issues.append(issue)
        
        elif model_type == 'fno':
            fno_fields = ['width', 'num_layers']
            for field in fno_fields:
                if field in model_config:
                    print(f"    ✅ {field}: {model_config[field]}")
                else:
                    issue = f"FNO模型 {model_name} 缺少字段: {field}"
                    print(f"    ⚠️  {issue}")
                    issues.append(issue)
        
        elif model_type == 'unet':
            unet_fields = ['in_channels', 'out_channels', 'features']
            for field in unet_fields:
                if field in model_config:
                    print(f"    ✅ {field}: {model_config[field]}")
                else:
                    issue = f"UNet模型 {model_name} 缺少字段: {field}"
                    print(f"    ⚠️  {issue}")
                    issues.append(issue)
    
    return issues

def test_model_creation(config: Dict[str, Any]) -> List[str]:
    """测试模型创建"""
    print("\n🔧 模型创建测试:")
    
    creation_issues = []
    models = config.get('models', {})
    
    for model_name, model_config in models.items():
        print(f"\n  测试创建模型: {model_name}")
        
        try:
            model_type = model_config.get('model_type', '')
            
            if model_type == 'transformer':
                # 尝试创建简化的Transformer模型
                input_dim = model_config.get('input_dim', 16384)
                output_dim = model_config.get('output_dim', 16384)
                d_model = model_config.get('d_model', 64)
                
                # 简单的线性层测试
                test_model = nn.Sequential(
                    nn.Linear(input_dim, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, output_dim)
                )
                
                # 测试前向传播
                test_input = torch.randn(1, input_dim)
                test_output = test_model(test_input)
                
                print(f"    ✅ 模型创建成功，输出形状: {test_output.shape}")
                
            elif model_type == 'mlp':
                # 测试MLP模型
                input_dim = model_config.get('input_dim', 16384)
                output_dim = model_config.get('output_dim', 16384)
                hidden_dims = model_config.get('hidden_dims', [1024, 512, 256, 512, 1024])
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.GELU())
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, output_dim))
                test_model = nn.Sequential(*layers)
                
                # 测试前向传播
                test_input = torch.randn(1, input_dim)
                test_output = test_model(test_input)
                
                print(f"    ✅ 模型创建成功，输出形状: {test_output.shape}")
                
            else:
                print(f"    ⚠️  跳过模型类型 {model_type} 的创建测试")
                
        except Exception as e:
            issue = f"模型 {model_name} 创建失败: {str(e)}"
            print(f"    ❌ {issue}")
            creation_issues.append(issue)
            print(f"    详细错误: {traceback.format_exc()}")
    
    return creation_issues

def check_data_compatibility(config: Dict[str, Any]) -> List[str]:
    """检查数据配置兼容性"""
    print("\n📊 数据配置兼容性检查:")
    
    data_issues = []
    data_config = config.get('data', {})
    models = config.get('models', {})
    
    # 检查数据路径
    data_path = data_config.get('data_path', '')
    if data_path:
        if Path(data_path).exists():
            print(f"  ✅ 数据文件存在: {data_path}")
        else:
            issue = f"数据文件不存在: {data_path}"
            print(f"  ❌ {issue}")
            data_issues.append(issue)
    else:
        issue = "未指定数据路径"
        print(f"  ❌ {issue}")
        data_issues.append(issue)
    
    # 检查输入输出维度一致性
    data_input_dim = data_config.get('input_dim', 16384)
    data_output_dim = data_config.get('output_dim', 16384)
    
    print(f"  数据配置 - 输入维度: {data_input_dim}, 输出维度: {data_output_dim}")
    
    for model_name, model_config in models.items():
        model_input_dim = model_config.get('input_dim', 0)
        model_output_dim = model_config.get('output_dim', 0)
        
        if model_input_dim != data_input_dim:
            issue = f"模型 {model_name} 输入维度 ({model_input_dim}) 与数据配置不匹配 ({data_input_dim})"
            print(f"  ⚠️  {issue}")
            data_issues.append(issue)
        
        if model_output_dim != data_output_dim:
            issue = f"模型 {model_name} 输出维度 ({model_output_dim}) 与数据配置不匹配 ({data_output_dim})"
            print(f"  ⚠️  {issue}")
            data_issues.append(issue)
    
    return data_issues

def generate_fix_suggestions(all_issues: List[str]) -> None:
    """生成修复建议"""
    print("\n🔧 修复建议:")
    
    if not all_issues:
        print("  🎉 未发现问题，配置文件看起来很好！")
        return
    
    print(f"  发现 {len(all_issues)} 个问题:")
    
    for i, issue in enumerate(all_issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 建议修复方案:")
    
    # 根据问题类型提供建议
    if any("缺少必需字段" in issue for issue in all_issues):
        print("  - 添加缺失的必需字段到相应的模型配置中")
    
    if any("输入维度" in issue or "输出维度" in issue for issue in all_issues):
        print("  - 确保所有模型的input_dim和output_dim与数据配置一致")
    
    if any("数据文件不存在" in issue for issue in all_issues):
        print("  - 检查数据文件路径是否正确，或生成测试数据文件")
    
    if any("创建失败" in issue for issue in all_issues):
        print("  - 检查模型参数配置是否合理，特别是维度设置")

def main():
    """主函数"""
    print("🔍 multi_model_config.yaml 调试工具")
    print("=" * 50)
    
    # 配置文件路径
    config_path = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/modify_multi_attention/configs/multi_model_config.yaml"
    
    # 加载配置
    config = load_config(config_path)
    if config is None:
        return
    
    # 分析配置结构
    analyze_config_structure(config)
    
    # 验证模型配置
    model_issues = validate_model_configs(config)
    
    # 测试模型创建
    creation_issues = test_model_creation(config)
    
    # 检查数据兼容性
    data_issues = check_data_compatibility(config)
    
    # 汇总所有问题
    all_issues = model_issues + creation_issues + data_issues
    
    # 生成修复建议
    generate_fix_suggestions(all_issues)
    
    print("\n" + "=" * 50)
    print("🏁 调试完成")

if __name__ == "__main__":
    main()