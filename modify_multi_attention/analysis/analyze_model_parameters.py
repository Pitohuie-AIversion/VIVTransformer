#!/usr/bin/env python3
"""
模型参数量分析工具
用于分析不同模型的参数量，帮助调整配置以实现公平对比
"""

import torch
import torch.nn as nn
import yaml
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from run_crop_model_test import create_enhanced_model, SimpleMLP, SimpleTransformer

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_parameters(param_count):
    """格式化参数量显示"""
    if param_count >= 1e6:
        return f"{param_count/1e6:.2f}M"
    elif param_count >= 1e3:
        return f"{param_count/1e3:.2f}K"
    else:
        return str(param_count)

def analyze_model_parameters():
    """分析各模型的参数量"""
    
    # 加载配置文件
    config_path = project_root / "configs" / "unified_training_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    models_config = config['models']
    
    print("=" * 60)
    print("🔍 模型参数量分析报告")
    print("=" * 60)
    
    results = {}
    
    for model_name, model_config in models_config.items():
        print(f"\n📊 分析模型: {model_name.upper()}")
        print("-" * 40)
        
        try:
            if model_name == 'mlp':
                # MLP模型
                hidden_dims = model_config.get('hidden_dims', [256, 128])
                model = SimpleMLP(
                    input_dim=model_config['input_dim'],
                    output_dim=model_config['output_dim'],
                    hidden_dims=hidden_dims,
                    activation=model_config.get('activation', 'relu'),
                    dropout=model_config.get('dropout', 0.1),
                    use_batch_norm=model_config.get('use_batch_norm', True)
                )
            
            elif model_name == 'transformer':
                # Transformer模型
                model = SimpleTransformer(
                    input_dim=model_config['input_dim'],
                    output_dim=model_config['output_dim'],
                    d_model=model_config['d_model'],
                    num_heads=model_config['num_heads'],
                    num_layers=model_config['num_layers'],
                    dropout=model_config.get('dropout', 0.1),
                    seq_len=model_config['seq_len']
                )
            
            elif model_name in ['unet', 'fno']:
                # 增强模型
                model = create_enhanced_model(
                    model_config=model_config,
                    input_dim=model_config['input_dim'],
                    output_dim=model_config['output_dim']
                )
            
            else:
                print(f"⚠️ 未知模型类型: {model_name}")
                continue
            
            # 计算参数量
            param_count = count_parameters(model)
            results[model_name] = param_count
            
            print(f"参数量: {format_parameters(param_count)} ({param_count:,})")
            
            # 显示关键配置
            if model_name == 'mlp':
                hidden_dims = model_config.get('hidden_dims', [256, 128])
                print(f"隐藏层维度: {hidden_dims}")
                print(f"层数: {len(hidden_dims) + 1}")
            elif model_name == 'transformer':
                print(f"模型维度: {model_config['d_model']}")
                print(f"注意力头数: {model_config['num_heads']}")
                print(f"层数: {model_config['num_layers']}")
                print(f"序列长度: {model_config['seq_len']}")
            elif model_name == 'unet':
                print(f"通道数: {model_config.get('channels', 'N/A')}")
                print(f"输入空间维度: {model_config.get('input_spatial_dim', 'N/A')}")
                print(f"输出空间维度: {model_config.get('output_spatial_dim', 'N/A')}")
            elif model_name == 'fno':
                print(f"宽度: {model_config.get('width', 'N/A')}")
                print(f"层数: {model_config.get('num_layers', 'N/A')}")
                print(f"模式数: {model_config.get('modes', 'N/A')}")
                print(f"输入空间维度: {model_config.get('input_spatial_dim', 32)}")
                print(f"输出空间维度: {model_config.get('output_spatial_dim', 32)}")
            
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
            results[model_name] = 0
    
    # 参数量对比分析
    print("\n" + "=" * 60)
    print("📈 参数量对比分析")
    print("=" * 60)
    
    if results:
        max_params = max(results.values())
        min_params = min([p for p in results.values() if p > 0])
        
        print(f"最大参数量: {format_parameters(max_params)}")
        print(f"最小参数量: {format_parameters(min_params)}")
        print(f"参数量比例: {max_params/min_params:.2f}:1")
        
        print("\n📊 各模型参数量排序:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, param_count) in enumerate(sorted_results):
            if param_count > 0:
                ratio = param_count / min_params
                print(f"{i+1}. {model_name.upper()}: {format_parameters(param_count)} (x{ratio:.2f})")
    
    # 生成参数平衡建议
    print("\n" + "=" * 60)
    print("💡 参数平衡建议")
    print("=" * 60)
    
    if results:
        target_params = sum(results.values()) / len([p for p in results.values() if p > 0])
        print(f"建议目标参数量: {format_parameters(target_params)}")
        
        print("\n🔧 调整建议:")
        for model_name, param_count in results.items():
            if param_count > 0:
                if param_count > target_params * 1.5:
                    print(f"• {model_name.upper()}: 参数过多，建议减少模型复杂度")
                elif param_count < target_params * 0.5:
                    print(f"• {model_name.upper()}: 参数过少，建议增加模型复杂度")
                else:
                    print(f"• {model_name.upper()}: 参数量合理")
    
    return results

def generate_balanced_config(results):
    """生成参数平衡的配置建议"""
    if not results:
        return
    
    target_params = sum(results.values()) / len([p for p in results.values() if p > 0])
    
    print("\n" + "=" * 60)
    print("⚙️ 参数平衡配置建议")
    print("=" * 60)
    
    print("# 建议的配置调整")
    print("models:")
    
    # MLP调整建议
    if 'mlp' in results:
        mlp_params = results['mlp']
        if mlp_params > target_params * 1.2:
            print("  mlp:")
            print("    hidden_dim: 128  # 减少隐藏层维度")
            print("    num_layers: 4    # 减少层数")
        elif mlp_params < target_params * 0.8:
            print("  mlp:")
            print("    hidden_dim: 512  # 增加隐藏层维度")
            print("    num_layers: 6    # 增加层数")
    
    # Transformer调整建议
    if 'transformer' in results:
        trans_params = results['transformer']
        if trans_params > target_params * 1.2:
            print("  transformer:")
            print("    d_model: 128     # 减少模型维度")
            print("    num_heads: 4     # 减少注意力头数")
            print("    num_layers: 2    # 减少层数")
        elif trans_params < target_params * 0.8:
            print("  transformer:")
            print("    d_model: 512     # 增加模型维度")
            print("    num_heads: 8     # 增加注意力头数")
            print("    num_layers: 4    # 增加层数")
    
    # UNet调整建议
    if 'unet' in results:
        unet_params = results['unet']
        if unet_params > target_params * 1.2:
            print("  unet:")
            print("    channels: [16, 32, 64]  # 减少通道数")
        elif unet_params < target_params * 0.8:
            print("  unet:")
            print("    channels: [64, 128, 256, 512]  # 增加通道数")
    
    # FNO调整建议
    if 'fno' in results:
        fno_params = results['fno']
        if fno_params > target_params * 1.2:
            print("  fno:")
            print("    width: 32        # 减少网络宽度")
            print("    num_layers: 2    # 减少层数")
        elif fno_params < target_params * 0.8:
            print("  fno:")
            print("    width: 128       # 增加网络宽度")
            print("    num_layers: 6    # 增加层数")

if __name__ == "__main__":
    try:
        results = analyze_model_parameters()
        generate_balanced_config(results)
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()