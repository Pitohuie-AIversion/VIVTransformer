#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型复杂度分析脚本
检查模型参数和计算复杂度
"""

import yaml
import torch
import math

def analyze_model_complexity():
    """分析模型复杂度"""
    print("=" * 60)
    print("[MODEL] 模型复杂度分析报告")
    print("=" * 60)
    
    try:
        # 加载配置
        with open('dynamic_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        data_config = config['data']
        
        # 计算维度
        input_dim = data_config['input_resolution'][0] * data_config['input_resolution'][1]
        output_dim = data_config['output_resolution'][0] * data_config['output_resolution'][1]
        
        # 模型参数
        d_model = model_config['d_model']
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        dim_feedforward = model_config['dim_feedforward']
        max_time_steps = model_config['max_time_steps']
        attention_type = model_config['attention_type']
        
        print("[INFO] 模型架构参数:")
        print(f"  输入维度: {input_dim} ({data_config['input_resolution'][0]}x{data_config['input_resolution'][1]})")
        print(f"  输出维度: {output_dim} ({data_config['output_resolution'][0]}x{data_config['output_resolution'][1]})")
        print(f"  模型维度: {d_model}")
        print(f"  层数: {num_layers}")
        print(f"  注意力头数: {num_heads}")
        print(f"  前馈网络维度: {dim_feedforward}")
        print(f"  最大时间步: {max_time_steps}")
        print(f"  注意力类型: {attention_type}")
        
        # 参数量估算
        print("\n[PARAMS] 参数量估算:")
        
        # 输入投影层
        input_proj_params = input_dim * d_model
        print(f"  输入投影层: {input_proj_params:,} ({input_proj_params/1e6:.2f}M)")
        
        # Transformer层参数
        # 每层包含：多头注意力 + 前馈网络 + 层归一化
        per_layer_params = (
            # 多头注意力 (Q, K, V, O)
            4 * d_model * d_model +
            # 前馈网络
            d_model * dim_feedforward + dim_feedforward * d_model +
            # 层归一化 (2层)
            2 * d_model * 2
        )
        transformer_params = per_layer_params * num_layers
        print(f"  Transformer层: {transformer_params:,} ({transformer_params/1e6:.2f}M)")
        print(f"    - 每层参数: {per_layer_params:,} ({per_layer_params/1e6:.2f}M)")
        
        # 输出投影层
        output_proj_params = d_model * output_dim
        print(f"  输出投影层: {output_proj_params:,} ({output_proj_params/1e6:.2f}M)")
        
        # 总参数量
        total_params = input_proj_params + transformer_params + output_proj_params
        print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # 内存估算
        print("\n[MEM] 内存需求估算:")
        
        # 模型参数内存 (float32)
        model_memory_mb = total_params * 4 / 1024**2
        print(f"  模型参数内存: {model_memory_mb:.1f}MB")
        
        # 梯度内存 (与参数相同)
        gradient_memory_mb = model_memory_mb
        print(f"  梯度内存: {gradient_memory_mb:.1f}MB")
        
        # 优化器状态内存 (Adam需要2倍参数内存)
        optimizer_memory_mb = model_memory_mb * 2
        print(f"  优化器状态内存: {optimizer_memory_mb:.1f}MB")
        
        # 激活值内存估算 (批次大小=1)
        batch_size = data_config['batch_size']
        activation_memory_mb = (
            batch_size * max_time_steps * d_model * num_layers * 4 / 1024**2
        )
        print(f"  激活值内存: {activation_memory_mb:.1f}MB")
        
        # 总内存需求
        total_memory_mb = model_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_memory_mb
        print(f"  总内存需求: {total_memory_mb:.1f}MB ({total_memory_mb/1024:.2f}GB)")
        
        # 计算复杂度分析
        print("\n[COMPUTE] 计算复杂度分析:")
        
        # 注意力计算复杂度 O(n²d)
        attention_flops = num_layers * max_time_steps**2 * d_model
        print(f"  注意力计算: {attention_flops:,} FLOPs")
        
        # 前馈网络计算复杂度
        ffn_flops = num_layers * max_time_steps * (d_model * dim_feedforward * 2)
        print(f"  前馈网络计算: {ffn_flops:,} FLOPs")
        
        # 总计算量
        total_flops = attention_flops + ffn_flops
        print(f"  总计算量: {total_flops:,} FLOPs ({total_flops/1e9:.2f}G FLOPs)")
        
        # 潜在限制分析
        print("\n[WARN] 潜在限制分析:")
        limitations = []
        
        # GPU内存检查
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_usage_ratio = total_memory_mb / 1024 / gpu_memory_gb
            print(f"  GPU内存使用率: {memory_usage_ratio*100:.1f}% ({total_memory_mb/1024:.2f}GB / {gpu_memory_gb:.1f}GB)")
            
            if memory_usage_ratio > 0.8:
                limitations.append(f"GPU内存使用率过高({memory_usage_ratio*100:.1f}%)，可能导致OOM")
            elif memory_usage_ratio > 0.6:
                limitations.append(f"GPU内存使用率较高({memory_usage_ratio*100:.1f}%)，建议监控")
        
        # 模型复杂度检查
        if total_params > 100e6:  # 100M参数
            limitations.append(f"模型参数量较大({total_params/1e6:.1f}M)，训练可能较慢")
        
        # 序列长度检查
        if max_time_steps > 512:
            limitations.append(f"序列长度较长({max_time_steps})，注意力计算复杂度较高")
        
        # 批次大小检查
        if batch_size == 1:
            limitations.append("批次大小为1，GPU利用率可能不高")
        
        # 维度匹配检查
        if d_model % num_heads != 0:
            limitations.append(f"模型维度({d_model})不能被注意力头数({num_heads})整除")
        
        if not limitations:
            print("  [OK] 未发现明显限制")
        else:
            for limitation in limitations:
                print(f"  [WARN] {limitation}")
        
        # 优化建议
        print("\n[TIP] 优化建议:")
        suggestions = []
        
        if batch_size == 1 and torch.cuda.is_available():
            suggestions.append("考虑增加批次大小以提升GPU利用率")
        
        if total_params > 50e6 and data_config['num_samples'] < 1000:
            suggestions.append("模型复杂度相对数据量较高，注意过拟合风险")
        
        if max_time_steps > 100 and attention_type == 'sge':
            suggestions.append("序列较长时，考虑使用更高效的注意力机制")
        
        if dim_feedforward > 4 * d_model:
            suggestions.append(f"前馈网络维度({dim_feedforward})相对模型维度({d_model})过大，可适当减小")
        
        if not suggestions:
            print("  [OK] 当前配置较为合理")
        else:
            for suggestion in suggestions:
                print(f"  [TIP] {suggestion}")
        
        print("\n" + "=" * 60)
        print("[OK] 模型复杂度分析完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_model_complexity()