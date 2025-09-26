#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试FNO和UNet模型在real_data_crop_config.yaml配置下的兼容性

功能:
1. 验证FNO和UNet模型能否正确加载和运行
2. 测试模型对PDE数据集的处理能力
3. 生成兼容性报告

作者: AI Assistant
日期: 2025
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入模型
try:
    from modify_multi_attention.models.enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
    FNO_AVAILABLE = True
except ImportError:
    FNO_AVAILABLE = False

try:
    from modify_multi_attention.models.enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
    UNET_AVAILABLE = True
except ImportError:
    UNET_AVAILABLE = False

# 导入数据加载器
from modify_multi_attention.data.crop_dataloader import create_crop_dataloader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_compatibility(model, model_name, input_data, target_data):
    """
    测试模型兼容性
    
    Args:
        model: 模型实例
        model_name: 模型名称
        input_data: 输入数据
        target_data: 目标数据
        
    Returns:
        dict: 测试结果
    """
    try:
        model.eval()
        
        # 前向传播测试
        with torch.no_grad():
            output = model(input_data)
        
        # 检查输出形状
        if output.shape != target_data.shape:
            return {
                'success': False,
                'error': f'输出形状不匹配: 期望{target_data.shape}, 实际{output.shape}',
                'param_count': sum(p.numel() for p in model.parameters())
            }
        
        # 计算损失
        mse_loss = nn.MSELoss()(output, target_data).item()
        
        return {
            'success': True,
            'output_shape': output.shape,
            'mse_loss': mse_loss,
            'param_count': sum(p.numel() for p in model.parameters()),
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'param_count': sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0
        }

def create_test_models(config):
    """
    创建测试模型
    
    Args:
        config: 配置字典
        
    Returns:
        dict: 模型字典
    """
    models = {}
    
    # 获取配置参数
    input_resolution = config['data']['input_resolution']
    output_resolution = config['data']['output_resolution']
    
    # FNO模型
    if FNO_AVAILABLE:
        try:
            # FNO1d模型
            fno1d = EnhancedFNO1d(
                num_channels=1,
                modes=16,  # 默认modes值
                width=config['models']['fno']['width'],
                input_resolution=input_resolution[0],
                output_resolution=output_resolution[0]
            )
            models['fno1d'] = fno1d
            logger.info("✅ FNO1d模型创建成功")
            
            # FNO2d模型
            fno2d = EnhancedFNO2d(
                num_channels=1,
                modes1=16,  # 默认modes1值
                modes2=16,  # 默认modes2值
                width=config['models']['fno']['width'],
                input_resolution=tuple(input_resolution),
                output_resolution=tuple(output_resolution)
            )
            models['fno2d'] = fno2d
            logger.info("✅ FNO2d模型创建成功")
            
        except Exception as e:
            logger.error(f"❌ FNO模型创建失败: {e}")
    else:
        logger.warning("⚠️ FNO模型不可用")
    
    # UNet模型
    if UNET_AVAILABLE:
        try:
            # UNet1d模型
            unet1d = EnhancedUNet1d(
                in_channels=1,
                out_channels=1,
                init_features=64,  # 默认init_features值
                input_resolution=input_resolution[0],
                output_resolution=output_resolution[0] * output_resolution[1]  # 使用总的输出长度
            )
            models['unet1d'] = unet1d
            logger.info("✅ UNet1d模型创建成功")
            
            # UNet2d模型
            unet2d = EnhancedUNet2d(
                in_channels=1,
                out_channels=1,
                init_features=64,  # 默认init_features值
                input_resolution=tuple(input_resolution),
                output_resolution=tuple(output_resolution)
            )
            models['unet2d'] = unet2d
            logger.info("✅ UNet2d模型创建成功")
            
        except Exception as e:
            logger.error(f"❌ UNet模型创建失败: {e}")
    else:
        logger.warning("⚠️ UNet模型不可用")
    
    return models

def generate_test_data(input_dim, output_dim, batch_size=4):
    """
    生成测试数据
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        batch_size: 批次大小
        
    Returns:
        tuple: (输入数据, 目标数据)
    """
    input_data = torch.randn(batch_size, input_dim)
    target_data = torch.randn(batch_size, output_dim)
    return input_data, target_data

def main():
    """主函数"""
    logger.info("🚀 开始FNO和UNet模型兼容性测试...")
    
    # 加载配置
    config_path = "configs/real_data_crop_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"📋 加载配置文件: {config_path}")
    
    # 创建测试模型
    logger.info("🔧 创建测试模型...")
    models = create_test_models(config)
    
    if not models:
        logger.error("❌ 没有可用的模型进行测试")
        return
    
    # 生成测试数据
    input_dim = np.prod(config['data']['input_resolution'])  # 32*32 = 1024
    output_dim = np.prod(config['data']['output_resolution'])  # 128*128 = 16384
    
    logger.info(f"📊 生成测试数据: 输入维度={input_dim}, 输出维度={output_dim}")
    input_data, target_data = generate_test_data(input_dim, output_dim)
    
    # 测试所有模型
    results = {}
    logger.info("🧪 开始模型测试...")
    
    for model_name, model in models.items():
        logger.info(f"测试模型: {model_name}")
        result = test_model_compatibility(model, model_name, input_data, target_data)
        results[model_name] = result
        
        if result['success']:
            logger.info(f"✅ {model_name} 测试成功")
            logger.info(f"   参数数量: {result['param_count']:,}")
            logger.info(f"   输出形状: {result['output_shape']}")
            logger.info(f"   MSE损失: {result['mse_loss']:.6f}")
        else:
            logger.error(f"❌ {model_name} 测试失败: {result['error']}")
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"fno_unet_compatibility_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("FNO和UNet模型兼容性测试报告\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"输入维度: {input_dim} ({config['data']['input_resolution'][0]}x{config['data']['input_resolution'][1]})\n")
        f.write(f"输出维度: {output_dim} ({config['data']['output_resolution'][0]}x{config['data']['output_resolution'][1]})\n")
        f.write(f"总模型数: {len(models)}\n")
        f.write(f"成功模型数: {sum(1 for r in results.values() if r['success'])}\n\n")
        
        for model_name, result in results.items():
            f.write(f"模型: {model_name}\n")
            f.write(f"  状态: {'✅ 成功' if result['success'] else '❌ 失败'}\n")
            f.write(f"  参数数量: {result['param_count']:,}\n")
            
            if result['success']:
                f.write(f"  输出形状: {result['output_shape']}\n")
                f.write(f"  MSE损失: {result['mse_loss']:.6f}\n")
            else:
                f.write(f"  错误信息: {result['error']}\n")
            f.write("\n")
    
    logger.info(f"📊 兼容性报告已保存: {report_path}")
    logger.info("🎉 测试完成！")

if __name__ == "__main__":
    main()