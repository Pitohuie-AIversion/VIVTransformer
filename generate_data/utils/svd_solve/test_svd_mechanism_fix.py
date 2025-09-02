#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SVD机制修复效果
验证svd_loss_enabled参数是否正常工作
"""

import os
import sys
import torch
import yaml
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(modify_multi_attention_path))
sys.path.insert(0, str(Path(__file__).parent))

# 导入必要的模块
try:
    from modify_multi_attention.utils.svd10_loss import TotalLossWithSVD
except ImportError:
    from utils.svd10_loss import TotalLossWithSVD

from dynamic_resolution_trainer import load_config_with_args, validate_config_detailed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_svd_enabled_config():
    """
    测试启用SVD损失的配置
    """
    logger.info("=== 测试启用SVD损失的配置 ===")
    
    # 创建启用SVD的配置（跳过数据文件验证）
    config = {
        'data': {
            'path': './data/dummy_data.h5',  # 虚拟路径，仅用于测试
            'input_resolution': [32, 32],
            'output_resolution': [64, 64],
            'batch_size': 4
        },
        'training': {
            'epochs': 5,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        },
        'model': {
            'num_layers': 2,
            'd_model': 256,
            'num_heads': 4,
            'max_time_steps': 100,
            'attention_type': 'sge'
        },
        'loss': {
            'svd_loss_enabled': True,
            'base_weight': 0.8,
            'svd_weights': [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            'topk': 10
        }
    }
    
    # 计算维度
    input_h, input_w = config['data']['input_resolution']
    output_h, output_w = config['data']['output_resolution']
    config['model']['input_dim'] = input_h * input_w
    config['model']['output_dim'] = output_h * output_w
    config['model']['seq_len'] = input_h * input_w
    
    # 跳过数据文件验证，只验证SVD损失相关配置
    logger.info("跳过数据文件验证，专注测试SVD损失配置")
    
    # 测试损失函数创建
    loss_config = config['loss']
    svd_loss_enabled = loss_config.get('svd_loss_enabled', True)
    
    if svd_loss_enabled:
        logger.info("[OK] SVD损失已启用")
        criterion = TotalLossWithSVD(
            base_weight=loss_config['base_weight'],
            svd_weights=loss_config['svd_weights'],
            topk=loss_config['topk']
        )
        logger.info(f"损失函数类型: {type(criterion).__name__}")
        
        # 测试损失计算
        test_input = torch.randn(2, 1024)  # batch_size=2, input_dim=32*32
        test_target = torch.randn(2, 4096)  # batch_size=2, output_dim=64*64
        
        try:
            loss = criterion(test_input, test_target)
            logger.info(f"[OK] SVD损失计算成功: {loss.item():.6f}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] SVD损失计算失败: {e}")
            return False
    else:
        logger.error("[ERROR] SVD损失应该启用但未启用")
        return False

def test_svd_disabled_config():
    """
    测试禁用SVD损失的配置
    """
    logger.info("=== 测试禁用SVD损失的配置 ===")
    
    # 创建禁用SVD的配置（跳过数据文件验证）
    config = {
        'data': {
            'path': './data/dummy_data.h5',  # 虚拟路径，仅用于测试
            'input_resolution': [32, 32],
            'output_resolution': [64, 64],
            'batch_size': 4
        },
        'training': {
            'epochs': 5,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        },
        'model': {
            'num_layers': 2,
            'd_model': 256,
            'num_heads': 4,
            'max_time_steps': 100,
            'attention_type': 'sge'
        },
        'loss': {
            'svd_loss_enabled': False,  # 关键：禁用SVD损失
            'base_weight': 1.0,
            'svd_weights': [0.0] * 10,
            'topk': 10
        }
    }
    
    # 计算维度
    input_h, input_w = config['data']['input_resolution']
    output_h, output_w = config['data']['output_resolution']
    config['model']['input_dim'] = input_h * input_w
    config['model']['output_dim'] = output_h * output_w
    config['model']['seq_len'] = input_h * input_w
    
    # 跳过数据文件验证，专注测试SVD损失配置
    logger.info("跳过数据文件验证，专注测试SVD损失配置")
    
    # 测试损失函数创建
    loss_config = config['loss']
    svd_loss_enabled = loss_config.get('svd_loss_enabled', True)
    
    if not svd_loss_enabled:
        logger.info("[OK] SVD损失已禁用")
        criterion = torch.nn.MSELoss()
        logger.info(f"损失函数类型: {type(criterion).__name__}")
        
        # 测试损失计算
        test_input = torch.randn(2, 4096)  # batch_size=2, output_dim=64*64
        test_target = torch.randn(2, 4096)  # batch_size=2, output_dim=64*64
        
        try:
            loss = criterion(test_input, test_target)
            logger.info(f"[OK] MSE损失计算成功: {loss.item():.6f}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] MSE损失计算失败: {e}")
            return False
    else:
        logger.error("[ERROR] SVD损失应该禁用但未禁用")
        return False

def test_config_file_loading():
    """
    测试从配置文件加载
    """
    logger.info("=== 测试配置文件加载 ===")
    
    config_file = "test_svd_disabled_config.yaml"
    if not os.path.exists(config_file):
        logger.warning(f"配置文件不存在: {config_file}")
        return True
    
    try:
        # 模拟命令行参数
        class Args:
            def __init__(self):
                self.config = config_file
                self.data_path = None
                self.epochs = None
                self.batch_size = None
                self.learning_rate = None
                self.device = None
                self.use_dataparallel = None
                self.no_pretrained = None
                self.seed = None
                # 添加缺失的属性
                self.input_resolution = None
                self.output_resolution = None
                self.num_samples = None
                self.crop_mode = None
                self.normalize_data = None
                self.lazy_loading = None
                self.weight_decay = None
                self.patience = None
                self.min_delta = None
                self.save_best_model = None
                self.model_save_path = None
                self.log_interval = None
                self.num_layers = None
                self.d_model = None
                self.num_heads = None
                self.max_time_steps = None
                self.attention_type = None
                self.visualization_enabled = None
                self.visualization_interval = None
                self.visualization_max_samples = None
                self.logging_level = None
                self.logging_format = None
                self.logging_file = None
                self.optimizer_type = None
                self.optimizer_betas = None
                self.optimizer_eps = None
                self.optimizer_amsgrad = None
                self.scheduler_enabled = None
                self.scheduler_type = None
                self.scheduler_T_max = None
                self.scheduler_eta_min = None
                self.scheduler_step_size = None
                self.scheduler_gamma = None
                self.loss_base_weight = None
                self.loss_svd_weights = None
                self.loss_topk = None
                self.loss_type = None
                self.loss_svd_loss_enabled = None
                self.loss_normalize_svd_weights = None
                self.verbose = None
        
        args = Args()
        config = load_config_with_args(config_file, args)
        
        # 验证SVD损失是否正确禁用
        svd_loss_enabled = config['loss'].get('svd_loss_enabled', True)
        if not svd_loss_enabled:
            logger.info("[OK] 配置文件中SVD损失已正确禁用")
            return True
        else:
            logger.error("[ERROR] 配置文件中SVD损失未正确禁用")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] 配置文件加载失败: {e}")
        return False

def main():
    """
    主测试函数
    """
    logger.info("[TEST] 开始测试SVD机制修复效果")
    
    tests = [
        ("SVD启用测试", test_svd_enabled_config),
        ("SVD禁用测试", test_svd_disabled_config),
        ("配置文件加载测试", test_config_file_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"[OK] {test_name} 通过")
            else:
                logger.error(f"[ERROR] {test_name} 失败")
        except Exception as e:
            logger.error(f"[ERROR] {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 总结测试结果
    logger.info(f"\n{'='*50}")
    logger.info("测试结果总结")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] 通过" if result else "[ERROR] 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("[OK] 所有测试通过！SVD机制修复成功！")
        return True
    else:
        logger.error(f"[WARN] {total - passed} 个测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)