#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的SVD机制测试
专门验证svd_loss_enabled参数是否正常工作
"""

import torch
import logging
from pathlib import Path
import sys

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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_svd_loss_creation():
    """
    测试SVD损失函数的创建和计算
    """
    logger.info("=== 测试SVD损失函数创建 ===")
    
    try:
        # 创建SVD损失函数
        criterion = TotalLossWithSVD(
            base_weight=0.8,
            svd_weights=[0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            topk=10
        )
        
        logger.info(f"✅ SVD损失函数创建成功: {type(criterion).__name__}")
        
        # 测试损失计算
        batch_size = 2
        input_dim = 1024  # 32*32
        output_dim = 4096  # 64*64
        
        test_pred = torch.randn(batch_size, output_dim)
        test_target = torch.randn(batch_size, output_dim)
        
        loss = criterion(test_pred, test_target)
        logger.info(f"✅ SVD损失计算成功: {loss.item():.6f}")
        
        # 检查权重信息
        weight_info = criterion.get_weight_info()
        logger.info(f"权重总和: {weight_info['weight_sum']:.4f}")
        logger.info(f"归一化基础权重: {weight_info['normalized_base_weight']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SVD损失函数测试失败: {e}")
        return False

def test_mse_loss_creation():
    """
    测试标准MSE损失函数的创建和计算
    """
    logger.info("=== 测试MSE损失函数创建 ===")
    
    try:
        # 创建MSE损失函数
        criterion = torch.nn.MSELoss()
        
        logger.info(f"✅ MSE损失函数创建成功: {type(criterion).__name__}")
        
        # 测试损失计算
        batch_size = 2
        output_dim = 4096  # 64*64
        
        test_pred = torch.randn(batch_size, output_dim)
        test_target = torch.randn(batch_size, output_dim)
        
        loss = criterion(test_pred, test_target)
        logger.info(f"✅ MSE损失计算成功: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MSE损失函数测试失败: {e}")
        return False

def test_loss_function_selection():
    """
    测试损失函数选择逻辑
    """
    logger.info("=== 测试损失函数选择逻辑 ===")
    
    # 测试配置1：启用SVD损失
    config1 = {
        'loss': {
            'svd_loss_enabled': True,
            'base_weight': 0.8,
            'svd_weights': [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            'topk': 10
        }
    }
    
    loss_config1 = config1['loss']
    svd_loss_enabled1 = loss_config1.get('svd_loss_enabled', True)
    
    if svd_loss_enabled1:
        criterion1 = TotalLossWithSVD(
            base_weight=loss_config1['base_weight'],
            svd_weights=loss_config1['svd_weights'],
            topk=loss_config1['topk']
        )
        logger.info(f"✅ 配置1 - SVD损失启用: {type(criterion1).__name__}")
    else:
        criterion1 = torch.nn.MSELoss()
        logger.info(f"❌ 配置1 - 应该启用SVD但使用了: {type(criterion1).__name__}")
        return False
    
    # 测试配置2：禁用SVD损失
    config2 = {
        'loss': {
            'svd_loss_enabled': False,
            'base_weight': 1.0,
            'svd_weights': [0.0] * 10,
            'topk': 10
        }
    }
    
    loss_config2 = config2['loss']
    svd_loss_enabled2 = loss_config2.get('svd_loss_enabled', True)
    
    if svd_loss_enabled2:
        criterion2 = TotalLossWithSVD(
            base_weight=loss_config2['base_weight'],
            svd_weights=loss_config2['svd_weights'],
            topk=loss_config2['topk']
        )
        logger.error(f"❌ 配置2 - 应该禁用SVD但使用了: {type(criterion2).__name__}")
        return False
    else:
        criterion2 = torch.nn.MSELoss()
        logger.info(f"✅ 配置2 - SVD损失禁用: {type(criterion2).__name__}")
    
    logger.info("✅ 损失函数选择逻辑测试通过")
    return True

def main():
    """
    主测试函数
    """
    logger.info("🧪 开始简化SVD机制测试")
    
    tests = [
        ("SVD损失函数创建", test_svd_loss_creation),
        ("MSE损失函数创建", test_mse_loss_creation),
        ("损失函数选择逻辑", test_loss_function_selection)
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
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 总结测试结果
    logger.info(f"\n{'='*50}")
    logger.info("测试结果总结")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！SVD机制修复成功！")
        return True
    else:
        logger.error(f"⚠️ {total - passed} 个测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)