#!/usr/bin/env python3
"""
PDEBench数据集快速测试脚本
验证数据加载功能和基本训练流程
"""

import sys
import os
import logging
import torch
from pathlib import Path
import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent / "modify_multi_attention"))

# 设置环境变量
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_pdebench_quick.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_pdebench_data_loading():
    """测试PDEBench数据加载功能"""
    logger.info("=== PDEBench数据加载测试 ===")
    
    try:
        from data.dataloader import get_loaders
        from data.pdebench_dataset import PDEBenchDataset
        
        # 测试数据路径
        test_paths = [
            "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5",
            "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/",
            "data/toy"  # 备用toy数据
        ]
        
        for data_path in test_paths:
            if Path(data_path).exists() or data_path == "data/toy":
                logger.info(f"测试数据路径: {data_path}")
                
                try:
                    # 测试数据加载器
                    train_loader, val_loader, test_loader = get_loaders(
                        data_path=data_path,
                        batch_size=2,
                        max_samples=10  # 限制样本数量进行快速测试
                    )
                    
                    logger.info(f"[PASS] 数据加载器创建成功")
                    logger.info(f"  训练集批次数: {len(train_loader)}")
                    logger.info(f"  验证集批次数: {len(val_loader)}")
                    logger.info(f"  测试集批次数: {len(test_loader)}")
                    
                    # 测试一个批次
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        logger.info(f"  批次 {batch_idx}: 输入形状 {inputs.shape}, 目标形状 {targets.shape}")
                        if batch_idx >= 2:  # 只测试前3个批次
                            break
                    
                    return True
                    
                except Exception as e:
                    logger.warning(f"数据路径 {data_path} 测试失败: {e}")
                    continue
        
        logger.error("所有数据路径测试失败")
        return False
        
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        return False


def test_model_initialization():
    """测试模型初始化"""
    logger.info("=== 模型初始化测试 ===")
    
    try:
        from models.enhanced_transformer import EnhancedTransformer1d
        
        # 创建简单的模型配置
        model = EnhancedTransformer1d(
            input_channels=1,
            output_channels=1,
            d_model=128,
            num_heads=4,
            num_layers=2,
            input_resolution=20,  # 20个点
            output_resolution=200,  # 200个点
            attention_type='simplified_self_attention'
        )
        
        logger.info(f"[PASS] 模型创建成功")
        logger.info(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        test_input = torch.randn(2, 20, 1)  # [batch_size, input_resolution, input_channels]
        with torch.no_grad():
            output = model(test_input)
        
        logger.info(f"[PASS] 前向传播测试成功")
        logger.info(f"  输入形状: {test_input.shape}")
        logger.info(f"  输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"模型初始化测试失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("开始PDEBench快速测试")
    start_time = datetime.datetime.now()
    
    results = {
        "data_loading": test_pdebench_data_loading(),
        "model_initialization": test_model_initialization()
    }
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    logger.info("=== 测试结果汇总 ===")
    for test_name, result in results.items():
        status = "[PASS] 通过" if result else "[FAIL] 失败"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"测试总耗时: {duration.total_seconds():.2f}秒")
    
    # 保存测试结果
    with open("test_pdebench_results.txt", "w", encoding="utf-8") as f:
        f.write(f"PDEBench快速测试结果\n")
        f.write(f"测试时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试耗时: {duration.total_seconds():.2f}秒\n\n")
        
        for test_name, result in results.items():
            status = "通过" if result else "失败"
            f.write(f"{test_name}: {status}\n")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)