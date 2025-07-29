#!/usr/bin/env python3
"""
测试PDEBench数据集集成
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from data.dataloader import get_loaders
from data.pdebench_dataset import PDEBenchDataset
import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pdebench_dataset():
    """测试PDEBench数据集类"""
    logger.info("=== 测试PDEBench数据集类 ===")
    
    # PDEBench数据文件路径
    pdebench_data_path = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
    
    if not Path(pdebench_data_path).exists():
        logger.warning(f"PDEBench数据文件不存在: {pdebench_data_path}")
        return False
        
    try:
        # 创建数据集（限制样本数量以加快测试）
        dataset = PDEBenchDataset(pdebench_data_path, max_samples=100)
        
        logger.info(f"数据集大小: {len(dataset)}")
        
        # 获取统计信息
        stats = dataset.get_data_statistics()
        logger.info(f"数据集统计: {stats}")
        
        # 测试数据加载
        if len(dataset) > 0:
            sample = dataset[0]
            input_data, target_data, time_step = sample
            
            logger.info(f"样本形状 - 输入: {input_data.shape}, 目标: {target_data.shape}, 时间步: {time_step}")
            logger.info(f"数据类型 - 输入: {input_data.dtype}, 目标: {target_data.dtype}")
            
            return True
        else:
            logger.warning("数据集为空")
            return False
            
    except Exception as e:
        logger.error(f"测试PDEBench数据集时出错: {e}")
        return False


def test_dataloader_integration():
    """测试数据加载器集成"""
    logger.info("=== 测试数据加载器集成 ===")
    
    # 测试原有的压力数据集
    pressure_data_path = "../merged_all_pressures_separated_normalized.pt"
    
    if Path(pressure_data_path).exists():
        try:
            logger.info("测试压力数据集...")
            train_loader, valid_loader, test_loader = get_loaders(
                pressure_data_path, 
                batch_size=32, 
                dataset_type="auto",
                max_samples=100
            )
            
            # 测试一个批次
            batch = next(iter(train_loader))
            logger.info(f"压力数据集批次形状: {[x.shape if hasattr(x, 'shape') else type(x) for x in batch]}")
            
        except Exception as e:
            logger.error(f"测试压力数据集时出错: {e}")
    
    # 测试PDEBench数据集
    pdebench_data_path = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
    
    if Path(pdebench_data_path).exists():
        try:
            logger.info("测试PDEBench数据集...")
            train_loader, valid_loader, test_loader = get_loaders(
                pdebench_data_path, 
                batch_size=16, 
                dataset_type="pdebench",
                max_samples=50
            )
            
            # 测试一个批次
            batch = next(iter(train_loader))
            logger.info(f"PDEBench数据集批次形状: {[x.shape if hasattr(x, 'shape') else type(x) for x in batch]}")
            
            return True
            
        except Exception as e:
            logger.error(f"测试PDEBench数据集时出错: {e}")
            return False
    else:
        logger.warning(f"PDEBench数据文件不存在: {pdebench_data_path}")
        return False


def test_auto_detection():
    """测试自动数据格式检测"""
    logger.info("=== 测试自动数据格式检测 ===")
    
    test_cases = [
        ("../merged_all_pressures_separated_normalized.pt", "pressure"),
        ("x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5", "pdebench"),
        ("x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/", "pdebench")
    ]
    
    for data_path, expected_type in test_cases:
        if Path(data_path).exists():
            try:
                logger.info(f"测试路径: {data_path}")
                train_loader, valid_loader, test_loader = get_loaders(
                    data_path, 
                    batch_size=8, 
                    dataset_type="auto",
                    max_samples=20
                )
                logger.info(f"✓ 成功检测并加载 {expected_type} 格式数据")
                
            except Exception as e:
                logger.error(f"✗ 自动检测失败: {e}")
        else:
            logger.warning(f"路径不存在: {data_path}")


def main():
    """主测试函数"""
    logger.info("开始PDEBench集成测试...")
    
    # 运行测试
    tests = [
        test_pdebench_dataset,
        test_dataloader_integration,
        test_auto_detection
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"测试 {test_func.__name__} 时出现异常: {e}")
            results.append(False)
        
        logger.info("-" * 50)
    
    # 总结
    passed = sum(1 for r in results if r is True)
    total = len(results)
    
    logger.info(f"测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("✓ 所有测试通过！PDEBench集成成功")
    else:
        logger.warning("⚠ 部分测试失败，请检查配置")


if __name__ == "__main__":
    main()