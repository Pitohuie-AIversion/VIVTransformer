#!/usr/bin/env python3
"""
PDEBench数据集训练演示脚本
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from main import main
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pdebench_demo():
    """运行PDEBench数据集演示"""
    logger.info("=== PDEBench数据集训练演示 ===")
    
    # 检查PDEBench配置文件是否存在
    config_path = Path("configs/config_pdebench.yaml")
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return False
        
    # 检查PDEBench数据文件是否存在
    data_path = Path("x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5")
    if not data_path.exists():
        logger.error(f"PDEBench数据文件不存在: {data_path}")
        logger.info("请确保PDEBench数据已下载到正确位置")
        return False
        
    try:
        # 模拟命令行参数
        sys.argv = ['run_pdebench_demo.py', '--config', str(config_path)]
        
        logger.info(f"使用配置文件: {config_path}")
        logger.info(f"使用数据文件: {data_path}")
        logger.info("开始训练...")
        
        # 调用主训练函数
        main()
        
        logger.info("✓ PDEBench演示训练完成")
        return True
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        return False


def quick_test():
    """快速测试PDEBench数据加载"""
    logger.info("=== 快速测试PDEBench数据加载 ===")
    
    try:
        from data.dataloader import get_loaders
        
        data_path = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
        
        if not Path(data_path).exists():
            logger.error(f"数据文件不存在: {data_path}")
            return False
            
        # 加载少量数据进行测试
        train_loader, valid_loader, test_loader = get_loaders(
            data_path,
            batch_size=8,
            dataset_type="pdebench",
            max_samples=50
        )
        
        # 测试一个批次
        batch = next(iter(train_loader))
        input_data, target_data, time_steps = batch
        
        logger.info(f"✓ 成功加载PDEBench数据")
        logger.info(f"  批次大小: {input_data.shape[0]}")
        logger.info(f"  输入形状: {input_data.shape}")
        logger.info(f"  目标形状: {target_data.shape}")
        logger.info(f"  时间步形状: {time_steps.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"快速测试失败: {e}")
        return False


def main_demo():
    """主演示函数"""
    parser = argparse.ArgumentParser(description='PDEBench数据集演示')
    parser.add_argument('--mode', choices=['test', 'train'], default='test',
                       help='运行模式: test(快速测试) 或 train(完整训练)')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        success = quick_test()
    else:
        success = run_pdebench_demo()
        
    if success:
        logger.info("[OK] 演示成功完成！")
    else:
        logger.error("[ERROR] 演示失败，请检查配置和数据文件")
        
    return success


if __name__ == "__main__":
    main_demo()