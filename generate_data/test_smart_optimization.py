#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能多核优化测试脚本
验证修改后的智能worker数量设置是否正常工作
"""

import os
import sys
import yaml
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_smart_optimization():
    """测试智能优化逻辑"""
    
    # 加载配置文件
    config_path = "dynamic_config_server_downsampling.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"无法加载配置文件: {e}")
        return
    
    logger.info(f"🔧 测试智能优化逻辑 (CPU核心数: {os.cpu_count()})")
    
    # 提取配置
    data_config = config['data']
    dataloader_config = config.get('dataloader', {})
    
    # 模拟智能优化逻辑
    num_workers = dataloader_config.get('num_workers', 0)
    cpu_count = os.cpu_count()
    
    logger.info(f"初始配置: num_workers={num_workers}")
    logger.info(f"数据配置: batch_size={data_config.get('batch_size')}, downsampling={data_config.get('downsampling', {}).get('enabled')}")
    
    if num_workers == 0 and cpu_count > 4:
        # 检查是否启用智能优化
        enable_smart_workers = dataloader_config.get('enable_smart_workers', True)
        min_workers = dataloader_config.get('min_workers', 0)
        max_workers = dataloader_config.get('max_workers', 32)
        
        logger.info(f"智能优化配置: enable={enable_smart_workers}, min={min_workers}, max={max_workers}")
        
        if enable_smart_workers:
            # 根据数据处理复杂度智能设置worker数量
            use_downsampling = data_config.get('downsampling', {}).get('enabled', False)
            batch_size = data_config.get('batch_size', 16)
            
            logger.info(f"数据处理分析: 降采样={use_downsampling}, 批次大小={batch_size}")
            
            if use_downsampling:
                # 降采样需要更多CPU计算，使用更多worker
                if cpu_count >= 64:  # 超级服务器
                    num_workers = min(max_workers, max(8, cpu_count // 6))  # 使用1/6核心，最少8个
                    logger.info(f"[INFO] 超级服务器降采样模式: {cpu_count}核 → {num_workers}个worker (1/6核心)")
                else:  # 普通服务器
                    num_workers = min(max_workers, max(4, cpu_count // 4))  # 使用1/4核心，最少4个
                    logger.info(f"[INFO] 普通服务器降采样模式: {cpu_count}核 → {num_workers}个worker (1/4核心)")
            else:
                # 简单裁剪操作，使用较少worker避免进程开销
                if batch_size >= 64:  # 大批次可以受益于并行
                    num_workers = min(max_workers, max(2, cpu_count // 8))  # 使用1/8核心，最少2个
                    logger.info(f"[INFO] 大批次裁剪模式: {cpu_count}核 → {num_workers}个worker (1/8核心)")
                else:  # 小批次使用单进程
                    num_workers = max(min_workers, 0)
                    logger.info(f"[INFO] 小批次裁剪模式: {cpu_count}核 → {num_workers}个worker (单进程)")
        else:
            # 传统优化策略
            if cpu_count >= 64:  # 超级服务器
                num_workers = min(max_workers, cpu_count // 3)  # 使用1/3的核心
                logger.info(f"[INFO] 传统超级服务器模式: {cpu_count}核 → {num_workers}个worker (1/3核心)")
            else:  # 普通服务器
                num_workers = min(max_workers, cpu_count // 2)  # 使用1/2的核心
                logger.info(f"[INFO] 传统普通服务器模式: {cpu_count}核 → {num_workers}个worker (1/2核心)")
        
        # 确保在合理范围内
        final_workers = max(min_workers, min(max_workers, num_workers))
        
        if final_workers != num_workers:
            logger.info(f"[WARN] 调整到合理范围: {num_workers} → {final_workers}")
            num_workers = final_workers
        
        logger.info(f"[INFO] 最终设置: num_workers={num_workers}")
        
        # 分析结果
        cpu_utilization = (num_workers / cpu_count) * 100
        logger.info(f"[INFO] 预期CPU利用率: {cpu_utilization:.1f}% ({num_workers}/{cpu_count}核心)")
        
        if num_workers == 0:
            logger.info("[TIP] 建议: 单进程模式，适合轻量级数据处理")
        elif num_workers <= 4:
            logger.info("[TIP] 建议: 低并行度，适合简单数据处理")
        elif num_workers <= 16:
            logger.info("[TIP] 建议: 中等并行度，适合中等复杂度数据处理")
        else:
            logger.info("[TIP] 建议: 高并行度，适合复杂数据处理（如降采样）")
    
    else:
        logger.info("[INFO] 不满足自动优化条件，使用配置文件设置")
    
    return num_workers

def test_different_scenarios():
    """测试不同场景下的优化结果"""
    
    logger.info("\n[TEST] 测试不同场景下的智能优化")
    
    scenarios = [
        {
            'name': '降采样 + 大批次',
            'downsampling': True,
            'batch_size': 256
        },
        {
            'name': '降采样 + 小批次',
            'downsampling': True,
            'batch_size': 32
        },
        {
            'name': '裁剪 + 大批次',
            'downsampling': False,
            'batch_size': 128
        },
        {
            'name': '裁剪 + 小批次',
            'downsampling': False,
            'batch_size': 16
        }
    ]
    
    cpu_count = os.cpu_count()
    max_workers = 32
    min_workers = 0
    
    logger.info(f"系统配置: {cpu_count}核CPU, max_workers={max_workers}")
    logger.info("-" * 60)
    
    for scenario in scenarios:
        logger.info(f"\n[INFO] 场景: {scenario['name']}")
        
        use_downsampling = scenario['downsampling']
        batch_size = scenario['batch_size']
        
        if use_downsampling:
            # 降采样需要更多CPU计算
            if cpu_count >= 64:  # 超级服务器
                num_workers = min(max_workers, max(8, cpu_count // 6))
            else:  # 普通服务器
                num_workers = min(max_workers, max(4, cpu_count // 4))
        else:
            # 简单裁剪操作
            if batch_size >= 64:  # 大批次
                num_workers = min(max_workers, max(2, cpu_count // 8))
            else:  # 小批次
                num_workers = max(min_workers, 0)
        
        # 确保在合理范围内
        num_workers = max(min_workers, min(max_workers, num_workers))
        
        cpu_utilization = (num_workers / cpu_count) * 100
        
        logger.info(f"  配置: 降采样={use_downsampling}, 批次={batch_size}")
        logger.info(f"  结果: {num_workers}个worker, CPU利用率={cpu_utilization:.1f}%")

def main():
    """主函数"""
    logger.info("[INFO] 开始智能优化测试")
    
    # 测试当前配置
    optimal_workers = test_smart_optimization()
    
    # 测试不同场景
    test_different_scenarios()
    
    logger.info("\n[OK] 智能优化测试完成")
    logger.info(f"当前配置推荐: {optimal_workers}个worker")

if __name__ == "__main__":
    main()