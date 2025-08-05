#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分辨率降采样器使用演示

功能:
1. 演示如何使用新的分辨率降采样器
2. 对比降采样和裁剪两种方法的效果
3. 展示不同降采样方法的差异
4. 提供完整的使用示例

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模块
from resolution_downsampler import ResolutionDownsampler, DownsampledResolutionDataset

# 简化配置加载
def load_config():
    """加载简化的配置"""
    return {
        'data': {
            'input_resolution': [64, 64],
            'output_resolution': [32, 32],
            'num_samples': 100,
            'normalize_data': True,
            'lazy_loading': False
        },
        'downsampler': {
            'method': 'bilinear',
            'preserve_aspect_ratio': True
        }
    }

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_test_data(shape: Tuple[int, int, int] = (10, 128, 128)) -> np.ndarray:
    """
    创建测试数据
    
    Args:
        shape: 数据形状 (batch_size, height, width)
        
    Returns:
        测试数据
    """
    batch_size, height, width = shape
    
    # 创建具有不同频率成分的测试数据
    data = np.zeros(shape, dtype=np.float32)
    
    for i in range(batch_size):
        # 创建网格
        x = np.linspace(0, 2*np.pi, width)
        y = np.linspace(0, 2*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # 生成复杂的波形模式
        freq1 = 2 + i * 0.5  # 不同的频率
        freq2 = 3 + i * 0.3
        
        pattern = (np.sin(freq1 * X) * np.cos(freq2 * Y) + 
                  0.5 * np.sin(2 * freq1 * X + freq2 * Y) +
                  0.3 * np.cos(freq1 * X - freq2 * Y))
        
        # 添加一些噪声
        noise = 0.1 * np.random.randn(height, width)
        
        data[i] = pattern + noise
    
    logger.info(f"创建测试数据: {shape}")
    logger.info(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
    
    return data

def compare_methods(original_data: np.ndarray, 
                   target_size: Tuple[int, int],
                   sample_idx: int = 0) -> Dict[str, np.ndarray]:
    """
    比较不同降采样方法的效果
    
    Args:
        original_data: 原始数据
        target_size: 目标尺寸
        sample_idx: 要比较的样本索引
        
    Returns:
        不同方法的结果字典
    """
    results = {}
    methods = ['nearest', 'bilinear', 'bicubic', 'area']
    
    # 获取单个样本
    sample = original_data[sample_idx:sample_idx+1]
    
    for method in methods:
        try:
            downsampler = ResolutionDownsampler(
                downsample_method=method,
                preserve_aspect_ratio=False
            )
            
            downsampled = downsampler.downsample_data(sample, target_size, method='scipy')
            results[method] = downsampled[0]
            
            logger.info(f"{method} 降采样完成: {sample.shape} -> {downsampled.shape}")
            
        except Exception as e:
            logger.error(f"{method} 降采样失败: {e}")
            results[method] = None
    
    return results

def simulate_crop_method(data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    模拟裁剪方法（用于对比）
    
    Args:
        data: 输入数据
        target_size: 目标尺寸
        
    Returns:
        裁剪后的数据
    """
    _, orig_h, orig_w = data.shape
    target_h, target_w = target_size
    
    if target_h > orig_h or target_w > orig_w:
        raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 ({orig_h}, {orig_w})")
    
    # 中心裁剪
    start_h = (orig_h - target_h) // 2
    start_w = (orig_w - target_w) // 2
    end_h = start_h + target_h
    end_w = start_w + target_w
    
    return data[:, start_h:end_h, start_w:end_w]

def visualize_comparison(original: np.ndarray,
                        downsampled_results: Dict[str, np.ndarray],
                        cropped: np.ndarray,
                        save_path: str = None):
    """
    可视化比较结果
    
    Args:
        original: 原始数据
        downsampled_results: 降采样结果字典
        cropped: 裁剪结果
        save_path: 保存路径
    """
    # 计算子图数量
    num_methods = len([v for v in downsampled_results.values() if v is not None])
    total_plots = num_methods + 2  # 原始 + 裁剪 + 降采样方法
    
    cols = 3
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    # 显示原始数据
    if plot_idx < len(axes.flat):
        im = axes.flat[plot_idx].imshow(original, cmap='viridis')
        axes.flat[plot_idx].set_title(f'原始数据 {original.shape}')
        axes.flat[plot_idx].axis('off')
        plt.colorbar(im, ax=axes.flat[plot_idx])
        plot_idx += 1
    
    # 显示裁剪结果
    if plot_idx < len(axes.flat) and cropped is not None:
        im = axes.flat[plot_idx].imshow(cropped[0], cmap='viridis')
        axes.flat[plot_idx].set_title(f'裁剪方法 {cropped.shape[1:]}\n(信息丢失)')
        axes.flat[plot_idx].axis('off')
        plt.colorbar(im, ax=axes.flat[plot_idx])
        plot_idx += 1
    
    # 显示降采样结果
    for method, result in downsampled_results.items():
        if result is not None and plot_idx < len(axes.flat):
            im = axes.flat[plot_idx].imshow(result, cmap='viridis')
            axes.flat[plot_idx].set_title(f'{method.upper()} 降采样\n{result.shape}\n(保持全局信息)')
            axes.flat[plot_idx].axis('off')
            plt.colorbar(im, ax=axes.flat[plot_idx])
            plot_idx += 1
    
    # 隐藏多余的子图
    for i in range(plot_idx, len(axes.flat)):
        axes.flat[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def analyze_information_preservation(original: np.ndarray,
                                   processed: np.ndarray,
                                   method_name: str) -> Dict[str, float]:
    """
    分析信息保持程度
    
    Args:
        original: 原始数据
        processed: 处理后的数据
        method_name: 方法名称
        
    Returns:
        分析结果字典
    """
    # 将处理后的数据调整到原始尺寸进行比较
    from scipy.ndimage import zoom
    
    orig_h, orig_w = original.shape
    proc_h, proc_w = processed.shape
    
    # 上采样到原始尺寸
    zoom_h = orig_h / proc_h
    zoom_w = orig_w / proc_w
    upsampled = zoom(processed, (zoom_h, zoom_w), order=1)
    
    # 计算各种指标
    mse = np.mean((original - upsampled) ** 2)
    mae = np.mean(np.abs(original - upsampled))
    
    # 计算相关系数
    correlation = np.corrcoef(original.flatten(), upsampled.flatten())[0, 1]
    
    # 计算频域保持程度
    orig_fft = np.fft.fft2(original)
    proc_fft = np.fft.fft2(upsampled)
    
    # 频域相似度
    freq_similarity = np.abs(np.corrcoef(
        np.abs(orig_fft).flatten(), 
        np.abs(proc_fft).flatten()
    )[0, 1])
    
    results = {
        'method': method_name,
        'mse': float(mse),
        'mae': float(mae),
        'correlation': float(correlation),
        'freq_similarity': float(freq_similarity),
        'compression_ratio': float((orig_h * orig_w) / (proc_h * proc_w))
    }
    
    logger.info(f"{method_name} 信息保持分析:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  相关系数: {correlation:.6f}")
    logger.info(f"  频域相似度: {freq_similarity:.6f}")
    logger.info(f"  压缩比: {results['compression_ratio']:.2f}x")
    
    return results

def demo_downsampler_dataset():
    """
    演示降采样数据集的使用
    """
    logger.info("=== 降采样数据集演示 ===")
    
    # 使用简化的配置加载
    config = load_config()
    
    # 创建测试数据文件（如果原始数据文件不存在）
    data_path = config['data'].get('path', 'test_data.h5')
    
    if not Path(data_path).exists():
        logger.info("原始数据文件不存在，创建测试数据")
        
        # 创建测试数据
        test_data = create_test_data((50, 128, 128))
        
        # 保存为HDF5格式
        import h5py
        test_data_path = Path(__file__).parent / 'test_downsampler_data.h5'
        
        with h5py.File(test_data_path, 'w') as f:
            f.create_dataset('tensor', data=test_data)
        
        data_path = str(test_data_path)
        logger.info(f"测试数据已保存到: {data_path}")
    
    try:
        # 创建降采样数据集
        dataset = DownsampledResolutionDataset(
            data_path=data_path,
            input_resolution=tuple(config['data']['output_resolution']),  # 使用output作为input
            output_resolution=tuple(config['data']['input_resolution']),   # 使用input作为output
            num_samples=config['data']['num_samples'],
            downsample_method='bilinear',  # 使用scipy兼容的方法
            preserve_aspect_ratio=config['downsampler']['preserve_aspect_ratio'],
            normalize_data=config['data']['normalize_data'],
            lazy_loading=config['data']['lazy_loading']
        )
        
        # 获取数据统计信息
        stats = dataset.get_data_statistics()
        logger.info("数据集统计信息:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # 测试数据加载
        logger.info("\n测试数据加载:")
        for i in range(min(3, len(dataset))):
            input_data, output_data, idx = dataset[i]
            logger.info(f"样本 {i}:")
            logger.info(f"  输入形状: {input_data.shape}")
            logger.info(f"  输出形状: {output_data.shape}")
            logger.info(f"  输入范围: [{input_data.min():.4f}, {input_data.max():.4f}]")
            logger.info(f"  输出范围: [{output_data.min():.4f}, {output_data.max():.4f}]")
        
        logger.info("✅ 降采样数据集演示完成")
        
    except Exception as e:
        logger.error(f"降采样数据集演示失败: {e}")
        raise

def main():
    """
    主演示函数
    """
    logger.info("=== 分辨率降采样器完整演示 ===")
    
    # 1. 创建测试数据
    logger.info("\n1. 创建测试数据")
    test_data = create_test_data((5, 128, 128))
    
    # 2. 比较不同降采样方法
    logger.info("\n2. 比较不同降采样方法")
    target_size = (64, 64)
    sample_idx = 0
    
    downsampled_results = compare_methods(test_data, target_size, sample_idx)
    
    # 3. 模拟裁剪方法进行对比
    logger.info("\n3. 对比裁剪方法")
    try:
        cropped_result = simulate_crop_method(test_data[sample_idx:sample_idx+1], target_size)
        logger.info(f"裁剪结果形状: {cropped_result.shape}")
    except Exception as e:
        logger.error(f"裁剪方法失败: {e}")
        cropped_result = None
    
    # 4. 可视化比较
    logger.info("\n4. 可视化比较")
    save_path = Path(__file__).parent / 'downsampler_comparison.png'
    
    try:
        visualize_comparison(
            original=test_data[sample_idx],
            downsampled_results=downsampled_results,
            cropped=cropped_result,
            save_path=str(save_path)
        )
    except Exception as e:
        logger.error(f"可视化失败: {e}")
    
    # 5. 信息保持分析
    logger.info("\n5. 信息保持分析")
    original_sample = test_data[sample_idx]
    
    analysis_results = []
    
    # 分析裁剪方法
    if cropped_result is not None:
        crop_analysis = analyze_information_preservation(
            original_sample, cropped_result[0], "裁剪方法"
        )
        analysis_results.append(crop_analysis)
    
    # 分析降采样方法
    for method, result in downsampled_results.items():
        if result is not None:
            analysis = analyze_information_preservation(
                original_sample, result, f"{method}降采样"
            )
            analysis_results.append(analysis)
    
    # 6. 演示数据集使用
    logger.info("\n6. 演示数据集使用")
    try:
        demo_downsampler_dataset()
    except Exception as e:
        logger.error(f"数据集演示失败: {e}")
    
    # 7. 总结
    logger.info("\n=== 演示总结 ===")
    logger.info("✅ 分辨率降采样器演示完成")
    logger.info("\n主要优势:")
    logger.info("  1. 保持全局信息，避免裁剪造成的信息丢失")
    logger.info("  2. 支持多种插值方法，适应不同需求")
    logger.info("  3. 可以处理任意尺寸变换，不受原始尺寸限制")
    logger.info("  4. 更好的频域特征保持")
    logger.info("\n建议使用场景:")
    logger.info("  - 需要保持全局信息的超分辨率任务")
    logger.info("  - 多尺度特征学习")
    logger.info("  - 图像压缩和重建")
    logger.info("  - 物理场数据的多分辨率分析")

if __name__ == "__main__":
    # 设置结果目录
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 运行演示
    main()