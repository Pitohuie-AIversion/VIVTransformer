#!/usr/bin/env python3
"""
SVD 反投影功能演示脚本
展示如何使用新添加的 inverse_project_output, inverse_project_input, 
reconstruct_output, reconstruct_input 方法进行数据重建

Usage:
    python demo_svd_reconstruction.py [--data-path path/to/data.h5] [--config path/to/config.yaml]
"""

import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免阻塞
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加父目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入必要模块（仅导入 SVD 投影器）
from pde_process.svd_modal_projection import SVDModalProjector, SVDProjectionConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_data(batch_size=5, input_size=(32, 32), output_size=(64, 64)):
    """创建演示用的合成数据"""
    
    # 生成具有空间结构的合成数据
    input_h, input_w = input_size
    output_h, output_w = output_size
    
    input_data = []
    output_data = []
    
    for i in range(batch_size):
        # 输入：高斯分布 + 正弦波
        x = np.linspace(0, 2*np.pi, input_w)
        y = np.linspace(0, 2*np.pi, input_h)
        X, Y = np.meshgrid(x, y)
        
        input_field = np.sin(X + i*0.5) * np.cos(Y + i*0.3) + 0.2 * np.random.randn(input_h, input_w)
        input_data.append(input_field)
        
        # 输出：输入的双线性上采样 + 非线性变换
        from scipy.ndimage import zoom
        output_field = zoom(input_field, (output_h/input_h, output_w/input_w), order=1)
        output_field = np.tanh(output_field * 1.5)  # 非线性变换
        output_data.append(output_field)
    
    return np.array(input_data), np.array(output_data)

def demo_svd_reconstruction(output_dir: Path):
    """演示 SVD 反投影功能"""
    
    logger.info("=== SVD 反投影功能演示 ===")
    
    # 1. 创建演示数据
    logger.info("1. 创建演示数据...")
    input_data, output_data = create_demo_data(batch_size=10, input_size=(32, 32), output_size=(64, 64))
    logger.info(f"输入数据形状: {input_data.shape}")
    logger.info(f"输出数据形状: {output_data.shape}")
    
    # 展平数据用于 SVD 训练
    input_flat = input_data.reshape(input_data.shape[0], -1)  # (10, 1024)
    output_flat = output_data.reshape(output_data.shape[0], -1)  # (10, 4096)
    
    # 2. 创建 SVD 投影器
    logger.info("2. 创建和训练 SVD 投影器...")
    config = SVDProjectionConfig(
        n_modes=50,  # 目标潜在维度
        standardize_data=True,
        energy_threshold=0.95,
        auto_select_modes=True
    )
    
    projector = SVDModalProjector(config)
    projector.fit(input_flat, output_flat)
    
    logger.info(f"SVD 投影器训练完成:")
    logger.info(f"  - 潜在维度: {projector.latent_dim}")
    logger.info(f"  - 输入降维: {input_flat.shape[1]} -> {projector.latent_dim}")
    logger.info(f"  - 输出降维: {output_flat.shape[1]} -> {projector.latent_dim}")
    
    # 3. 模拟数据集初始化（设置 SVD 投影器）
    logger.info("3. 创建模拟数据集实例...")
    
    # 创建一个模拟的数据集实例来测试反投影方法
    class MockDataset:
        def __init__(self, projector, input_resolution, output_resolution):
            self.svd_projector = projector
            self.use_svd_projection = True
            self.normalize_data = True
            self.input_resolution = input_resolution
            self.output_resolution = output_resolution
            
            # 模拟归一化参数
            self.global_min = np.min([input_flat.min(), output_flat.min()])
            self.global_max = np.max([input_flat.max(), output_flat.max()])
            self.input_min = self.global_min
            self.input_max = self.global_max
            self.output_min = self.global_min
            self.output_max = self.global_max
        
        def _denormalize_data(self, data, data_min, data_max):
            """反归一化数据"""
            return data * (data_max - data_min) + data_min
    
    dataset = MockDataset(projector, (32, 32), (64, 64))
    
    # 绑定我们的新方法到数据集实例（避免导入整个备份训练器）
    from dynamic_resolution_trainer_backup import (
        inverse_project_output, inverse_project_input,
        reconstruct_output, reconstruct_input
    )
    
    # 绑定方法
    dataset.inverse_project_output = inverse_project_output.__get__(dataset, MockDataset)
    dataset.inverse_project_input = inverse_project_input.__get__(dataset, MockDataset)
    dataset.reconstruct_output = reconstruct_output.__get__(dataset, MockDataset)
    dataset.reconstruct_input = reconstruct_input.__get__(dataset, MockDataset)
    
    logger.info("✅ 数据集实例已配置 SVD 反投影方法")
    
    # 4. 进行前向投影
    logger.info("4. 进行前向投影...")
    input_projected = projector.transform_input(input_flat)
    output_projected = projector.transform_output(output_flat)
    
    logger.info(f"投影后形状:")
    logger.info(f"  - 输入投影: {input_projected.shape}")
    logger.info(f"  - 输出投影: {output_projected.shape}")
    
    # 5. 测试基础反投影方法
    logger.info("5. 测试基础反投影方法...")
    
    # 测试输出反投影
    output_reconstructed_basic = dataset.inverse_project_output(output_projected)
    logger.info(f"输出基础反投影: {output_projected.shape} -> {output_reconstructed_basic.shape}")
    
    # 测试输入反投影
    input_reconstructed_basic = dataset.inverse_project_input(input_projected)
    logger.info(f"输入基础反投影: {input_projected.shape} -> {input_reconstructed_basic.shape}")
    
    # 6. 测试完整重建方法
    logger.info("6. 测试完整重建方法...")
    
    # 测试输出完整重建（反投影 + 反归一化 + 空间重塑）
    output_reconstructed_full = dataset.reconstruct_output(
        output_projected, 
        denormalize=True, 
        reshape_to_spatial=True
    )
    logger.info(f"输出完整重建: {output_projected.shape} -> {output_reconstructed_full.shape}")
    
    # 测试输入完整重建
    input_reconstructed_full = dataset.reconstruct_input(
        input_projected, 
        denormalize=True, 
        reshape_to_spatial=True
    )
    logger.info(f"输入完整重建: {input_projected.shape} -> {input_reconstructed_full.shape}")
    
    # 7. 计算重建误差
    logger.info("7. 计算重建误差...")
    
    # 对于展平数据的误差
    input_mse_flat = np.mean((input_flat - input_reconstructed_basic)**2)
    output_mse_flat = np.mean((output_flat - output_reconstructed_basic)**2)
    
    # 对于空间数据的误差
    input_mse_spatial = np.mean((input_data - input_reconstructed_full)**2)
    output_mse_spatial = np.mean((output_data - output_reconstructed_full)**2)
    
    logger.info(f"重建误差统计:")
    logger.info(f"  - 输入展平数据 MSE: {input_mse_flat:.6f}")
    logger.info(f"  - 输出展平数据 MSE: {output_mse_flat:.6f}")
    logger.info(f"  - 输入空间数据 MSE: {input_mse_spatial:.6f}")
    logger.info(f"  - 输出空间数据 MSE: {output_mse_spatial:.6f}")
    
    # 相对误差
    input_relative_error = np.sqrt(input_mse_spatial) / (np.std(input_data) + 1e-8)
    output_relative_error = np.sqrt(output_mse_spatial) / (np.std(output_data) + 1e-8)
    
    logger.info(f"  - 输入相对误差: {input_relative_error:.4f}")
    logger.info(f"  - 输出相对误差: {output_relative_error:.4f}")
    
    # 8. 可视化结果
    logger.info("8. 可视化重建结果...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 选择第一个样本进行可视化
    sample_idx = 0
    
    # 输入数据对比
    axes[0, 0].imshow(input_data[sample_idx], cmap='viridis')
    axes[0, 0].set_title('原始输入')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(input_reconstructed_full[sample_idx], cmap='viridis')
    axes[0, 1].set_title('重建输入')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(input_data[sample_idx] - input_reconstructed_full[sample_idx], cmap='RdBu')
    axes[0, 2].set_title('输入误差')
    axes[0, 2].axis('off')
    
    axes[0, 3].hist((input_data - input_reconstructed_full).flatten(), bins=50, alpha=0.7)
    axes[0, 3].set_title('输入误差分布')
    axes[0, 3].set_xlabel('误差值')
    axes[0, 3].set_ylabel('频次')
    
    # 输出数据对比
    axes[1, 0].imshow(output_data[sample_idx], cmap='viridis')
    axes[1, 0].set_title('原始输出')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(output_reconstructed_full[sample_idx], cmap='viridis')
    axes[1, 1].set_title('重建输出')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(output_data[sample_idx] - output_reconstructed_full[sample_idx], cmap='RdBu')
    axes[1, 2].set_title('输出误差')
    axes[1, 2].axis('off')
    
    axes[1, 3].hist((output_data - output_reconstructed_full).flatten(), bins=50, alpha=0.7)
    axes[1, 3].set_title('输出误差分布')
    axes[1, 3].set_xlabel('误差值')
    axes[1, 3].set_ylabel('频次')
    
    # 投影维度对比
    axes[2, 0].plot(input_projected[sample_idx], 'b-', label='输入投影', alpha=0.7)
    axes[2, 0].plot(output_projected[sample_idx], 'r-', label='输出投影', alpha=0.7)
    axes[2, 0].set_title('潜在空间表示')
    axes[2, 0].set_xlabel('潜在维度')
    axes[2, 0].set_ylabel('投影值')
    axes[2, 0].legend()
    
    # 重建质量统计
    axes[2, 1].bar(['输入', '输出'], [input_relative_error, output_relative_error])
    axes[2, 1].set_title('相对重建误差')
    axes[2, 1].set_ylabel('相对误差')
    
    # 数据压缩比
    original_size = input_flat.shape[1] + output_flat.shape[1]
    compressed_size = projector.latent_dim * 2
    compression_ratio = original_size / compressed_size
    
    axes[2, 2].bar(['原始', '压缩'], [original_size, compressed_size])
    axes[2, 2].set_title(f'数据压缩 (压缩比: {compression_ratio:.1f}x)')
    axes[2, 2].set_ylabel('维度数')
    
    # 奇异值分布
    input_sv = projector.input_singular_values[:20]  # 前20个奇异值
    output_sv = projector.output_singular_values[:20]
    
    axes[2, 3].semilogy(input_sv, 'b-o', label='输入奇异值', markersize=4)
    axes[2, 3].semilogy(output_sv, 'r-s', label='输出奇异值', markersize=4)
    axes[2, 3].set_title('主要奇异值')
    axes[2, 3].set_xlabel('成分索引')
    axes[2, 3].set_ylabel('奇异值 (对数)')
    axes[2, 3].legend()
    
    plt.tight_layout()
    
    # 保存图像
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / 'svd_reconstruction_demo.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()  # 使用非交互式后端时无需显示
    
    logger.info(f"📊 可视化结果已保存: {save_path}")
    
    # 9. 保存演示结果
    logger.info("9. 保存演示结果...")
    
    results = {
        'compression_ratio': float(compression_ratio),
        'latent_dim': int(projector.latent_dim),
        'input_relative_error': float(input_relative_error),
        'output_relative_error': float(output_relative_error),
        'input_mse': float(input_mse_spatial),
        'output_mse': float(output_mse_spatial),
    }
    
    import json
    results_path = output_dir / 'svd_reconstruction_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📊 演示结果已保存: {results_path}")
    
    logger.info("🎉 SVD 反投影功能演示完成!")
    logger.info(f"✅ 压缩比: {compression_ratio:.1f}x")
    logger.info(f"✅ 输入重建相对误差: {input_relative_error:.4f}")
    logger.info(f"✅ 输出重建相对误差: {output_relative_error:.4f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SVD 反投影功能演示')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='输出目录 (默认: ./results)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        demo_svd_reconstruction(output_dir)
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()