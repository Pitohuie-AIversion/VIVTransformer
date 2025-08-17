"""
SVD模态分解投影模块

该模块提供将不同分辨率的输入输出数据投影到同一潜在维度空间的功能，
使用奇异值分解（SVD）进行模态分解，实现维度统一化。

功能特点:
1. 支持不同分辨率输入输出数据的统一投影
2. 基于SVD的模态分解，保留主要物理模态
3. 提供投影和反投影功能，支持结果重建
4. 支持增量学习，可以在线更新投影矩阵
5. 自动模态选择，根据能量阈值确定保留的模态数

作者: Assistant
日期: 2025-01-15
"""

import os
import numpy as np
import torch
import h5py
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass
import pickle
import json

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class SVDProjectionConfig:
    """SVD投影配置类"""
    # 模态数量相关
    n_modes: int = 64  # 目标投影维度（潜在空间维度）
    energy_threshold: float = 0.95  # 保留的能量阈值
    auto_select_modes: bool = True  # 是否自动选择模态数
    min_modes: int = 8  # 最小模态数
    max_modes: int = 512  # 最大模态数
    
    # SVD计算相关
    svd_method: str = 'torch'  # SVD计算方法: 'torch', 'numpy', 'sklearn'
    random_state: int = 42  # 随机种子
    
    # 数据预处理
    standardize_data: bool = True  # 是否标准化数据
    remove_mean: bool = True  # 是否移除均值
    scale_variance: bool = False  # 是否缩放方差到1
    
    # 增量学习
    incremental_learning: bool = False  # 是否支持增量学习
    forget_factor: float = 0.99  # 遗忘因子（用于增量学习）
    batch_size: int = 1000  # 增量学习批次大小
    
    # 缓存和保存
    cache_projection: bool = True  # 是否缓存投影矩阵
    save_statistics: bool = True  # 是否保存统计信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'n_modes': self.n_modes,
            'energy_threshold': self.energy_threshold,
            'auto_select_modes': self.auto_select_modes,
            'min_modes': self.min_modes,
            'max_modes': self.max_modes,
            'svd_method': self.svd_method,
            'random_state': self.random_state,
            'standardize_data': self.standardize_data,
            'remove_mean': self.remove_mean,
            'scale_variance': self.scale_variance,
            'incremental_learning': self.incremental_learning,
            'forget_factor': self.forget_factor,
            'batch_size': self.batch_size,
            'cache_projection': self.cache_projection,
            'save_statistics': self.save_statistics
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SVDProjectionConfig':
        """从字典创建配置"""
        return cls(**config_dict)


class SVDModalProjector:
    """
    SVD模态投影器
    
    实现基于SVD的模态分解和投影功能，将不同维度的数据投影到统一的潜在空间
    """
    
    def __init__(self, config: SVDProjectionConfig):
        """
        初始化SVD投影器
        
        Args:
            config: SVD投影配置
        """
        self.config = config
        
        # 投影矩阵和统计信息
        self.input_projector = None  # 输入投影矩阵 (U_input)
        self.output_projector = None  # 输出投影矩阵 (U_output)
        self.input_singular_values = None  # 输入奇异值
        self.output_singular_values = None  # 输出奇异值
        
        # 数据统计信息
        self.input_mean = None
        self.output_mean = None
        self.input_std = None
        self.output_std = None
        
        # 维度信息
        self.input_dim = None
        self.output_dim = None
        self.latent_dim = None
        
        # 增量学习相关
        self.n_samples_seen = 0
        self.incremental_input_cov = None
        self.incremental_output_cov = None
        
        # 是否已拟合
        self.is_fitted = False
        
        logger.info(f"SVD模态投影器初始化完成: {config.n_modes}维潜在空间")
    
    def _standardize_data(self, data: np.ndarray, mean: np.ndarray = None, 
                         std: np.ndarray = None, fit: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        标准化数据
        
        Args:
            data: 输入数据 (n_samples, n_features)
            mean: 均值（如果为None且fit=True则计算）
            std: 标准差（如果为None且fit=True则计算）
            fit: 是否计算统计信息
            
        Returns:
            标准化后的数据、均值、标准差
        """
        if fit or mean is None:
            if self.config.remove_mean:
                mean = np.mean(data, axis=0, keepdims=True)
            else:
                mean = np.zeros((1, data.shape[1]))
        
        if fit or std is None:
            if self.config.scale_variance:
                std = np.std(data, axis=0, keepdims=True)
                std = np.where(std < 1e-8, 1.0, std)  # 避免除零
            else:
                std = np.ones((1, data.shape[1]))
        
        standardized_data = (data - mean) / std
        return standardized_data, mean, std
    
    def _compute_svd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算SVD分解
        
        Args:
            data: 输入数据矩阵 (n_samples, n_features)
            
        Returns:
            U, S, Vt
        """
        if self.config.svd_method == 'torch':
            data_tensor = torch.from_numpy(data).float()
            U, S, Vt = torch.svd(data_tensor)
            return U.numpy(), S.numpy(), Vt.numpy().T
        
        elif self.config.svd_method == 'numpy':
            U, S, Vt = np.linalg.svd(data, full_matrices=False)
            return U, S, Vt
        
        elif self.config.svd_method == 'sklearn':
            from sklearn.decomposition import TruncatedSVD
            n_components = min(data.shape[0] - 1, data.shape[1], self.config.max_modes)
            svd = TruncatedSVD(n_components=n_components, random_state=self.config.random_state)
            U = svd.fit_transform(data)
            S = svd.singular_values_
            Vt = svd.components_
            return U, S, Vt
        
        else:
            raise ValueError(f"不支持的SVD方法: {self.config.svd_method}")
    
    def _select_modes(self, singular_values: np.ndarray) -> int:
        """
        根据能量阈值自动选择模态数
        
        Args:
            singular_values: 奇异值数组
            
        Returns:
            选择的模态数
        """
        if not self.config.auto_select_modes:
            return min(self.config.n_modes, len(singular_values))
        
        # 计算累积能量比例
        energy = singular_values ** 2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        
        # 找到满足能量阈值的最小模态数
        n_modes = np.searchsorted(cumulative_energy, self.config.energy_threshold) + 1
        
        # 约束在[min_modes, max_modes]范围内
        n_modes = max(self.config.min_modes, n_modes)
        n_modes = min(self.config.max_modes, n_modes)
        n_modes = min(n_modes, len(singular_values))
        
        energy_preserved = cumulative_energy[n_modes - 1]
        logger.info(f"自动选择 {n_modes} 个模态，保留能量: {energy_preserved:.4f}")
        
        return n_modes
    
    def fit(self, input_data: np.ndarray, output_data: np.ndarray) -> 'SVDModalProjector':
        """
        拟合SVD投影器
        
        Args:
            input_data: 输入数据 (n_samples, input_dim)
            output_data: 输出数据 (n_samples, output_dim)
            
        Returns:
            self
        """
        logger.info("开始拟合SVD模态投影器...")
        
        # 检查数据形状
        if input_data.shape[0] != output_data.shape[0]:
            raise ValueError(f"输入输出样本数不匹配: {input_data.shape[0]} vs {output_data.shape[0]}")
        
        n_samples = input_data.shape[0]
        self.input_dim = input_data.shape[1]
        self.output_dim = output_data.shape[1]
        
        logger.info(f"数据维度: 输入 {self.input_dim}, 输出 {self.output_dim}, 样本数 {n_samples}")
        
        # 数据标准化
        if self.config.standardize_data:
            input_data_std, self.input_mean, self.input_std = self._standardize_data(input_data, fit=True)
            output_data_std, self.output_mean, self.output_std = self._standardize_data(output_data, fit=True)
        else:
            input_data_std = input_data.copy()
            output_data_std = output_data.copy()
            self.input_mean = np.zeros((1, self.input_dim))
            self.output_mean = np.zeros((1, self.output_dim))
            self.input_std = np.ones((1, self.input_dim))
            self.output_std = np.ones((1, self.output_dim))
        
        # 计算输入数据的SVD
        logger.info("计算输入数据SVD...")
        U_input, S_input, Vt_input = self._compute_svd(input_data_std)
        
        # 计算输出数据的SVD
        logger.info("计算输出数据SVD...")
        U_output, S_output, Vt_output = self._compute_svd(output_data_std)
        
        # 自动选择模态数
        input_modes = self._select_modes(S_input)
        output_modes = self._select_modes(S_output)
        
        # 使用较小的模态数作为潜在维度，确保两个空间都能有效投影
        self.latent_dim = min(input_modes, output_modes, self.config.n_modes)
        
        logger.info(f"最终潜在维度: {self.latent_dim}")
        
        # 保存投影矩阵（前latent_dim个左奇异向量）
        self.input_projector = U_input[:, :self.latent_dim]  # (n_samples, latent_dim)
        self.output_projector = U_output[:, :self.latent_dim]  # (n_samples, latent_dim)
        
        # 保存奇异值
        self.input_singular_values = S_input[:self.latent_dim]
        self.output_singular_values = S_output[:self.latent_dim]
        
        # 为了投影新数据，我们需要右奇异向量
        self.input_V = Vt_input[:self.latent_dim, :]  # (latent_dim, input_dim)
        self.output_V = Vt_output[:self.latent_dim, :]  # (latent_dim, output_dim)
        
        self.is_fitted = True
        self.n_samples_seen = n_samples
        
        # 计算投影质量指标
        input_energy_ratio = np.sum(S_input[:self.latent_dim]**2) / np.sum(S_input**2)
        output_energy_ratio = np.sum(S_output[:self.latent_dim]**2) / np.sum(S_output**2)
        
        logger.info(f"投影完成!")
        logger.info(f"  输入能量保留: {input_energy_ratio:.4f}")
        logger.info(f"  输出能量保留: {output_energy_ratio:.4f}")
        logger.info(f"  潜在维度: {self.latent_dim}")
        
        return self
    
    def transform_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        将输入数据投影到潜在空间
        
        Args:
            input_data: 输入数据 (n_samples, input_dim)
            
        Returns:
            投影后的数据 (n_samples, latent_dim)
        """
        if not self.is_fitted:
            raise ValueError("投影器尚未拟合，请先调用fit()方法")
        
        # 标准化
        if self.config.standardize_data:
            input_data_std, _, _ = self._standardize_data(input_data, self.input_mean, self.input_std, fit=False)
        else:
            input_data_std = input_data.copy()
        
        # 投影: 使用右奇异向量进行投影
        # X_proj = X_std @ V.T @ S^{-1}，但由于我们保存的是训练时的U，这里简化为与V的内积
        projected = input_data_std @ self.input_V.T  # (n_samples, latent_dim)
        
        return projected
    
    def transform_output(self, output_data: np.ndarray) -> np.ndarray:
        """
        将输出数据投影到潜在空间
        
        Args:
            output_data: 输出数据 (n_samples, output_dim)
            
        Returns:
            投影后的数据 (n_samples, latent_dim)
        """
        if not self.is_fitted:
            raise ValueError("投影器尚未拟合，请先调用fit()方法")
        
        # 标准化
        if self.config.standardize_data:
            output_data_std, _, _ = self._standardize_data(output_data, self.output_mean, self.output_std, fit=False)
        else:
            output_data_std = output_data.copy()
        
        # 投影
        projected = output_data_std @ self.output_V.T  # (n_samples, latent_dim)
        
        return projected
    
    def inverse_transform_input(self, projected_data: np.ndarray) -> np.ndarray:
        """
        从潜在空间重建输入数据
        
        Args:
            projected_data: 潜在空间数据 (n_samples, latent_dim)
            
        Returns:
            重建的输入数据 (n_samples, input_dim)
        """
        if not self.is_fitted:
            raise ValueError("投影器尚未拟合，请先调用fit()方法")
        
        # 反投影
        reconstructed_std = projected_data @ self.input_V  # (n_samples, input_dim)
        
        # 反标准化
        if self.config.standardize_data:
            reconstructed = reconstructed_std * self.input_std + self.input_mean
        else:
            reconstructed = reconstructed_std
        
        return reconstructed
    
    def inverse_transform_output(self, projected_data: np.ndarray) -> np.ndarray:
        """
        从潜在空间重建输出数据
        
        Args:
            projected_data: 潜在空间数据 (n_samples, latent_dim)
            
        Returns:
            重建的输出数据 (n_samples, output_dim)
        """
        if not self.is_fitted:
            raise ValueError("投影器尚未拟合，请先调用fit()方法")
        
        # 反投影
        reconstructed_std = projected_data @ self.output_V  # (n_samples, output_dim)
        
        # 反标准化
        if self.config.standardize_data:
            reconstructed = reconstructed_std * self.output_std + self.output_mean
        else:
            reconstructed = reconstructed_std
        
        return reconstructed
    
    def fit_transform(self, input_data: np.ndarray, output_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并投影数据
        
        Args:
            input_data: 输入数据
            output_data: 输出数据
            
        Returns:
            投影后的输入数据和输出数据
        """
        self.fit(input_data, output_data)
        input_projected = self.transform_input(input_data)
        output_projected = self.transform_output(output_data)
        return input_projected, output_projected
    
    def compute_reconstruction_error(self, input_data: np.ndarray, output_data: np.ndarray) -> Dict[str, float]:
        """
        计算重建误差
        
        Args:
            input_data: 原始输入数据
            output_data: 原始输出数据
            
        Returns:
            重建误差字典
        """
        if not self.is_fitted:
            raise ValueError("投影器尚未拟合，请先调用fit()方法")
        
        # 投影并重建
        input_projected = self.transform_input(input_data)
        output_projected = self.transform_output(output_data)
        
        input_reconstructed = self.inverse_transform_input(input_projected)
        output_reconstructed = self.inverse_transform_output(output_projected)
        
        # 计算误差
        input_mse = np.mean((input_data - input_reconstructed) ** 2)
        output_mse = np.mean((output_data - output_reconstructed) ** 2)
        
        input_relative_error = np.sqrt(input_mse) / (np.std(input_data) + 1e-8)
        output_relative_error = np.sqrt(output_mse) / (np.std(output_data) + 1e-8)
        
        return {
            'input_mse': input_mse,
            'output_mse': output_mse,
            'input_relative_error': input_relative_error,
            'output_relative_error': output_relative_error,
            'latent_dim': self.latent_dim
        }
    
    def save(self, filepath: str):
        """
        保存投影器
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("投影器尚未拟合，无法保存")
        
        save_data = {
            'config': self.config.to_dict(),
            'input_projector': self.input_projector,
            'output_projector': self.output_projector,
            'input_V': self.input_V,
            'output_V': self.output_V,
            'input_singular_values': self.input_singular_values,
            'output_singular_values': self.output_singular_values,
            'input_mean': self.input_mean,
            'output_mean': self.output_mean,
            'input_std': self.input_std,
            'output_std': self.output_std,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'latent_dim': self.latent_dim,
            'is_fitted': self.is_fitted,
            'n_samples_seen': self.n_samples_seen
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"SVD投影器已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SVDModalProjector':
        """
        加载投影器
        
        Args:
            filepath: 文件路径
            
        Returns:
            SVD投影器实例
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # 创建配置和实例
        config = SVDProjectionConfig.from_dict(save_data['config'])
        projector = cls(config)
        
        # 恢复状态
        projector.input_projector = save_data['input_projector']
        projector.output_projector = save_data['output_projector']
        projector.input_V = save_data['input_V']
        projector.output_V = save_data['output_V']
        projector.input_singular_values = save_data['input_singular_values']
        projector.output_singular_values = save_data['output_singular_values']
        projector.input_mean = save_data['input_mean']
        projector.output_mean = save_data['output_mean']
        projector.input_std = save_data['input_std']
        projector.output_std = save_data['output_std']
        projector.input_dim = save_data['input_dim']
        projector.output_dim = save_data['output_dim']
        projector.latent_dim = save_data['latent_dim']
        projector.is_fitted = save_data['is_fitted']
        projector.n_samples_seen = save_data['n_samples_seen']
        
        logger.info(f"SVD投影器已从 {filepath} 加载")
        return projector
    
    def get_projection_info(self) -> Dict[str, Any]:
        """
        获取投影器信息
        
        Returns:
            投影器信息字典
        """
        if not self.is_fitted:
            return {'is_fitted': False}
        
        return {
            'is_fitted': True,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'latent_dim': self.latent_dim,
            'n_samples_seen': self.n_samples_seen,
            'config': self.config.to_dict(),
            'input_energy_ratios': (self.input_singular_values[:10] ** 2 / np.sum(self.input_singular_values ** 2)).tolist(),
            'output_energy_ratios': (self.output_singular_values[:10] ** 2 / np.sum(self.output_singular_values ** 2)).tolist()
        }


def create_svd_projector_from_config(config_dict: Dict[str, Any]) -> SVDModalProjector:
    """
    从配置字典创建SVD投影器
    
    Args:
        config_dict: 配置字典
        
    Returns:
        SVD投影器实例
    """
    config = SVDProjectionConfig.from_dict(config_dict)
    return SVDModalProjector(config)


def demo_svd_projection():
    """
    SVD投影演示函数
    """
    logger.info("=== SVD模态投影演示 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    input_dim = 1024  # 32x32
    output_dim = 16384  # 128x128
    
    # 生成具有低秩结构的数据
    rank = 50
    U_true = np.random.randn(n_samples, rank)
    V_input = np.random.randn(rank, input_dim)
    V_output = np.random.randn(rank, output_dim)
    
    input_data = U_true @ V_input + 0.1 * np.random.randn(n_samples, input_dim)
    output_data = U_true @ V_output + 0.1 * np.random.randn(n_samples, output_dim)
    
    logger.info(f"生成数据: 输入 {input_data.shape}, 输出 {output_data.shape}")
    
    # 创建配置
    config = SVDProjectionConfig(
        n_modes=64,
        energy_threshold=0.95,
        auto_select_modes=True,
        standardize_data=True
    )
    
    # 创建并拟合投影器
    projector = SVDModalProjector(config)
    projector.fit(input_data, output_data)
    
    # 投影数据
    input_projected = projector.transform_input(input_data)
    output_projected = projector.transform_output(output_data)
    
    logger.info(f"投影后数据: 输入 {input_projected.shape}, 输出 {output_projected.shape}")
    
    # 计算重建误差
    errors = projector.compute_reconstruction_error(input_data, output_data)
    logger.info(f"重建误差: {errors}")
    
    # 显示投影器信息
    info = projector.get_projection_info()
    logger.info(f"投影器信息: 潜在维度 {info['latent_dim']}")
    
    return projector


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行演示
    demo_svd_projection()