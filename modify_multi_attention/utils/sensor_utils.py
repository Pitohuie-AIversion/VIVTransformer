#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器数据处理工具模块

功能:
1. 传感器数据格式转换
2. 空间插值和时间插值
3. 数据质量检查和清洗
4. 多模态数据融合
5. 传感器网络优化

作者: AI Assistant
日期: 2025
"""

import numpy as np
import torch
import scipy.interpolate
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SensorDataProcessor:
    """传感器数据处理器"""
    
    def __init__(self, target_resolution: Tuple[int, int] = (64, 64)):
        self.target_resolution = target_resolution
        self.H, self.W = target_resolution
    
    def spatial_interpolation(self, 
                            sensor_positions: np.ndarray,
                            sensor_readings: np.ndarray,
                            method: str = 'rbf') -> np.ndarray:
        """
        空间插值：从稀疏传感器读数重建稠密场
        
        Args:
            sensor_positions: [N, 2] 传感器位置 (归一化坐标 0-1)
            sensor_readings: [N] 或 [T, N] 传感器读数
            method: 插值方法 ('rbf', 'idw', 'kriging')
        
        Returns:
            interpolated_field: [H, W] 或 [T, H, W] 插值场
        """
        # 创建目标网格
        x = np.linspace(0, 1, self.W)
        y = np.linspace(0, 1, self.H)
        X, Y = np.meshgrid(x, y)
        target_points = np.column_stack([X.ravel(), Y.ravel()])
        
        if sensor_readings.ndim == 1:
            # 单时间步
            if method == 'rbf':
                interpolated = self._rbf_interpolation(
                    sensor_positions, sensor_readings, target_points
                )
            elif method == 'idw':
                interpolated = self._idw_interpolation(
                    sensor_positions, sensor_readings, target_points
                )
            else:
                raise ValueError(f"不支持的插值方法: {method}")
            
            return interpolated.reshape(self.H, self.W)
        
        else:
            # 多时间步
            T = sensor_readings.shape[0]
            result = np.zeros((T, self.H, self.W))
            
            for t in range(T):
                if method == 'rbf':
                    interpolated = self._rbf_interpolation(
                        sensor_positions, sensor_readings[t], target_points
                    )
                elif method == 'idw':
                    interpolated = self._idw_interpolation(
                        sensor_positions, sensor_readings[t], target_points
                    )
                else:
                    raise ValueError(f"不支持的插值方法: {method}")
                
                result[t] = interpolated.reshape(self.H, self.W)
            
            return result
    
    def _rbf_interpolation(self, 
                          sensor_positions: np.ndarray,
                          sensor_readings: np.ndarray,
                          target_points: np.ndarray) -> np.ndarray:
        """径向基函数插值"""
        try:
            rbf = scipy.interpolate.Rbf(
                sensor_positions[:, 0], sensor_positions[:, 1], sensor_readings,
                function='multiquadric', smooth=0.1
            )
            return rbf(target_points[:, 0], target_points[:, 1])
        except Exception as e:
            logger.warning(f"RBF插值失败，使用IDW替代: {e}")
            return self._idw_interpolation(sensor_positions, sensor_readings, target_points)
    
    def _idw_interpolation(self, 
                          sensor_positions: np.ndarray,
                          sensor_readings: np.ndarray,
                          target_points: np.ndarray,
                          power: float = 2.0) -> np.ndarray:
        """反距离权重插值"""
        # 计算距离
        distances = cdist(target_points, sensor_positions)
        
        # 避免除零
        distances = np.maximum(distances, 1e-10)
        
        # 计算权重
        weights = 1.0 / (distances ** power)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights_normalized = weights / weights_sum
        
        # 插值
        interpolated = np.sum(weights_normalized * sensor_readings[np.newaxis, :], axis=1)
        
        return interpolated
    
    def temporal_interpolation(self, 
                             time_series: np.ndarray,
                             time_points: np.ndarray,
                             target_times: np.ndarray,
                             method: str = 'cubic') -> np.ndarray:
        """
        时间插值：处理不规则时间采样
        
        Args:
            time_series: [T, ...] 时间序列数据
            time_points: [T] 时间点
            target_times: [T_new] 目标时间点
            method: 插值方法 ('linear', 'cubic', 'spline')
        
        Returns:
            interpolated_series: [T_new, ...] 插值后的时间序列
        """
        original_shape = time_series.shape[1:]
        time_series_flat = time_series.reshape(len(time_points), -1)
        
        interpolated_flat = np.zeros((len(target_times), time_series_flat.shape[1]))
        
        for i in range(time_series_flat.shape[1]):
            if method == 'linear':
                interpolated_flat[:, i] = np.interp(
                    target_times, time_points, time_series_flat[:, i]
                )
            elif method in ['cubic', 'spline']:
                try:
                    f = scipy.interpolate.interp1d(
                        time_points, time_series_flat[:, i], 
                        kind='cubic', bounds_error=False, fill_value='extrapolate'
                    )
                    interpolated_flat[:, i] = f(target_times)
                except Exception:
                    # 回退到线性插值
                    interpolated_flat[:, i] = np.interp(
                        target_times, time_points, time_series_flat[:, i]
                    )
        
        return interpolated_flat.reshape((len(target_times),) + original_shape)
    
    def data_quality_check(self, 
                          sensor_data: np.ndarray,
                          threshold_std: float = 3.0) -> Dict[str, Union[bool, np.ndarray]]:
        """
        数据质量检查
        
        Args:
            sensor_data: 传感器数据
            threshold_std: 异常值检测阈值（标准差倍数）
        
        Returns:
            quality_report: 质量报告字典
        """
        report = {}
        
        # 检查缺失值
        missing_mask = np.isnan(sensor_data) | np.isinf(sensor_data)
        report['has_missing'] = np.any(missing_mask)
        report['missing_ratio'] = np.mean(missing_mask)
        report['missing_mask'] = missing_mask
        
        # 检查异常值
        if not report['has_missing']:
            mean_val = np.mean(sensor_data)
            std_val = np.std(sensor_data)
            outlier_mask = np.abs(sensor_data - mean_val) > threshold_std * std_val
            report['has_outliers'] = np.any(outlier_mask)
            report['outlier_ratio'] = np.mean(outlier_mask)
            report['outlier_mask'] = outlier_mask
        else:
            valid_data = sensor_data[~missing_mask]
            if len(valid_data) > 0:
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                outlier_mask = np.abs(sensor_data - mean_val) > threshold_std * std_val
                outlier_mask = outlier_mask & ~missing_mask  # 排除缺失值
                report['has_outliers'] = np.any(outlier_mask)
                report['outlier_ratio'] = np.mean(outlier_mask)
                report['outlier_mask'] = outlier_mask
            else:
                report['has_outliers'] = False
                report['outlier_ratio'] = 0.0
                report['outlier_mask'] = np.zeros_like(sensor_data, dtype=bool)
        
        # 数据范围检查
        if not np.all(missing_mask):
            valid_data = sensor_data[~missing_mask]
            report['data_range'] = (np.min(valid_data), np.max(valid_data))
            report['data_std'] = np.std(valid_data)
        else:
            report['data_range'] = (np.nan, np.nan)
            report['data_std'] = np.nan
        
        # 整体质量评分
        quality_score = 1.0
        if report['has_missing']:
            quality_score -= 0.3 * report['missing_ratio']
        if report['has_outliers']:
            quality_score -= 0.2 * report['outlier_ratio']
        
        report['quality_score'] = max(0.0, quality_score)
        
        return report
    
    def clean_sensor_data(self, 
                         sensor_data: np.ndarray,
                         method: str = 'interpolate') -> np.ndarray:
        """
        传感器数据清洗
        
        Args:
            sensor_data: 原始传感器数据
            method: 清洗方法 ('interpolate', 'remove', 'median_filter')
        
        Returns:
            cleaned_data: 清洗后的数据
        """
        quality_report = self.data_quality_check(sensor_data)
        
        if not quality_report['has_missing'] and not quality_report['has_outliers']:
            return sensor_data.copy()
        
        cleaned_data = sensor_data.copy()
        
        if method == 'interpolate':
            # 插值填补缺失值和异常值
            if quality_report['has_missing']:
                missing_mask = quality_report['missing_mask']
                if sensor_data.ndim == 1:
                    # 1D数据
                    valid_indices = np.where(~missing_mask)[0]
                    if len(valid_indices) > 1:
                        cleaned_data[missing_mask] = np.interp(
                            np.where(missing_mask)[0],
                            valid_indices,
                            sensor_data[valid_indices]
                        )
                else:
                    # 多维数据，逐维处理
                    for i in range(sensor_data.shape[-1]):
                        data_slice = sensor_data[..., i]
                        mask_slice = missing_mask[..., i]
                        if np.any(~mask_slice):
                            # 使用最近邻填充
                            from scipy.ndimage import distance_transform_edt
                            indices = distance_transform_edt(
                                mask_slice, return_distances=False, return_indices=True
                            )
                            cleaned_data[..., i] = data_slice[tuple(indices)]
            
            if quality_report['has_outliers']:
                outlier_mask = quality_report['outlier_mask']
                # 用中位数替换异常值
                median_val = np.nanmedian(cleaned_data)
                cleaned_data[outlier_mask] = median_val
        
        elif method == 'remove':
            # 移除异常值（设为NaN）
            if quality_report['has_outliers']:
                cleaned_data[quality_report['outlier_mask']] = np.nan
        
        elif method == 'median_filter':
            # 中值滤波
            from scipy.ndimage import median_filter
            if sensor_data.ndim == 1:
                cleaned_data = median_filter(sensor_data, size=3)
            else:
                cleaned_data = median_filter(sensor_data, size=3)
        
        return cleaned_data


class MultiModalFusion:
    """多模态传感器数据融合"""
    
    def __init__(self, fusion_method: str = 'weighted_average'):
        self.fusion_method = fusion_method
    
    def fuse_sensor_data(self, 
                        sensor_data_list: List[np.ndarray],
                        sensor_weights: Optional[List[float]] = None,
                        sensor_types: Optional[List[str]] = None) -> np.ndarray:
        """
        融合多模态传感器数据
        
        Args:
            sensor_data_list: 传感器数据列表
            sensor_weights: 传感器权重
            sensor_types: 传感器类型
        
        Returns:
            fused_data: 融合后的数据
        """
        if len(sensor_data_list) == 1:
            return sensor_data_list[0]
        
        # 标准化数据形状
        normalized_data = self._normalize_data_shapes(sensor_data_list)
        
        if self.fusion_method == 'weighted_average':
            return self._weighted_average_fusion(normalized_data, sensor_weights)
        elif self.fusion_method == 'concatenate':
            return self._concatenate_fusion(normalized_data)
        elif self.fusion_method == 'attention':
            return self._attention_fusion(normalized_data)
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")
    
    def _normalize_data_shapes(self, sensor_data_list: List[np.ndarray]) -> List[np.ndarray]:
        """标准化数据形状"""
        # 找到最大维度
        max_shape = tuple(max(data.shape[i] for data in sensor_data_list) 
                         for i in range(max(len(data.shape) for data in sensor_data_list)))
        
        normalized_data = []
        for data in sensor_data_list:
            # 填充到相同形状
            padded_data = np.zeros(max_shape)
            slices = tuple(slice(0, s) for s in data.shape)
            padded_data[slices] = data
            normalized_data.append(padded_data)
        
        return normalized_data
    
    def _weighted_average_fusion(self, 
                               sensor_data_list: List[np.ndarray],
                               weights: Optional[List[float]] = None) -> np.ndarray:
        """加权平均融合"""
        if weights is None:
            weights = [1.0 / len(sensor_data_list)] * len(sensor_data_list)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 归一化
        
        fused_data = np.zeros_like(sensor_data_list[0])
        for data, weight in zip(sensor_data_list, weights):
            fused_data += weight * data
        
        return fused_data
    
    def _concatenate_fusion(self, sensor_data_list: List[np.ndarray]) -> np.ndarray:
        """拼接融合"""
        return np.concatenate(sensor_data_list, axis=-1)
    
    def _attention_fusion(self, sensor_data_list: List[np.ndarray]) -> np.ndarray:
        """注意力融合"""
        # 简化的注意力机制
        stacked_data = np.stack(sensor_data_list, axis=0)  # [N_sensors, ...]
        
        # 计算注意力权重（基于数据方差）
        variances = np.var(stacked_data, axis=tuple(range(1, stacked_data.ndim)))
        attention_weights = variances / np.sum(variances)
        
        # 应用注意力权重
        fused_data = np.sum(stacked_data * attention_weights.reshape(-1, *([1] * (stacked_data.ndim - 1))), axis=0)
        
        return fused_data


class SensorNetworkOptimizer:
    """传感器网络优化器"""
    
    def __init__(self, domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1))):
        self.domain_bounds = domain_bounds
    
    def optimize_sensor_placement(self, 
                                 num_sensors: int,
                                 method: str = 'uniform',
                                 existing_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        优化传感器布置
        
        Args:
            num_sensors: 传感器数量
            method: 优化方法 ('uniform', 'random', 'greedy', 'genetic')
            existing_positions: 现有传感器位置
        
        Returns:
            optimal_positions: [N, 2] 优化后的传感器位置
        """
        if method == 'uniform':
            return self._uniform_placement(num_sensors)
        elif method == 'random':
            return self._random_placement(num_sensors)
        elif method == 'greedy':
            return self._greedy_placement(num_sensors, existing_positions)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def _uniform_placement(self, num_sensors: int) -> np.ndarray:
        """均匀布置"""
        # 尽可能均匀分布
        grid_size = int(np.ceil(np.sqrt(num_sensors)))
        
        x_coords = np.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], grid_size)
        y_coords = np.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], grid_size)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        positions = np.column_stack([X.ravel(), Y.ravel()])
        
        # 选择前num_sensors个位置
        return positions[:num_sensors]
    
    def _random_placement(self, num_sensors: int) -> np.ndarray:
        """随机布置"""
        x_range = self.domain_bounds[0][1] - self.domain_bounds[0][0]
        y_range = self.domain_bounds[1][1] - self.domain_bounds[1][0]
        
        positions = np.random.rand(num_sensors, 2)
        positions[:, 0] = positions[:, 0] * x_range + self.domain_bounds[0][0]
        positions[:, 1] = positions[:, 1] * y_range + self.domain_bounds[1][0]
        
        return positions
    
    def _greedy_placement(self, 
                         num_sensors: int,
                         existing_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """贪心布置（最大化覆盖距离）"""
        if existing_positions is None:
            # 第一个传感器放在中心
            positions = [np.array([
                (self.domain_bounds[0][0] + self.domain_bounds[0][1]) / 2,
                (self.domain_bounds[1][0] + self.domain_bounds[1][1]) / 2
            ])]
            start_idx = 1
        else:
            positions = list(existing_positions)
            start_idx = 0
        
        # 候选位置网格
        resolution = 50
        x_coords = np.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], resolution)
        y_coords = np.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        candidates = np.column_stack([X.ravel(), Y.ravel()])
        
        for _ in range(start_idx, num_sensors):
            if len(positions) == 0:
                # 第一个传感器
                best_pos = candidates[len(candidates) // 2]
            else:
                # 选择距离现有传感器最远的位置
                existing_array = np.array(positions)
                distances = cdist(candidates, existing_array)
                min_distances = np.min(distances, axis=1)
                best_idx = np.argmax(min_distances)
                best_pos = candidates[best_idx]
            
            positions.append(best_pos)
        
        return np.array(positions)
    
    def evaluate_coverage(self, 
                         sensor_positions: np.ndarray,
                         coverage_radius: float = 0.1) -> float:
        """评估传感器网络覆盖率"""
        # 创建评估网格
        resolution = 100
        x_coords = np.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], resolution)
        y_coords = np.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        
        # 计算每个网格点到最近传感器的距离
        distances = cdist(grid_points, sensor_positions)
        min_distances = np.min(distances, axis=1)
        
        # 计算覆盖率
        covered_points = np.sum(min_distances <= coverage_radius)
        coverage_ratio = covered_points / len(grid_points)
        
        return coverage_ratio


def convert_sensor_format(data: Union[np.ndarray, torch.Tensor], 
                         source_format: str, 
                         target_format: str) -> Union[np.ndarray, torch.Tensor]:
    """
    传感器数据格式转换
    
    Args:
        data: 输入数据
        source_format: 源格式 ('numpy', 'torch', 'list')
        target_format: 目标格式 ('numpy', 'torch', 'list')
    
    Returns:
        converted_data: 转换后的数据
    """
    if source_format == target_format:
        return data
    
    # 转换为numpy作为中间格式
    if source_format == 'torch':
        numpy_data = data.detach().cpu().numpy()
    elif source_format == 'list':
        numpy_data = np.array(data)
    else:
        numpy_data = data
    
    # 转换为目标格式
    if target_format == 'torch':
        return torch.from_numpy(numpy_data).float()
    elif target_format == 'list':
        return numpy_data.tolist()
    else:
        return numpy_data


def validate_sensor_config(config: Dict) -> bool:
    """验证传感器配置"""
    required_fields = ['type', 'num_sensors']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"传感器配置缺少必需字段: {field}")
            return False
    
    if config['type'] not in ['time_series', 'spatial_sparse', 'spatial_dense']:
        logger.error(f"不支持的传感器类型: {config['type']}")
        return False
    
    if config['type'] == 'time_series' and 'temporal_length' not in config:
        logger.error("时间序列传感器配置缺少 temporal_length 字段")
        return False
    
    return True