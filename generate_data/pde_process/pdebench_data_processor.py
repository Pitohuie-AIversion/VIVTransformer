#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDEBench数据集截取和处理脚本

功能描述:
    1. 从PDEBench HDF5文件中读取PDE数据
    2. 截取指定时间范围和样本数量的数据
    3. 对数据进行归一化处理
    4. 将数据组织成适合机器学习的格式
    5. 支持多种PDE类型的数据处理

数据格式:
    - 输入: HDF5文件，包含多个seed的PDE解数据
    - 输出: PyTorch张量，形状为[samples, time_steps, spatial_dims...]
    
使用方法:
    1. 直接运行脚本: python pdebench_data_processor.py
    2. 导入使用: from pdebench_data_processor import PDEBenchProcessor
    
作者: AI Assistant
日期: 2025
版本: 1.0
"""

import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import argparse
from tqdm import tqdm

# 设置matplotlib支持中文显示
# 统一使用全局 sitecustomize.py 的中文字体和负号设置
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDEBenchConfig:
    """PDEBench数据处理配置类"""
    
    def __init__(self):
        # 数据路径配置
        self.PDEBENCH_ROOT = r"x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download"
        
        # 数据截取配置
        self.MAX_SAMPLES = 1000  # 最大样本数量
        self.START_TIME_IDX = 0  # 开始时间索引
        self.END_TIME_IDX = None  # 结束时间索引，None表示到最后
        self.TIME_STEP_INTERVAL = 1  # 时间步间隔
        
        # 空间截取配置
        self.SPATIAL_CROP = None  # 空间裁剪，格式: ((x_start, x_end), (y_start, y_end))
        self.SPATIAL_DOWNSAMPLE = 1  # 空间下采样因子
        
        # 输出配置
        self.OUTPUT_FILE = 'pdebench_processed_data.pt'
        self.NORMALIZE_DATA = True
        self.FLATTEN_SPATIAL = True  # 是否将空间维度展平
        
        # 支持的PDE类型
        self.SUPPORTED_PDES = {
            'darcy': '2D_DarcyFlow_beta0.1_Train.hdf5',
            'diff_sorp': '1D_diff-sorp_NA_NA.h5',
            'reacdiff_2d': '2D_diff-react_NA_NA.h5',
            'burgers': '1D_Burgers_Sols_Nu0.001.hdf5',
            'advection': '1D_Advection_Sols_beta0.1.hdf5'
        }


class PDEBenchProcessor:
    """PDEBench数据处理器"""
    
    def __init__(self, config: PDEBenchConfig):
        self.config = config
        self.processed_data = {}
        self.normalization_params = {}
        
    def load_hdf5_data(self, file_path: str, pde_type: str = 'auto') -> Optional[Dict]:
        """
        加载HDF5文件数据
        
        Args:
            file_path: HDF5文件路径
            pde_type: PDE类型，用于确定数据结构
            
        Returns:
            包含数据的字典，如果失败返回None
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return None
                
            logger.info(f"正在加载文件: {file_path}")
            
            with h5py.File(file_path, 'r') as h5_file:
                # 检查文件结构
                keys = list(h5_file.keys())
                logger.info(f"HDF5文件包含的数据集: {keys}")
                
                # 检查是否是PDEBench格式（包含tensor数据集）
                if 'tensor' in keys:
                    return self._load_pdebench_format(h5_file, file_path, pde_type)
                else:
                    # 尝试原来的seed格式
                    return self._load_seed_format(h5_file, file_path, pde_type)
                    
        except Exception as e:
            logger.error(f"加载HDF5文件时出错: {str(e)}")
            return None
    
    def _load_pdebench_format(self, h5_file, file_path: str, pde_type: str) -> Optional[Dict]:
        """
        加载PDEBench格式的HDF5数据
        
        Args:
            h5_file: 打开的HDF5文件对象
            file_path: 文件路径
            pde_type: PDE类型
            
        Returns:
            包含数据的字典
        """
        try:
            # 获取主要数据
            tensor_data = h5_file['tensor']
            logger.info(f"Tensor数据形状: {tensor_data.shape}")
            
            # 限制样本数量
            total_samples = tensor_data.shape[0]
            max_samples = min(self.config.MAX_SAMPLES, total_samples) if self.config.MAX_SAMPLES else total_samples
            
            logger.info(f"总样本数: {total_samples}, 使用样本数: {max_samples}")
            
            # 读取数据
            data = np.array(tensor_data[:max_samples], dtype=np.float32)
            
            # 如果数据只有一个时间步，我们需要创建时间序列
            if len(data.shape) == 4 and data.shape[1] == 1:
                # 形状: [samples, 1, height, width] -> [samples, height, width]
                data = data.squeeze(1)
                logger.info(f"移除单一时间维度，新形状: {data.shape}")
                
                # 为了创建时间序列，我们可以使用不同的样本作为时间步
                # 或者使用空间扰动来模拟时间演化
                data = self._create_temporal_sequence(data)
            
            # 应用时间截取
            data = self._apply_temporal_crop(data)
            
            # 应用空间处理
            data = self._apply_spatial_processing(data)
            
            if data is None:
                logger.error("数据处理后为空")
                return None
            
            # 创建样本信息
            sample_info = []
            for i in range(data.shape[0]):
                sample_info.append({
                    'sample_id': i,
                    'original_shape': tensor_data.shape,
                    'pde_type': pde_type
                })
            
            logger.info(f"成功加载PDEBench数据，最终形状: {data.shape}")
            
            return {
                'data': data,
                'sample_info': sample_info,
                'file_path': file_path,
                'pde_type': pde_type
            }
            
        except Exception as e:
            logger.error(f"加载PDEBench格式数据时出错: {str(e)}")
            return None
    
    def _load_seed_format(self, h5_file, file_path: str, pde_type: str) -> Optional[Dict]:
        """
        加载原始seed格式的HDF5数据
        
        Args:
            h5_file: 打开的HDF5文件对象
            file_path: 文件路径
            pde_type: PDE类型
            
        Returns:
            包含数据的字典
        """
        try:
            # 获取所有seed
            seeds = list(h5_file.keys())
            logger.info(f"找到 {len(seeds)} 个样本")
            
            # 限制样本数量
            if self.config.MAX_SAMPLES and len(seeds) > self.config.MAX_SAMPLES:
                seeds = seeds[:self.config.MAX_SAMPLES]
                logger.info(f"限制样本数量为: {len(seeds)}")
            
            # 收集所有数据
            all_samples = []
            sample_info = []
            
            for seed in tqdm(seeds, desc="加载样本"):
                try:
                    # 读取数据
                    data = np.array(h5_file[f"{seed}/data"], dtype=np.float32)
                    
                    # 应用时间截取
                    data = self._apply_temporal_crop(data)
                    
                    # 应用空间处理
                    data = self._apply_spatial_processing(data)
                    
                    if data is not None:
                        all_samples.append(data)
                        sample_info.append({
                            'seed': seed,
                            'original_shape': data.shape,
                            'pde_type': pde_type
                        })
                        
                except Exception as e:
                    logger.warning(f"处理样本 {seed} 时出错: {str(e)}")
                    continue
            
            if not all_samples:
                logger.error("没有成功加载任何样本")
                return None
            
            # 转换为numpy数组
            try:
                data_array = np.stack(all_samples, axis=0)
                logger.info(f"成功加载数据，形状: {data_array.shape}")
                
                return {
                    'data': data_array,
                    'sample_info': sample_info,
                    'file_path': file_path,
                    'pde_type': pde_type
                }
                
            except Exception as e:
                logger.error(f"堆叠数据时出错: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"加载seed格式数据时出错: {str(e)}")
            return None
    
    def _create_temporal_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        从静态数据创建时间序列
        
        Args:
            data: 输入数据 [samples, height, width]
            
        Returns:
            时间序列数据 [samples, time_steps, height, width]
        """
        try:
            samples, height, width = data.shape
            
            # 方法1: 使用连续的样本作为时间序列
            time_steps = min(10, samples // 10)  # 每10个样本创建一个时间序列
            if time_steps < 2:
                time_steps = 2
            
            # 重新组织数据
            num_sequences = samples // time_steps
            if num_sequences == 0:
                # 如果样本太少，复制数据
                logger.warning(f"样本数量太少({samples})，复制数据创建时间序列")
                temporal_data = np.tile(data[:1], (time_steps, 1, 1, 1))
                temporal_data = temporal_data.transpose(1, 0, 2, 3)  # [1, time_steps, height, width]
            else:
                # 重新整理为时间序列
                selected_data = data[:num_sequences * time_steps]
                temporal_data = selected_data.reshape(num_sequences, time_steps, height, width)
            
            logger.info(f"创建时间序列: {data.shape} -> {temporal_data.shape}")
            return temporal_data
            
        except Exception as e:
            logger.error(f"创建时间序列时出错: {str(e)}")
            # 返回原始数据，添加时间维度
            return data[:, np.newaxis, :, :]
    
    def _apply_temporal_crop(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        应用时间维度的截取
        
        Args:
            data: 输入数据，时间维度通常是第一个维度
            
        Returns:
            截取后的数据
        """
        try:
            time_steps = data.shape[0]
            
            # 确定开始和结束索引
            start_idx = self.config.START_TIME_IDX
            end_idx = self.config.END_TIME_IDX if self.config.END_TIME_IDX is not None else time_steps
            
            # 边界检查
            start_idx = max(0, start_idx)
            end_idx = min(time_steps, end_idx)
            
            if start_idx >= end_idx:
                logger.warning(f"时间截取范围无效: [{start_idx}, {end_idx})")
                return None
            
            # 应用时间步间隔
            indices = range(start_idx, end_idx, self.config.TIME_STEP_INTERVAL)
            
            return data[indices]
            
        except Exception as e:
            logger.error(f"时间截取时出错: {str(e)}")
            return None
    
    def _apply_spatial_processing(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        应用空间维度的处理（裁剪、下采样等）
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        try:
            # 空间裁剪
            if self.config.SPATIAL_CROP and len(data.shape) >= 3:
                (x_start, x_end), (y_start, y_end) = self.config.SPATIAL_CROP
                if len(data.shape) == 3:  # [time, height, width]
                    data = data[:, x_start:x_end, y_start:y_end]
                elif len(data.shape) == 4:  # [time, height, width, channels]
                    data = data[:, x_start:x_end, y_start:y_end, :]
            
            # 空间下采样
            if self.config.SPATIAL_DOWNSAMPLE > 1 and len(data.shape) >= 3:
                step = self.config.SPATIAL_DOWNSAMPLE
                if len(data.shape) == 3:  # [time, height, width]
                    data = data[:, ::step, ::step]
                elif len(data.shape) == 4:  # [time, height, width, channels]
                    data = data[:, ::step, ::step, :]
            
            # 展平空间维度
            if self.config.FLATTEN_SPATIAL and len(data.shape) >= 3:
                time_steps = data.shape[0]
                spatial_dims = np.prod(data.shape[1:])
                data = data.reshape(time_steps, spatial_dims)
            
            return data
            
        except Exception as e:
            logger.error(f"空间处理时出错: {str(e)}")
            return None
    
    def normalize_data(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        对数据进行归一化
        
        Args:
            data: 输入数据
            
        Returns:
            归一化后的数据和归一化参数
        """
        try:
            if not self.config.NORMALIZE_DATA:
                return data, {}
            
            # 计算全局最小值和最大值
            min_val = np.min(data)
            max_val = np.max(data)
            
            # 避免除零
            if max_val == min_val:
                logger.warning("数据的最大值和最小值相等，跳过归一化")
                return data, {'min_val': min_val, 'max_val': max_val, 'normalized': False}
            
            # 归一化到[0, 1]
            normalized_data = (data - min_val) / (max_val - min_val)
            
            norm_params = {
                'min_val': float(min_val),
                'max_val': float(max_val),
                'normalized': True
            }
            
            logger.info(f"数据归一化完成，范围: [{min_val:.6f}, {max_val:.6f}] -> [0, 1]")
            
            return normalized_data, norm_params
            
        except Exception as e:
            logger.error(f"归一化时出错: {str(e)}")
            return data, {}
    
    def create_input_output_pairs(self, data: np.ndarray, sequence_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建输入-输出对，用于序列预测任务
        
        Args:
            data: 输入数据 [samples, time_steps, ...]
            sequence_length: 输入序列长度
            
        Returns:
            输入和输出数据对
        """
        try:
            samples, time_steps = data.shape[:2]
            
            if time_steps <= sequence_length:
                logger.warning(f"时间步数 {time_steps} 不足以创建长度为 {sequence_length} 的序列")
                return data[:, :-1], data[:, 1:]
            
            inputs = []
            outputs = []
            
            for i in range(time_steps - sequence_length):
                inputs.append(data[:, i:i+sequence_length])
                outputs.append(data[:, i+sequence_length])
            
            inputs = np.stack(inputs, axis=1)  # [samples, sequences, seq_len, ...]
            outputs = np.stack(outputs, axis=1)  # [samples, sequences, ...]
            
            # 重新整理维度
            inputs = inputs.reshape(-1, *inputs.shape[2:])  # [samples*sequences, seq_len, ...]
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # [samples*sequences, ...]
            
            logger.info(f"创建输入-输出对: 输入形状 {inputs.shape}, 输出形状 {outputs.shape}")
            
            return inputs, outputs
            
        except Exception as e:
            logger.error(f"创建输入-输出对时出错: {str(e)}")
            return data[:, :-1], data[:, 1:]
    
    def process_pde_dataset(self, pde_type: str, sequence_length: int = 1) -> bool:
        """
        处理指定类型的PDE数据集
        
        Args:
            pde_type: PDE类型
            sequence_length: 序列长度
            
        Returns:
            处理是否成功
        """
        try:
            if pde_type not in self.config.SUPPORTED_PDES:
                logger.error(f"不支持的PDE类型: {pde_type}")
                logger.info(f"支持的类型: {list(self.config.SUPPORTED_PDES.keys())}")
                return False
            
            # 构建文件路径
            filename = self.config.SUPPORTED_PDES[pde_type]
            file_path = os.path.join(self.config.PDEBENCH_ROOT, filename)
            
            # 加载数据
            data_dict = self.load_hdf5_data(file_path, pde_type)
            if data_dict is None:
                return False
            
            # 归一化数据
            normalized_data, norm_params = self.normalize_data(data_dict['data'])
            
            # 创建输入-输出对
            inputs, outputs = self.create_input_output_pairs(normalized_data, sequence_length)
            
            # 转换为PyTorch张量
            inputs_tensor = torch.from_numpy(inputs).float()
            outputs_tensor = torch.from_numpy(outputs).float()
            
            # 保存处理后的数据
            self.processed_data[pde_type] = {
                'inputs': inputs_tensor,
                'outputs': outputs_tensor,
                'sample_info': data_dict['sample_info'],
                'original_shape': data_dict['data'].shape,
                'processed_shape': {
                    'inputs': inputs_tensor.shape,
                    'outputs': outputs_tensor.shape
                }
            }
            
            self.normalization_params[pde_type] = norm_params
            
            logger.info(f"成功处理 {pde_type} 数据集")
            logger.info(f"  输入形状: {inputs_tensor.shape}")
            logger.info(f"  输出形状: {outputs_tensor.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理PDE数据集时出错: {str(e)}")
            return False
    
    def save_processed_data(self, output_path: str = None) -> bool:
        """
        保存处理后的数据
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            保存是否成功
        """
        try:
            if not self.processed_data:
                logger.error("没有处理后的数据可保存")
                return False
            
            if output_path is None:
                output_path = self.config.OUTPUT_FILE
            
            # 准备保存的数据
            save_data = {
                'processed_data': self.processed_data,
                'normalization_params': self.normalization_params,
                'config': {
                    'max_samples': self.config.MAX_SAMPLES,
                    'time_range': (self.config.START_TIME_IDX, self.config.END_TIME_IDX),
                    'time_step_interval': self.config.TIME_STEP_INTERVAL,
                    'spatial_crop': self.config.SPATIAL_CROP,
                    'spatial_downsample': self.config.SPATIAL_DOWNSAMPLE,
                    'flatten_spatial': self.config.FLATTEN_SPATIAL,
                    'normalize_data': self.config.NORMALIZE_DATA
                },
                'pde_types': list(self.processed_data.keys())
            }
            
            torch.save(save_data, output_path)
            logger.info(f"数据已保存到: {output_path}")
            
            # 打印数据统计
            self._print_data_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"保存数据时出错: {str(e)}")
            return False
    
    def _print_data_statistics(self):
        """打印数据统计信息"""
        logger.info("=== 数据统计信息 ===")
        for pde_type, data in self.processed_data.items():
            logger.info(f"PDE类型: {pde_type}")
            logger.info(f"  输入数据: {data['inputs'].shape}")
            logger.info(f"  输出数据: {data['outputs'].shape}")
            logger.info(f"  样本数量: {len(data['sample_info'])}")
            logger.info(f"  原始形状: {data['original_shape']}")
            
            if pde_type in self.normalization_params:
                norm_params = self.normalization_params[pde_type]
                if norm_params.get('normalized', False):
                    logger.info(f"  归一化范围: [{norm_params['min_val']:.6f}, {norm_params['max_val']:.6f}]")
            logger.info("-" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDEBench数据处理器')
    parser.add_argument('--pde_type', type=str, default='darcy',
                       help='PDE类型 (darcy, diff_sorp, reacdiff_2d, burgers, advection)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='最大样本数量')
    parser.add_argument('--sequence_length', type=int, default=1,
                       help='输入序列长度')
    parser.add_argument('--output_file', type=str, default='pdebench_processed_data.pt',
                       help='输出文件名')
    parser.add_argument('--start_time', type=int, default=0,
                       help='开始时间索引')
    parser.add_argument('--end_time', type=int, default=None,
                       help='结束时间索引')
    parser.add_argument('--time_interval', type=int, default=1,
                       help='时间步间隔')
    parser.add_argument('--spatial_downsample', type=int, default=1,
                       help='空间下采样因子')
    parser.add_argument('--no_normalize', action='store_true',
                       help='不进行数据归一化')
    parser.add_argument('--no_flatten', action='store_true',
                       help='不展平空间维度')
    
    args = parser.parse_args()
    
    # 创建配置
    config = PDEBenchConfig()
    config.MAX_SAMPLES = args.max_samples
    config.START_TIME_IDX = args.start_time
    config.END_TIME_IDX = args.end_time
    config.TIME_STEP_INTERVAL = args.time_interval
    config.SPATIAL_DOWNSAMPLE = args.spatial_downsample
    config.OUTPUT_FILE = args.output_file
    config.NORMALIZE_DATA = not args.no_normalize
    config.FLATTEN_SPATIAL = not args.no_flatten
    
    # 创建处理器
    processor = PDEBenchProcessor(config)
    
    logger.info(f"开始处理 {args.pde_type} 数据集...")
    
    # 处理数据
    success = processor.process_pde_dataset(args.pde_type, args.sequence_length)
    
    if success:
        # 保存数据
        processor.save_processed_data()
        logger.info("✓ 数据处理完成！")
    else:
        logger.error("✗ 数据处理失败")


if __name__ == "__main__":
    main()