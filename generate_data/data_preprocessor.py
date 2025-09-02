#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理器 - 独立的数据预处理脚本

功能:
1. 独立处理数据，避免与训练过程的CPU竞争
2. 预处理后的数据保存为HDF5文件，供训练脚本直接加载
3. 支持多种分辨率转换和归一化
4. 生成训练、验证、测试数据集

作者: AI Assistant
日期: 2025
"""

# 设置无头模式环境变量
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

import sys
import torch
import numpy as np
import h5py
import yaml
import argparse
import logging
import gc
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# 导入数据处理模块
try:
    from pde_process.resolution_downsampler.resolution_downsampler import (
        ResolutionDownsampler, DownsampledResolutionDataset
    )
    HAS_DOWNSAMPLER = True
    logger.info("[OK] 成功导入降分辨率模块")
except ImportError as e:
    HAS_DOWNSAMPLER = False
    logger.warning(f"[WARN] 降分辨率模块导入失败: {e}")
    logger.warning("将使用传统的裁剪方法")

class DataPreprocessor:
    """
    数据预处理器类
    负责将原始数据预处理为训练就绪的格式
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data']
        
        # 分辨率配置
        self.input_resolution = tuple(self.data_config['input_resolution'])
        self.output_resolution = tuple(self.data_config['output_resolution'])
        
        # 数据配置
        self.num_samples = self.data_config['num_samples']
        self.normalize_data = self.data_config.get('normalize_data', True)
        self.crop_mode = self.data_config.get('crop_mode', 'center')
        
        # 数据集分割比例
        self.train_ratio = self.data_config.get('train_ratio', 0.7)
        self.valid_ratio = self.data_config.get('valid_ratio', 0.2)
        self.test_ratio = self.data_config.get('test_ratio', 0.1)
        
        # 降采样配置
        self.downsampling_config = self.data_config.get('downsampling', {})
        self.use_downsampling = self.downsampling_config.get('enabled', False) and HAS_DOWNSAMPLER
        
        # 归一化参数
        self.global_min = None
        self.global_max = None
        
        logger.info(f"数据预处理器初始化完成:")
        logger.info(f"  输入分辨率: {self.input_resolution}")
        logger.info(f"  输出分辨率: {self.output_resolution}")
        logger.info(f"  样本数量: {self.num_samples}")
        logger.info(f"  数据归一化: {self.normalize_data}")
        logger.info(f"  使用降采样: {self.use_downsampling}")
    
    def load_original_data(self, data_path: str) -> np.ndarray:
        """
        加载原始数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            原始数据数组
        """
        logger.info(f"加载原始数据: {data_path}")
        
        with h5py.File(data_path, 'r') as f:
            if 'tensor' in f:
                tensor_data = f['tensor']
                logger.info(f"原始数据形状: {tensor_data.shape}")
                
                # 读取指定数量的样本
                actual_samples = min(self.num_samples, tensor_data.shape[0])
                data = np.array(tensor_data[:actual_samples], dtype=np.float32)
                
                # 如果数据是4D，移除通道维度
                if len(data.shape) == 4 and data.shape[1] == 1:
                    data = data.squeeze(1)
                    logger.info(f"移除通道维度后形状: {data.shape}")
                
                return data
            else:
                raise ValueError("数据文件中未找到'tensor'键")
    
    def compute_normalization_params(self, data: np.ndarray):
        """
        计算归一化参数
        
        Args:
            data: 原始数据
        """
        logger.info("计算全局归一化参数...")
        
        self.global_min = np.min(data)
        self.global_max = np.max(data)
        
        logger.info(f"全局数据范围: [{self.global_min:.6f}, {self.global_max:.6f}]")
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        归一化数据到[0, 1]范围
        
        Args:
            data: 输入数据
            
        Returns:
            归一化后的数据
        """
        eps = 1e-8
        if abs(self.global_max - self.global_min) < eps:
            logger.warning(f"数据范围过小，可能存在常数数据: [{self.global_min:.6f}, {self.global_max:.6f}]")
            return np.zeros_like(data)
        return (data - self.global_min) / (self.global_max - self.global_min)
    
    def crop_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        裁剪数据到目标尺寸
        
        Args:
            data: 输入数据 (samples, height, width)
            target_size: 目标尺寸 (height, width)
            
        Returns:
            裁剪后的数据
        """
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        if target_h > orig_h or target_w > orig_w:
            raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 ({orig_h}, {orig_w})")
        
        if self.crop_mode == 'center':
            # 中心裁剪
            start_h = (orig_h - target_h) // 2
            start_w = (orig_w - target_w) // 2
        elif self.crop_mode == 'corner':
            # 左上角裁剪
            start_h = 0
            start_w = 0
        else:
            raise ValueError(f"不支持的裁剪模式: {self.crop_mode}")
        
        end_h = start_h + target_h
        end_w = start_w + target_w
        
        return data[:, start_h:end_h, start_w:end_w]
    
    def resize_data(self, data: np.ndarray, target_size: Tuple[int, int], method: str = 'bilinear') -> np.ndarray:
        """
        调整数据尺寸
        
        Args:
            data: 输入数据
            target_size: 目标尺寸
            method: 插值方法
            
        Returns:
            调整尺寸后的数据
        """
        from scipy.ndimage import zoom
        
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        zoom_h = target_h / orig_h
        zoom_w = target_w / orig_w
        
        # 根据插值方法选择order参数
        if method == 'nearest':
            order = 0
        elif method == 'bilinear':
            order = 1
        elif method == 'bicubic':
            order = 3
        else:
            logger.warning(f"未知的插值方法: {method}，使用双线性插值")
            order = 1
        
        resized_data = np.zeros((data.shape[0], target_h, target_w))
        for i in tqdm(range(data.shape[0]), desc=f"调整尺寸到{target_size}"):
            resized_data[i] = zoom(data[i], (zoom_h, zoom_w), order=order)
        
        return resized_data
    
    def process_data(self, data_path: str, output_dir: str):
        """
        处理数据并保存
        
        Args:
            data_path: 原始数据路径
            output_dir: 输出目录
        """
        start_time = time.time()
        logger.info("=== 开始数据预处理 ===")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载原始数据
        logger.info("步骤 1/6: 加载原始数据")
        original_data = self.load_original_data(data_path)
        logger.info(f"加载完成，数据形状: {original_data.shape}")
        
        # 2. 计算归一化参数
        if self.normalize_data:
            logger.info("步骤 2/6: 计算归一化参数")
            self.compute_normalization_params(original_data)
        else:
            logger.info("步骤 2/6: 跳过归一化")
        
        # 3. 生成输入数据（裁剪到输入分辨率）
        logger.info("步骤 3/6: 生成输入数据")
        input_data = self.crop_data(original_data, self.input_resolution)
        logger.info(f"输入数据形状: {input_data.shape}")
        
        # 4. 生成输出数据
        logger.info("步骤 4/6: 生成输出数据")
        if self.use_downsampling:
            logger.info("使用降采样方法生成输出数据")
            # 使用降采样模块
            output_data = self.resize_data(
                original_data, 
                self.output_resolution, 
                method=self.downsampling_config.get('method', 'bilinear')
            )
        else:
            logger.info("使用裁剪方法生成输出数据")
            # 使用裁剪方法
            output_data = self.crop_data(original_data, self.output_resolution)
        
        logger.info(f"输出数据形状: {output_data.shape}")
        
        # 5. 归一化数据
        if self.normalize_data:
            logger.info("步骤 5/6: 归一化数据")
            input_data = self.normalize_data(input_data)
            output_data = self.normalize_data(output_data)
            logger.info("数据归一化完成")
        else:
            logger.info("步骤 5/6: 跳过数据归一化")
        
        # 6. 分割数据集并保存
        logger.info("步骤 6/6: 分割数据集并保存")
        self.split_and_save_data(input_data, output_data, output_path)
        
        # 保存归一化信息
        if self.normalize_data:
            self.save_normalization_info(output_path)
        
        # 保存配置信息
        self.save_config_info(output_path)
        
        end_time = time.time()
        logger.info(f"[OK] 数据预处理完成！总耗时: {end_time - start_time:.2f}秒")
        logger.info(f"[INFO] 预处理数据保存在: {output_path}")
    
    def split_and_save_data(self, input_data: np.ndarray, output_data: np.ndarray, output_path: Path):
        """
        分割数据集并保存
        
        Args:
            input_data: 输入数据
            output_data: 输出数据
            output_path: 输出路径
        """
        total_samples = input_data.shape[0]
        
        # 计算分割点
        train_size = int(self.train_ratio * total_samples)
        valid_size = int(self.valid_ratio * total_samples)
        test_size = total_samples - train_size - valid_size
        
        logger.info(f"数据集分割: 训练集={train_size}, 验证集={valid_size}, 测试集={test_size}")
        
        # 分割数据
        train_input = input_data[:train_size]
        train_output = output_data[:train_size]
        
        valid_input = input_data[train_size:train_size + valid_size]
        valid_output = output_data[train_size:train_size + valid_size]
        
        test_input = input_data[train_size + valid_size:]
        test_output = output_data[train_size + valid_size:]
        
        # 保存训练集
        train_path = output_path / 'train_data.h5'
        with h5py.File(train_path, 'w') as f:
            f.create_dataset('input', data=train_input, compression='gzip')
            f.create_dataset('output', data=train_output, compression='gzip')
        logger.info(f"[OK] 训练集已保存: {train_path}")
        
        # 保存验证集
        valid_path = output_path / 'valid_data.h5'
        with h5py.File(valid_path, 'w') as f:
            f.create_dataset('input', data=valid_input, compression='gzip')
            f.create_dataset('output', data=valid_output, compression='gzip')
        logger.info(f"[OK] 验证集已保存: {valid_path}")
        
        # 保存测试集
        test_path = output_path / 'test_data.h5'
        with h5py.File(test_path, 'w') as f:
            f.create_dataset('input', data=test_input, compression='gzip')
            f.create_dataset('output', data=test_output, compression='gzip')
        logger.info(f"[OK] 测试集已保存: {test_path}")
    
    def save_normalization_info(self, output_path: Path):
        """
        保存归一化信息
        
        Args:
            output_path: 输出路径
        """
        norm_info = {
            'global_min': float(self.global_min),
            'global_max': float(self.global_max),
            'normalization_method': 'global_minmax',
            'input_resolution': self.input_resolution,
            'output_resolution': self.output_resolution,
            'num_samples': self.num_samples
        }
        
        norm_path = output_path / 'normalization_info.json'
        with open(norm_path, 'w', encoding='utf-8') as f:
            json.dump(norm_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[INFO] 归一化信息已保存: {norm_path}")
    
    def save_config_info(self, output_path: Path):
        """
        保存配置信息
        
        Args:
            output_path: 输出路径
        """
        config_path = output_path / 'preprocessing_config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"⚙️ 预处理配置已保存: {config_path}")

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='数据预处理器')
    parser.add_argument('--config', type=str, default='dynamic_config_server_optimized_final.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='原始数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_data',
                       help='预处理数据输出目录')
    parser.add_argument('--num_samples', type=int,
                       help='处理的样本数量（覆盖配置文件）')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    logger.info(f"加载配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("[OK] 配置文件加载成功")
    return config

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 覆盖命令行参数
        if args.num_samples:
            config['data']['num_samples'] = args.num_samples
            logger.info(f"使用命令行指定的样本数量: {args.num_samples}")
        
        # 创建数据预处理器
        preprocessor = DataPreprocessor(config)
        
        # 执行数据预处理
        preprocessor.process_data(args.data_path, args.output_dir)
        
        logger.info("[OK] 数据预处理完成！")
        logger.info("[TIP] 使用提示:")
        logger.info(f"   - 预处理数据位于: {args.output_dir}")
        logger.info("   - 可以使用 simple_trainer.py 直接加载预处理数据进行训练")
        logger.info("   - 归一化信息已保存，可用于反归一化预测结果")
        
    except KeyboardInterrupt:
        logger.warning("[WARN] 数据预处理被用户中断")
    except FileNotFoundError as e:
        logger.error(f"[ERROR] 文件未找到: {str(e)}")
        logger.error("请检查数据文件路径是否正确")
    except Exception as e:
        logger.error(f"[ERROR] 数据预处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()