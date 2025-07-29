#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建输入32x32、输出128x128的数据集

功能:
1. 加载原始128x128的Darcy Flow数据
2. 创建32x32的输入（空间裁剪）
3. 保持128x128的输出（原始尺寸）
4. 保存为训练用的数据集

作者: AI Assistant
日期: 2025
"""

import os
import numpy as np
import h5py
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_darcy_data(data_path, num_samples=100):
    """
    加载Darcy Flow数据
    
    Args:
        data_path: 数据文件路径
        num_samples: 加载的样本数量
    
    Returns:
        numpy.ndarray: 形状为 (num_samples, 128, 128) 的数据
    """
    logger.info(f"加载Darcy Flow数据: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        logger.info(f"文件键: {list(f.keys())}")
        
        if 'tensor' in f:
            tensor_data = f['tensor']
            logger.info(f"原始数据形状: {tensor_data.shape}")
            
            # 读取指定数量的样本
            actual_samples = min(num_samples, tensor_data.shape[0])
            data = np.array(tensor_data[:actual_samples], dtype=np.float32)
            
            # 如果数据是4D (samples, channels, height, width)，移除通道维度
            if len(data.shape) == 4 and data.shape[1] == 1:
                data = data.squeeze(1)  # 移除通道维度
                logger.info(f"移除通道维度后形状: {data.shape}")
            
            return data
        else:
            raise ValueError("数据文件中未找到'tensor'键")

def create_spatial_crop(data, crop_size=32):
    """
    创建空间裁剪数据
    
    Args:
        data: 原始数据，形状为 (num_samples, height, width)
        crop_size: 裁剪尺寸
    
    Returns:
        numpy.ndarray: 裁剪后的数据，形状为 (num_samples, crop_size, crop_size)
    """
    num_samples, height, width = data.shape
    
    # 计算裁剪区域（中心裁剪）
    start_h = (height - crop_size) // 2
    end_h = start_h + crop_size
    start_w = (width - crop_size) // 2
    end_w = start_w + crop_size
    
    logger.info(f"裁剪区域: [{start_h}:{end_h}, {start_w}:{end_w}]")
    
    # 执行裁剪
    cropped_data = data[:, start_h:end_h, start_w:end_w]
    logger.info(f"裁剪后形状: {cropped_data.shape}")
    
    return cropped_data

def create_input_output_pairs(original_data, cropped_data):
    """
    创建输入-输出对
    
    Args:
        original_data: 原始128x128数据
        cropped_data: 裁剪后32x32数据
    
    Returns:
        tuple: (inputs, outputs)
            - inputs: 32x32数据展平为1024维
            - outputs: 128x128数据展平为16384维
    """
    # 展平空间维度
    inputs = cropped_data.reshape(cropped_data.shape[0], -1)  # (num_samples, 1024)
    outputs = original_data.reshape(original_data.shape[0], -1)  # (num_samples, 16384)
    
    logger.info(f"输入形状: {inputs.shape} (32x32 = {32*32})")
    logger.info(f"输出形状: {outputs.shape} (128x128 = {128*128})")
    
    return inputs, outputs

def save_dataset(inputs, outputs, save_path):
    """
    保存数据集
    
    Args:
        inputs: 输入数据
        outputs: 输出数据
        save_path: 保存路径
    """
    logger.info(f"保存数据集到: {save_path}")
    
    np.savez_compressed(
        save_path,
        inputs=inputs,
        outputs=outputs,
        input_shape=(32, 32),
        output_shape=(128, 128),
        description="输入32x32，输出128x128的Darcy Flow数据集"
    )
    
    # 计算文件大小
    file_size = Path(save_path).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"数据集大小: {file_size:.2f} MB")

def analyze_dataset(inputs, outputs):
    """
    分析数据集统计信息
    
    Args:
        inputs: 输入数据
        outputs: 输出数据
    """
    logger.info("=== 数据集统计信息 ===")
    logger.info(f"样本数量: {inputs.shape[0]}")
    logger.info(f"输入维度: {inputs.shape[1]} (32x32)")
    logger.info(f"输出维度: {outputs.shape[1]} (128x128)")
    
    logger.info(f"输入数据范围: [{inputs.min():.6f}, {inputs.max():.6f}]")
    logger.info(f"输入数据均值: {inputs.mean():.6f}")
    logger.info(f"输入数据标准差: {inputs.std():.6f}")
    
    logger.info(f"输出数据范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
    logger.info(f"输出数据均值: {outputs.mean():.6f}")
    logger.info(f"输出数据标准差: {outputs.std():.6f}")

def main():
    """
    主函数
    """
    # 数据路径
    data_path = r"X:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5"
    save_path = "input32_output128_dataset.npz"
    
    try:
        # 1. 加载原始数据
        logger.info("=== 步骤1: 加载原始数据 ===")
        original_data = load_darcy_data(data_path, num_samples=100)
        
        # 2. 创建空间裁剪
        logger.info("=== 步骤2: 创建空间裁剪 ===")
        cropped_data = create_spatial_crop(original_data, crop_size=32)
        
        # 3. 创建输入-输出对
        logger.info("=== 步骤3: 创建输入-输出对 ===")
        inputs, outputs = create_input_output_pairs(original_data, cropped_data)
        
        # 4. 分析数据集
        logger.info("=== 步骤4: 分析数据集 ===")
        analyze_dataset(inputs, outputs)
        
        # 5. 保存数据集
        logger.info("=== 步骤5: 保存数据集 ===")
        save_dataset(inputs, outputs, save_path)
        
        logger.info("=== 数据集创建完成 ===")
        logger.info(f"生成文件: {save_path}")
        logger.info("数据格式:")
        logger.info("  - inputs: (num_samples, 1024) - 32x32裁剪数据")
        logger.info("  - outputs: (num_samples, 16384) - 128x128原始数据")
        
    except Exception as e:
        logger.error(f"创建数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()