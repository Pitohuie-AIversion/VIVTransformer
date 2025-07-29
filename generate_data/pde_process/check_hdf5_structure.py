#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查HDF5文件结构的脚本
"""

import h5py
import numpy as np

def check_hdf5_structure(file_path):
    """检查HDF5文件的结构"""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件: {file_path}")
            print(f"顶级键: {list(f.keys())}")
            
            for key in f.keys():
                dataset = f[key]
                print(f"\n数据集: {key}")
                print(f"  形状: {dataset.shape}")
                print(f"  数据类型: {dataset.dtype}")
                
                if len(dataset.shape) > 0:
                    print(f"  数据范围: [{np.min(dataset[...]):.6f}, {np.max(dataset[...]):.6f}]")
                
                # 如果是tensor数据集，显示更多信息
                if key == 'tensor':
                    print(f"  详细形状分析:")
                    if len(dataset.shape) == 4:
                        print(f"    维度解释: [samples, time_steps, height, width]")
                        print(f"    样本数: {dataset.shape[0]}")
                        print(f"    时间步数: {dataset.shape[1]}")
                        print(f"    空间尺寸: {dataset.shape[2]} x {dataset.shape[3]}")
                    elif len(dataset.shape) == 3:
                        print(f"    维度解释: [samples, height, width] 或 [time_steps, height, width]")
                        print(f"    第一维: {dataset.shape[0]}")
                        print(f"    空间尺寸: {dataset.shape[1]} x {dataset.shape[2]}")
                    
                    # 显示一些样本数据
                    print(f"  样本数据预览:")
                    if len(dataset.shape) >= 2:
                        sample_data = dataset[0] if len(dataset.shape) > 2 else dataset
                        print(f"    第一个样本形状: {sample_data.shape}")
                        print(f"    第一个样本数据范围: [{np.min(sample_data):.6f}, {np.max(sample_data):.6f}]")
            
    except Exception as e:
        print(f"检查文件时出错: {str(e)}")

if __name__ == "__main__":
    file_path = r"x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
    check_hdf5_structure(file_path)