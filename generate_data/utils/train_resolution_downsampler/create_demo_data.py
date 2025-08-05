#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建演示数据文件
"""

import h5py
import numpy as np
from pathlib import Path

def create_demo_data():
    """创建演示数据文件"""
    # 创建演示数据 (50个样本，128x128分辨率)
    data = np.random.rand(50, 128, 128).astype(np.float32)
    
    # 保存到HDF5文件
    demo_file = Path('demo_data.h5')
    with h5py.File(demo_file, 'w') as f:
        f.create_dataset('tensor', data=data)
    
    print(f"演示数据已创建: {demo_file}")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: [{data.min():.6f}, {data.max():.6f}]")
    
    return str(demo_file)

if __name__ == "__main__":
    create_demo_data()