#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import traceback
import numpy as np
import h5py
from pathlib import Path

# 添加路径
sys.path.insert(0, '.')
sys.path.insert(0, '..')

try:
    from pde_process.resolution_downsampler.resolution_downsampler import DownsampledResolutionDataset
    print("✅ 模块导入成功")
    
    # 创建测试数据
    test_file = 'test_simple.h5'
    with h5py.File(test_file, 'w') as f:
        test_data = np.random.rand(5, 64, 64).astype(np.float32)
        f.create_dataset('tensor', data=test_data)
    print("✅ 测试数据创建成功")
    
    # 创建数据集
    dataset = DownsampledResolutionDataset(
        data_path=test_file,
        input_resolution=(32, 32),
        output_resolution=(16, 16),
        num_samples=3,
        downsample_method='bilinear',
        normalize_data=True,
        lazy_loading=False
    )
    print("✅ 数据集创建成功")
    
    # 测试数据加载
    sample_input, sample_output, sample_idx = dataset[0]
    print(f"✅ 数据加载成功")
    print(f"输入形状: {sample_input.shape}")
    print(f"输出形状: {sample_output.shape}")
    print(f"样本索引: {sample_idx}")
    
    # 清理
    Path(test_file).unlink(missing_ok=True)
    print("✅ 测试完成")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    traceback.print_exc()