#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间裁剪演示脚本 - 32x32裁剪

功能:
1. 直接加载Darcy Flow数据集
2. 演示32x32空间裁剪功能
3. 可视化原始数据和裁剪后的数据
4. 展示裁剪前后的数据统计信息

作者: AI Assistant
日期: 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_crop_data():
    """
    直接加载数据并进行32x32裁剪
    """
    data_path = r"X:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5"
    
    print("=== 加载和裁剪Darcy Flow数据 ===")
    print(f"数据文件: {data_path}")
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"\n文件键: {list(f.keys())}")
            
            if 'tensor' in f:
                tensor_data = f['tensor']
                print(f"原始数据形状: {tensor_data.shape}")
                print(f"数据类型: {tensor_data.dtype}")
                
                # 读取前5个样本
                num_samples = 5
                original_data = np.array(tensor_data[:num_samples], dtype=np.float32)
                print(f"加载样本数: {num_samples}")
                print(f"加载数据形状: {original_data.shape}")
                
                # 如果数据是4D (samples, channels, height, width)，移除通道维度
                if len(original_data.shape) == 4 and original_data.shape[1] == 1:
                    original_data = original_data.squeeze(1)  # 移除通道维度
                    print(f"移除通道维度后形状: {original_data.shape}")
                
                # 执行32x32空间裁剪
                height, width = original_data.shape[1], original_data.shape[2]
                print(f"原始空间尺寸: {height}x{width}")
                
                # 计算中心裁剪区域
                crop_size = 32
                center_h = height // 2
                center_w = width // 2
                start_h = center_h - crop_size // 2
                end_h = start_h + crop_size
                start_w = center_w - crop_size // 2
                end_w = start_w + crop_size
                
                print(f"裁剪区域: [{start_h}:{end_h}, {start_w}:{end_w}]")
                
                # 执行裁剪
                cropped_data = original_data[:, start_h:end_h, start_w:end_w]
                print(f"裁剪后形状: {cropped_data.shape}")
                
                return original_data, cropped_data, (start_h, end_h, start_w, end_w)
            else:
                print("错误: 未找到'tensor'键")
                return None, None, None
                
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return None, None, None

def visualize_spatial_crop(original_data, cropped_data, crop_region, sample_idx=0):
    """
    可视化空间裁剪结果
    
    Args:
        original_data: 原始数据 [samples, height, width]
        cropped_data: 裁剪后的数据 [samples, crop_height, crop_width]
        crop_region: 裁剪区域 (start_h, end_h, start_w, end_w)
        sample_idx: 要可视化的样本索引
    """
    start_h, end_h, start_w, end_w = crop_region
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始数据
    orig_sample = original_data[sample_idx]
    im1 = axes[0].imshow(orig_sample, cmap='viridis')
    axes[0].set_title(f'原始数据 {orig_sample.shape}\n样本 {sample_idx+1}')
    axes[0].set_xlabel('X坐标')
    axes[0].set_ylabel('Y坐标')
    
    # 在原始图上标记裁剪区域
    from matplotlib.patches import Rectangle
    rect = Rectangle((start_w, start_h), end_w-start_w, end_h-start_h, 
                    linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(start_w, start_h-5, '裁剪区域', color='red', fontsize=10, weight='bold')
    
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 裁剪后的数据
    crop_sample = cropped_data[sample_idx]
    im2 = axes[1].imshow(crop_sample, cmap='viridis')
    axes[1].set_title(f'裁剪后数据 {crop_sample.shape}\n32x32区域')
    axes[1].set_xlabel('X坐标')
    axes[1].set_ylabel('Y坐标')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 对比图 - 从原始数据中提取相同区域进行验证
    orig_crop_region = orig_sample[start_h:end_h, start_w:end_w]
    diff = np.abs(orig_crop_region - crop_sample)
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'差异图\n最大差异: {diff.max():.2e}')
    axes[2].set_xlabel('X坐标')
    axes[2].set_ylabel('Y坐标')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('spatial_crop_32x32_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print(f"\n=== 样本 {sample_idx+1} 统计信息 ===")
    print(f"原始数据:")
    print(f"  形状: {orig_sample.shape}")
    print(f"  数值范围: [{orig_sample.min():.4f}, {orig_sample.max():.4f}]")
    print(f"  均值: {orig_sample.mean():.4f}")
    print(f"  标准差: {orig_sample.std():.4f}")
    
    print(f"\n裁剪后数据:")
    print(f"  形状: {crop_sample.shape}")
    print(f"  数值范围: [{crop_sample.min():.4f}, {crop_sample.max():.4f}]")
    print(f"  均值: {crop_sample.mean():.4f}")
    print(f"  标准差: {crop_sample.std():.4f}")
    
    print(f"\n裁剪验证:")
    print(f"  最大差异: {diff.max():.2e} (应该为0或接近0)")
    print(f"  差异均值: {diff.mean():.2e}")

def analyze_multiple_samples(original_data, cropped_data):
    """
    分析多个样本的统计信息
    """
    print(f"\n=== 多样本分析 ===")
    print(f"总样本数: {original_data.shape[0]}")
    
    for i in range(min(3, original_data.shape[0])):
        orig_sample = original_data[i]
        crop_sample = cropped_data[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  原始: 均值={orig_sample.mean():.4f}, 标准差={orig_sample.std():.4f}, 范围=[{orig_sample.min():.4f}, {orig_sample.max():.4f}]")
        print(f"  裁剪: 均值={crop_sample.mean():.4f}, 标准差={crop_sample.std():.4f}, 范围=[{crop_sample.min():.4f}, {crop_sample.max():.4f}]")
        
        # 计算裁剪区域相对于原始数据的统计特性
        ratio_mean = crop_sample.mean() / orig_sample.mean() if orig_sample.mean() != 0 else 0
        ratio_std = crop_sample.std() / orig_sample.std() if orig_sample.std() != 0 else 0
        print(f"  比率: 均值比={ratio_mean:.3f}, 标准差比={ratio_std:.3f}")

def demonstrate_spatial_crop():
    """
    演示32x32空间裁剪功能
    """
    print("=== PDEBench 32x32空间裁剪演示 ===")
    print("数据集: 2D_DarcyFlow_beta0.1_Train.hdf5")
    print("裁剪方式: 从中心区域裁剪32x32")
    
    # 1. 加载和裁剪数据
    original_data, cropped_data, crop_region = load_and_crop_data()
    
    if original_data is None:
        print("数据加载失败，演示终止")
        return
    
    # 2. 可视化第一个样本
    print("\n=== 可视化结果 ===")
    visualize_spatial_crop(original_data, cropped_data, crop_region, sample_idx=0)
    
    # 3. 分析多个样本
    analyze_multiple_samples(original_data, cropped_data)
    
    # 4. 保存裁剪后的数据
    print("\n=== 保存数据 ===")
    output_file = "spatial_crop_32x32_demo.npz"
    np.savez(output_file, 
             original=original_data, 
             cropped=cropped_data,
             crop_region=crop_region)
    print(f"数据已保存到: {output_file}")
    
    # 5. 展示空间裁剪的应用场景
    print("\n=== 空间裁剪应用场景 ===")
    print("1. 减少计算复杂度: 32x32 vs 128x128 = 16倍减少")
    print("2. 关注感兴趣区域: 中心区域通常包含主要特征")
    print("3. 数据增强: 可以从不同位置裁剪获得更多样本")
    print("4. 内存优化: 减少存储和传输需求")
    
    # 6. 计算压缩比
    original_size = original_data.nbytes
    cropped_size = cropped_data.nbytes
    compression_ratio = original_size / cropped_size
    print(f"\n=== 数据压缩效果 ===")
    print(f"原始数据大小: {original_size / 1024 / 1024:.2f} MB")
    print(f"裁剪后大小: {cropped_size / 1024 / 1024:.2f} MB")
    print(f"压缩比: {compression_ratio:.1f}x")
    print(f"空间减少: {(1 - 1/compression_ratio)*100:.1f}%")

def main():
    """
    主函数
    """
    try:
        demonstrate_spatial_crop()
        print("\n=== 演示完成 ===")
        print("生成的文件:")
        print("- spatial_crop_32x32_demo.npz (裁剪后的数据)")
        print("- spatial_crop_32x32_visualization.png (可视化图片)")
        
    except Exception as e:
        print(f"演示过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()