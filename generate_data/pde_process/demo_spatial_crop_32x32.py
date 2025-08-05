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
                
                # 读取前10个样本
                num_samples = 10
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

def visualize_spatial_crop(original_data, cropped_data, crop_region, sample_idx=0, save_name=None):
    """
    可视化空间裁剪结果
    
    Args:
        original_data: 原始数据 [samples, height, width]
        cropped_data: 裁剪后的数据 [samples, crop_height, crop_width]
        crop_region: 裁剪区域 (start_h, end_h, start_w, end_w)
        sample_idx: 要可视化的样本索引
        save_name: 保存文件名
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
    filename = save_name if save_name else f'spatial_crop_32x32_sample_{sample_idx+1}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化已保存: {filename}")
    
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

def create_multiple_crop_regions(height, width, crop_size=32):
    """
    创建多个不同的裁剪区域
    
    Args:
        height: 原始数据高度
        width: 原始数据宽度
        crop_size: 裁剪尺寸
    
    Returns:
        list: 裁剪区域列表，每个元素为 (start_h, end_h, start_w, end_w, name)
    """
    regions = []
    
    # 只保留中心区域
    center_h = height // 2
    center_w = width // 2
    start_h = center_h - crop_size // 2
    end_h = start_h + crop_size
    start_w = center_w - crop_size // 2
    end_w = start_w + crop_size
    regions.append((start_h, end_h, start_w, end_w, "中心区域"))
    
    return regions

def generate_multiple_visualizations():
    """
    生成多组可视化结果
    """
    print("=== 生成多组可视化 ===")
    
    # 加载数据
    data_path = r"X:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5"
    
    try:
        with h5py.File(data_path, 'r') as f:
            tensor_data = f['tensor']
            # 随机抽取10个样本
            total_samples = tensor_data.shape[0]
            num_samples = 10
            random_indices = np.random.choice(total_samples, num_samples, replace=False)
            random_indices = np.sort(random_indices)  # h5py要求索引按递增顺序
            original_data = np.array(tensor_data[random_indices], dtype=np.float32)
            print(f"随机选择的样本索引: {random_indices}")
            
            if len(original_data.shape) == 4 and original_data.shape[1] == 1:
                original_data = original_data.squeeze(1)
            
            height, width = original_data.shape[1], original_data.shape[2]
            print(f"加载了 {num_samples} 个样本，尺寸: {height}x{width}")
            
            # 创建多个裁剪区域
            crop_regions = create_multiple_crop_regions(height, width)
            
            # 为每个区域生成可视化
            for region_idx, (start_h, end_h, start_w, end_w, region_name) in enumerate(crop_regions):
                print(f"\n=== 处理{region_name} ===")
                
                # 对所有样本进行裁剪
                cropped_data = original_data[:, start_h:end_h, start_w:end_w]
                crop_region = (start_h, end_h, start_w, end_w)
                
                # 为所有10个样本生成可视化
                for sample_idx in range(num_samples):
                    save_name = f'crop_{region_name}_sample_{sample_idx+1}.png'
                    print(f"生成样本 {sample_idx+1} 在{region_name}的可视化...")
                    
                    visualize_spatial_crop(original_data, cropped_data, crop_region, 
                                         sample_idx=sample_idx, save_name=save_name)
                    
                    # 打印该样本在该区域的统计信息
                    orig_sample = original_data[sample_idx]
                    crop_sample = cropped_data[sample_idx]
                    print(f"  样本{sample_idx+1} {region_name}: 原始均值={orig_sample.mean():.4f}, 裁剪均值={crop_sample.mean():.4f}")
            
            # 生成对比图：同一样本的不同区域
            print("\n=== 生成同一样本不同区域对比图 ===")
            sample_idx = 0  # 使用第一个样本
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (start_h, end_h, start_w, end_w, region_name) in enumerate(crop_regions):
                if i >= 6:  # 最多显示6个区域
                    break
                    
                cropped_sample = original_data[sample_idx, start_h:end_h, start_w:end_w]
                im = axes[i].imshow(cropped_sample, cmap='viridis')
                axes[i].set_title(f'{region_name}\n均值: {cropped_sample.mean():.4f}')
                axes[i].set_xlabel('X坐标')
                axes[i].set_ylabel('Y坐标')
                plt.colorbar(im, ax=axes[i], shrink=0.8)
            
            # 隐藏多余的子图
            for i in range(len(crop_regions), 6):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            comparison_filename = f'sample_{sample_idx+1}_all_regions_comparison.png'
            plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"对比图已保存: {comparison_filename}")
            
            # 生成10个样本网格图
            print("\n=== 生成10个随机样本网格图 ===")
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            axes = axes.flatten()
            
            # 使用中心区域裁剪
            center_region = crop_regions[0]  # 中心区域
            start_h, end_h, start_w, end_w, _ = center_region
            
            for i in range(num_samples):
                cropped_sample = original_data[i, start_h:end_h, start_w:end_w]
                im = axes[i].imshow(cropped_sample, cmap='viridis')
                axes[i].set_title(f'样本 {random_indices[i]+1}\n均值: {cropped_sample.mean():.4f}')
                axes[i].set_xlabel('X坐标')
                axes[i].set_ylabel('Y坐标')
                plt.colorbar(im, ax=axes[i], shrink=0.6)
            
            plt.tight_layout()
            grid_filename = 'random_10_samples_grid.png'
            plt.savefig(grid_filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"10个随机样本网格图已保存: {grid_filename}")
            
    except Exception as e:
        print(f"生成多组可视化时出错: {str(e)}")
        import traceback
        traceback.print_exc()

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
    
    # 4. 生成多组可视化
    print("\n=== 生成多组可视化 ===")
    generate_multiple_visualizations()
    
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