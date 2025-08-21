#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证生成的每模态相似度热力图
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def verify_images():
    """验证生成的图片文件"""
    output_dir = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/output_modality_analysis"
    
    # 查找所有 per_mode_similarity PNG 文件
    png_files = []
    for file in os.listdir(output_dir):
        if file.startswith("per_mode_similarity_mode_") and file.endswith(".png"):
            png_files.append(os.path.join(output_dir, file))
    
    print(f"找到 {len(png_files)} 个每模态相似度热力图文件")
    
    # 验证每个文件
    for png_path in png_files:
        try:
            # 使用 PIL 打开图片
            img = Image.open(png_path)
            print(f"✓ {os.path.basename(png_path)}: {img.size} 像素, 模式: {img.mode}")
            
            # 检查图片是否为空或纯色
            img_array = np.array(img)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
            print(f"  - 唯一颜色数: {unique_colors}")
            
            if unique_colors < 10:
                print(f"  ⚠️ 图片可能存在问题：颜色数过少 ({unique_colors})")
            
        except Exception as e:
            print(f"✗ {os.path.basename(png_path)}: 无法打开 - {e}")
    
    # 尝试重新生成一个简单的热力图作为对比
    print("\n生成测试热力图进行对比...")
    try:
        # 生成随机相似度矩阵
        np.random.seed(42)
        similarity_matrix = np.random.rand(50, 50)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # 对称化
        np.fill_diagonal(similarity_matrix, 1.0)  # 对角线为1
        
        # 绘制热力图
        plt.figure(figsize=(8, 6))
        plt.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        plt.colorbar(label='相似度')
        plt.title('测试热力图 - 随机相似度矩阵')
        plt.xlabel('样本索引')
        plt.ylabel('样本索引')
        
        test_path = os.path.join(output_dir, "test_heatmap.png")
        plt.savefig(test_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 验证测试图片
        test_img = Image.open(test_path)
        print(f"✓ 测试热力图生成成功: {test_img.size} 像素")
        
    except Exception as e:
        print(f"✗ 测试热力图生成失败: {e}")

if __name__ == "__main__":
    verify_images()