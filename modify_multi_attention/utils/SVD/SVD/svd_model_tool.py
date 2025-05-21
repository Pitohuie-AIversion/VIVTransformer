#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 解决多重 OpenMP 运行时冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

r"""
svd_model_tool.py: 一键运行脚本，用于加载并可视化 SVD 权重，以及检查 Transformer 模型参数
默认使用：
  - SVD 权重目录: F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results\bam\svd_results\mode1
  - 模型 checkpoint 目录:  F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results\bam
运行方式：
  python svd_model_tool.py
脚本会自动：
 1. 查找并加载 SVD 权重 .pt 文件
 2. 打印 SVD 权重列表结构
 3. 弹窗批量展示所有 SVD 权重热图（跳过空数据）
 4. 查找并加载模型 checkpoint .pt 文件
 5. 打印模型 checkpoint 参数形状
"""
import torch
import glob
import matplotlib.pyplot as plt

# —— 默认目录 ——
SVD_MODE1_DIR = r"F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results\bam\svd_results\mode1"
MODEL_DIR      = r"F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results\bam"


def find_first_pt_file(directory):
    pattern = os.path.join(directory, "*.pt")
    files = glob.glob(pattern)
    if files:
        return sorted(files)[0]
    return None


def inspect_svd_weights():
    pt_path = find_first_pt_file(SVD_MODE1_DIR)
    if not pt_path:
        print(f"Error: 在 {SVD_MODE1_DIR} 下未找到任何 .pt 文件")
        return None
    data = torch.load(pt_path)
    print(f"Loaded SVD weights: {pt_path}")
    print(f"  Reynolds count: {len(data)}")
    for i_re, time_list in enumerate(data):
        print(f"  [Re idx {i_re}] time steps: {len(time_list)}; first time step iterations: {len(time_list[0])}")
        break
    return data


def batch_plot_svd_weights(data):
    if data is None:
        return
    for i_re, time_list in enumerate(data):
        for i_t, iter_list in enumerate(time_list):
            for i_it, tensor in enumerate(iter_list):
                if tensor is None:
                    continue
                arr = tensor.cpu().numpy()
                plt.figure(figsize=(5,5))
                plt.imshow(arr, cmap='viridis', interpolation='nearest')
                plt.title(f"SVD Mode1 (Re={i_re}, t={i_t}, it={i_it})")
                plt.axis('off')
                plt.show()
    plt.close('all')


def inspect_model_checkpoint():
    pt_path = find_first_pt_file(MODEL_DIR)
    if not pt_path:
        print(f"Error: 在 {MODEL_DIR} 下未找到任何 .pt 文件")
        return None
    state = torch.load(pt_path, map_location='cpu')
    print(f"Loaded model checkpoint: {pt_path}")
    if 'state_dict' in state:
        state = state['state_dict']
    for k, v in state.items():
        print(f"  {k:30s} → {tuple(v.shape)}")
    return state


if __name__ == '__main__':
    # 1-3: SVD 权重部分
    svd_data = inspect_svd_weights()
    batch_plot_svd_weights(svd_data)
    # 4-5: 模型 checkpoint 部分
    inspect_model_checkpoint()
