# 输出数据模态分析报告

## 数据集信息
- 文件路径: `x:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5`
- nu: 形状=(10000, 128, 128), 大小=625.00MB
- tensor: 形状=(10000, 1, 128, 128), 大小=625.00MB
- x-coordinate: 形状=(128,), 大小=0.00MB
- y-coordinate: 形状=(128,), 大小=0.00MB

## 模态分析结果
- **分析类型**: 仅输出模态分析
- **有效模态数**: 9
- **压缩比**: 1820.4x
- **重建相对误差**: 0.096694

## 能量保持分析
- **90%能量所需模态数**: 1
- **95%能量所需模态数**: 3
- **99%能量所需模态数**: 9

## 主要发现
- 数据具有很强的低维结构，可以用较少的模态表示
- 重建质量很好，SVD能够很好地捕获数据结构

## 样本间模态相似度
- 余弦相似度 平均: 0.236, 最高: 1.000, 最低: -0.990
- Top相似样本对:
  - Top1: 样本(60, 176) 余弦相似度=1.000
  - Top2: 样本(147, 176) 余弦相似度=1.000
  - Top3: 样本(60, 147) 余弦相似度=0.999
  - Top4: 样本(24, 70) 余弦相似度=0.999
  - Top5: 样本(58, 176) 余弦相似度=0.999
- 相似度热力图PNG: output_modality_only_results_darcy010\output_sample_similarity.png
- 相似度热力图SVG: output_modality_only_results_darcy010\output_sample_similarity.svg
- 矩阵文件: Cosine=output_modality_only_results_darcy010\cosine_similarity.npy, Euclidean=output_modality_only_results_darcy010\euclidean_distance.npy
