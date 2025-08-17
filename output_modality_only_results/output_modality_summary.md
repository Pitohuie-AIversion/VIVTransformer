# 输出数据模态分析报告

## 数据集信息
- 文件路径: `demo_data.h5`
- tensor: 形状=(50, 128, 128), 大小=3.12MB

## 模态分析结果
- **分析类型**: 仅输出模态分析
- **有效模态数**: 32
- **压缩比**: 512.0x
- **重建相对误差**: 0.408703

## 能量保持分析
- **90%能量所需模态数**: 35
- **95%能量所需模态数**: 37
- **99%能量所需模态数**: 39

## 主要发现
- 数据复杂度适中
- 重建误差较大，可能需要更多模态或检查数据质量

## 样本间模态相似度
- 余弦相似度 平均: -0.026, 最高: 0.275, 最低: -0.345
- Top相似样本对:
  - Top1: 样本(2, 3) 余弦相似度=0.275
  - Top2: 样本(17, 33) 余弦相似度=0.244
  - Top3: 样本(5, 14) 余弦相似度=0.230
  - Top4: 样本(9, 17) 余弦相似度=0.211
  - Top5: 样本(4, 39) 余弦相似度=0.201
- 相似度热力图PNG: output_modality_only_results\output_sample_similarity.png
- 相似度热力图SVG: output_modality_only_results\output_sample_similarity.svg
- 矩阵文件: Cosine=output_modality_only_results\cosine_similarity.npy, Euclidean=output_modality_only_results\euclidean_distance.npy
