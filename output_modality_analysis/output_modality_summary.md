# 输出数据模态分析报告

## 数据集信息
- 文件路径: `x:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5`
- nu: 形状=(10000, 128, 128), 大小=625.00MB
- tensor: 形状=(10000, 1, 128, 128), 大小=625.00MB
- x-coordinate: 形状=(128,), 大小=0.00MB
- y-coordinate: 形状=(128,), 大小=0.00MB

## 模态分析结果
- **分析类型**: 仅输出模态分析
- **有效模态数**: 11
- **压缩比**: 1489.5x
- **重建相对误差**: 0.098865

## 能量保持分析
- **90%能量所需模态数**: 1
- **95%能量所需模态数**: 3
- **99%能量所需模态数**: 11

## 主要发现
- 数据具有很强的低维结构，可以用较少的模态表示
- 重建质量很好，SVD能够很好地捕获数据结构
