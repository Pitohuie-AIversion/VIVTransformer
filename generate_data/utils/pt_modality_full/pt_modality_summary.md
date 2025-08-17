# PyTorch .pt 文件模态分析报告

## 数据集信息
- 文件路径: `X:\2025\Graduation_project\simulation_results\merged_all_pressures_separated_normalized.pt`
- 数据类型: dict
- 包含键: ['pressure', 'in_pressure', 'reynolds_numbers', 'time_steps', 'min_values', 'max_values']
- pressure: 形状=[5, 176, 200, 200], 大小=134.28MB
- in_pressure: 形状=[5, 176, 20, 20], 大小=1.34MB

## 模态分析结果
- **分析类型**: PyTorch .pt 文件模态分析
- **有效模态数**: 4
- **压缩比**: 10000.0x
- **重建相对误差**: 0.000001

## 能量保持分析
- **90%能量所需模态数**: 2
- **95%能量所需模态数**: 3
- **99%能量所需模态数**: 4

## 主要发现
- 数据具有很强的低维结构，可以用较少的模态表示
- 重建质量很好，SVD能够很好地捕获数据结构
