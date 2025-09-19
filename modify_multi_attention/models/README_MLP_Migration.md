# PINN到MLP模型迁移指南

## 概述

本文档说明如何将增强的PINN模型(`enhanced_pinn.py`)迁移到普通的MLP模型(`enhanced_mlp.py`)，适用于不需要物理约束损失的任务。

## 主要变化

### 1. 移除的功能
- **物理约束损失**: 移除了`PINNLoss`类和相关的PDE函数
- **导数计算**: 移除了`compute_derivatives`方法
- **物理方程**: 移除了`heat_equation_2d`和`poisson_equation_2d`等PDE函数
- **物理损失权重**: 移除了`physics_loss_weight`参数

### 2. 保留的功能
- **傅里叶特征映射**: 保留了`FourierFeatureMapping`类，用于增强坐标表示
- **残差块**: 保留了`ResidualBlock`类，提供更好的梯度流
- **位置编码**: 保留了`SinusoidalPositionalEncoding`类
- **稀疏到稠密映射**: 保留了核心的分辨率转换功能

### 3. 新增功能
- **更灵活的输入处理**: 支持多种输入格式（批量特征、1D序列、2D图像）
- **改进的坐标生成**: 更高效的坐标网格生成
- **插值支持**: 自动处理不同分辨率之间的插值

## 使用方法

### 基本用法

```python
from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d

# 创建1D MLP模型
model_1d = create_enhanced_mlp1d(
    input_channels=1,
    output_channels=1,
    hidden_dim=256,
    num_layers=8,
    input_resolution=32,
    output_resolution=128
)

# 创建2D MLP模型
model_2d = create_enhanced_mlp2d(
    input_channels=1,
    output_channels=1,
    hidden_dim=256,
    num_layers=8,
    input_resolution=(32, 32),
    output_resolution=(128, 128)
)
```

### 输入格式支持

#### 1D模型
```python
# 序列输入: [batch, length, channels]
x_sequence = torch.randn(4, 32, 1)
output = model_1d(x_sequence)  # [4, 128, 1]

# 批量特征输入: [batch, channels]
x_batch = torch.randn(4, 1)
output = model_1d(x_batch)  # [4, 128, 1]
```

#### 2D模型
```python
# 图像输入: [batch, height, width, channels]
x_image = torch.randn(4, 32, 32, 1)
output = model_2d(x_image)  # [4, 128, 128, 1]

# 批量特征输入: [batch, channels]
x_batch = torch.randn(4, 1)
output = model_2d(x_batch)  # [4, 128, 128, 1]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_channels` | int | 1 | 输入通道数 |
| `output_channels` | int | 1 | 输出通道数 |
| `hidden_dim` | int | 256 | 隐藏层维度 |
| `num_layers` | int | 8 | 网络层数 |
| `input_resolution` | int/tuple | 32 | 输入分辨率 |
| `output_resolution` | int/tuple | 128 | 输出分辨率 |
| `use_fourier_features` | bool | True | 是否使用傅里叶特征映射 |
| `fourier_mapping_size` | int | 256 | 傅里叶映射大小 |
| `fourier_scale` | float | 10.0 | 傅里叶映射缩放因子 |
| `use_residual_blocks` | bool | True | 是否使用残差块 |
| `dropout` | float | 0.1 | Dropout概率 |
| `activation` | str | 'gelu' | 激活函数类型 |

## 迁移步骤

### 1. 替换导入
```python
# 原来的PINN导入
# from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d

# 新的MLP导入
from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
```

### 2. 更新模型创建
```python
# 原来的PINN模型创建
# model = create_enhanced_pinn1d(
#     output_dim=1,
#     hidden_dim=256,
#     num_layers=8,
#     input_resolution=32,
#     output_resolution=128
# )

# 新的MLP模型创建
model = create_enhanced_mlp1d(
    input_channels=1,
    output_channels=1,
    hidden_dim=256,
    num_layers=8,
    input_resolution=32,
    output_resolution=128
)
```

### 3. 移除物理损失
```python
# 移除PINN损失函数
# pinn_loss = PINNLoss(pde_function=some_pde_function)

# 使用标准损失函数
loss_fn = nn.MSELoss()
```

### 4. 简化训练循环
```python
# 原来的PINN训练
# loss, loss_dict = pinn_loss(model, x_data, y_data, x_pde, x_boundary)

# 新的MLP训练
predictions = model(x_data)
loss = loss_fn(predictions, y_data)
```

## 性能对比

| 特性 | PINN模型 | MLP模型 |
|------|----------|----------|
| 训练速度 | 较慢（需要计算导数） | 较快 |
| 内存使用 | 较高（梯度计算） | 较低 |
| 物理约束 | 支持 | 不支持 |
| 数据依赖 | 较低 | 较高 |
| 泛化能力 | 物理引导的泛化 | 数据驱动的泛化 |

## 适用场景

### 使用MLP模型的情况
- 有充足的训练数据
- 不需要物理约束
- 追求训练速度
- 纯数据驱动的任务

### 继续使用PINN模型的情况
- 训练数据稀少
- 需要物理约束
- 要求物理一致性
- 需要外推能力

## 注意事项

1. **数据需求**: MLP模型通常需要更多的训练数据来达到良好的性能
2. **泛化能力**: 没有物理约束的MLP可能在训练域外的泛化能力较弱
3. **参数调优**: 可能需要重新调整超参数以获得最佳性能
4. **验证策略**: 建议使用更严格的验证策略来防止过拟合

## 示例代码

完整的使用示例请参考`enhanced_mlp.py`文件中的测试代码。