# 分辨率降采样模块 (Resolution Downsampler)

## 概述

本模块提供了一个与现有裁剪模块功能相似但作用是对整张图降低分辨率的功能。与裁剪方法不同，降采样方法保持了图像的全局信息，避免了信息丢失。

## 主要文件

### 1. `resolution_downsampler.py`
核心模块，包含两个主要类：

#### `ResolutionDownsampler`
- **功能**: 对数据进行分辨率降采样
- **支持的方法**: 
  - `nearest`: 最近邻插值
  - `bilinear`: 双线性插值
  - `bicubic`: 双三次插值
  - `area`: 区域插值
  - `lanczos`: Lanczos插值
- **特性**:
  - 支持SciPy和OpenCV两种后端
  - 自动回退机制（OpenCV不可用时使用SciPy）
  - 支持批量处理
  - 可选的抗锯齿功能

#### `DownsampledResolutionDataset`
- **功能**: 生成使用降采样而非裁剪的不同分辨率数据集
- **特性**:
  - 继承自原始动态分辨率数据集
  - 支持懒加载
  - 支持数据归一化
  - 兼容PyTorch DataLoader

### 2. `downsampler_config.yaml`
配置文件示例，包含：
- 数据路径设置
- 输入输出分辨率配置
- 降采样方法选择
- 训练参数
- 模型配置

### 3. `demo_downsampler_usage.py`
完整的演示脚本，展示：
- 不同降采样方法的比较
- 与裁剪方法的对比
- 信息保持分析
- 数据集使用示例

## 使用方法

### 基本使用

```python
from resolution_downsampler import ResolutionDownsampler

# 创建降采样器
downsampler = ResolutionDownsampler(
    downsample_method='bilinear',
    preserve_aspect_ratio=False,
    anti_aliasing=True
)

# 降采样数据
downsampled_data = downsampler.downsample_data(
    data=original_data,
    target_size=(32, 32),
    method='scipy'  # 或 'opencv'
)
```

### 数据集使用

```python
from resolution_downsampler import DownsampledResolutionDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = DownsampledResolutionDataset(
    data_path="./data",
    input_resolution=(64, 64),
    output_resolution=(32, 32),
    downsample_method='bilinear',
    normalize_data=True
)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 运行演示

```bash
# 设置环境变量避免OpenMP警告
$env:KMP_DUPLICATE_LIB_OK='TRUE'

# 运行演示
python demo_downsampler_usage.py
```

## 主要优势

1. **保持全局信息**: 与裁剪不同，降采样保持了整个图像的信息
2. **多种插值方法**: 支持多种插值算法，适应不同需求
3. **任意尺寸变换**: 可以处理任意尺寸变换，不受原始尺寸限制
4. **更好的频域特征保持**: 降采样在频域上有更好的特性
5. **兼容性强**: 支持多种后端，自动回退机制

## 建议使用场景

- 需要保持全局信息的超分辨率任务
- 多尺度特征学习
- 图像压缩和重建
- 物理场数据的多分辨率分析
- 当原始数据尺寸小于目标输入尺寸时

## 与原始裁剪方法的对比

| 特性 | 裁剪方法 | 降采样方法 |
|------|----------|------------|
| 信息保持 | 部分信息丢失 | 全局信息保持 |
| 计算复杂度 | 低 | 中等 |
| 适用场景 | 局部特征重要 | 全局特征重要 |
| 尺寸限制 | 受原始尺寸限制 | 无限制 |
| 频域特性 | 可能引入高频噪声 | 更好的频域特性 |

## 依赖项

- numpy
- scipy
- torch
- matplotlib (用于演示)
- opencv-python (可选，用于更好的图像处理质量)

## 注意事项

1. 如果没有安装OpenCV，模块会自动使用SciPy作为后端
2. 在某些环境中可能会出现OpenMP库冲突警告，可以通过设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE` 来解决
3. 降采样方法在计算上比裁剪方法稍微复杂，但能提供更好的信息保持