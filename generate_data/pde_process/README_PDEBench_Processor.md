# PDEBench数据处理器

这是一个专门用于处理PDEBench数据集的工具包，能够从HDF5格式的PDE数据中截取、处理和转换数据，使其适合机器学习模型训练。

## 功能特性

### 🔧 核心功能
- **多格式支持**: 支持HDF5格式的PDEBench数据集
- **灵活截取**: 支持时间维度和空间维度的灵活截取
- **数据预处理**: 自动归一化、空间下采样、维度展平
- **序列生成**: 创建适合序列预测任务的输入-输出对
- **批量处理**: 支持多种PDE类型的批量处理
- **配置管理**: 基于YAML的灵活配置系统

### 📊 支持的PDE类型
- **Darcy Flow** (2D): 多孔介质中的流体流动
- **Diffusion-Sorption** (1D): 扩散吸附过程
- **Reaction-Diffusion** (2D): 反应扩散系统
- **Burgers Equation** (1D): 伯格斯方程
- **Advection Equation** (1D): 对流方程

## 快速开始

### 1. 环境准备

确保安装了以下依赖:
```bash
pip install numpy torch h5py matplotlib pyyaml tqdm
```

### 2. 数据准备

确保PDEBench数据文件位于正确路径:
```
x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/
├── 2D_DarcyFlow_beta0.1_Train.hdf5
├── 1D_Burgers_Sols_Nu0.001.hdf5
├── 1D_Advection_Sols_beta0.1.hdf5
└── ...
```

### 3. 基本使用

#### 快速测试
```bash
# 运行快速测试，验证系统是否正常工作
python run_pdebench_processor.py --quick_test
```

#### 处理单个PDE类型
```bash
# 使用默认配置处理Darcy Flow数据
python run_pdebench_processor.py --pde_type darcy

# 使用快速测试配置
python run_pdebench_processor.py --config quick_test --pde_type darcy
```

#### 处理多个PDE类型
```bash
# 同时处理多种PDE类型
python run_pdebench_processor.py --config basic --pde_type darcy,burgers,advection
```

#### 查看系统信息
```bash
# 查看所有可用配置
python run_pdebench_processor.py --list_configs

# 查看数据集信息
python run_pdebench_processor.py --dataset_info

# 查看特定PDE类型信息
python run_pdebench_processor.py --dataset_info darcy
```

## 配置系统

### 预定义配置

系统提供了多种预定义配置，适用于不同场景:

| 配置名称 | 用途 | 特点 |
|---------|------|------|
| `basic` | 基本使用 | 平衡的参数设置 |
| `quick_test` | 快速测试 | 小数据量，快速验证 |
| `high_resolution` | 高质量训练 | 保持空间结构，高分辨率 |
| `low_resolution` | 快速训练 | 低分辨率，快速训练 |
| `time_series` | 序列预测 | 长时间序列，适合预测任务 |
| `multi_scale` | 多尺度分析 | 多分辨率处理 |
| `sparse_temporal` | 稀疏采样 | 长期动态分析 |
| `region_focus` | 区域关注 | 关注特定空间区域 |

### 配置参数说明

```yaml
# 基本配置示例
basic:
  pdebench_root: "数据根目录路径"
  max_samples: 1000          # 最大样本数量
  start_time_idx: 0          # 开始时间索引
  end_time_idx: null         # 结束时间索引
  time_step_interval: 1      # 时间步间隔
  spatial_crop: null         # 空间裁剪 [[x_start, x_end], [y_start, y_end]]
  spatial_downsample: 1      # 空间下采样因子
  output_file: "output.pt"   # 输出文件名
  normalize_data: true       # 是否归一化
  flatten_spatial: true      # 是否展平空间维度
  sequence_length: 1         # 序列长度
```

## 编程接口

### 基本使用

```python
from pdebench_data_processor import PDEBenchProcessor, PDEBenchConfig
from config_loader import ConfigLoader

# 方法1: 使用配置文件
loader = ConfigLoader('pdebench_config.yaml')
config = loader.create_config('basic')
processor = PDEBenchProcessor(config)

# 处理数据
success = processor.process_pde_dataset('darcy', sequence_length=5)
if success:
    processor.save_processed_data()

# 方法2: 手动配置
config = PDEBenchConfig()
config.MAX_SAMPLES = 100
config.END_TIME_IDX = 50
config.SPATIAL_DOWNSAMPLE = 2

processor = PDEBenchProcessor(config)
processor.process_pde_dataset('darcy')
processor.save_processed_data()
```

### 高级使用

```python
# 自定义空间处理
config = PDEBenchConfig()
config.SPATIAL_CROP = ((10, 50), (10, 50))  # 裁剪区域
config.SPATIAL_DOWNSAMPLE = 4               # 4倍下采样
config.FLATTEN_SPATIAL = False              # 保持空间维度

# 时间序列配置
config.START_TIME_IDX = 0
config.END_TIME_IDX = 200
config.TIME_STEP_INTERVAL = 2

processor = PDEBenchProcessor(config)
processor.process_pde_dataset('darcy', sequence_length=10)
```

## 数据格式

### 输入格式
- **HDF5文件结构**: `seed/data`
- **数据维度**: `[time_steps, spatial_dimensions...]`
- **支持的空间维度**: 1D, 2D, 3D

### 输出格式
```python
# 保存的数据结构
{
    'processed_data': {
        'pde_type': {
            'inputs': torch.Tensor,    # [samples, sequence_length, features]
            'outputs': torch.Tensor,   # [samples, features]
            'sample_info': list,       # 样本信息
            'original_shape': tuple,   # 原始数据形状
            'processed_shape': dict    # 处理后形状
        }
    },
    'normalization_params': {          # 归一化参数
        'pde_type': {
            'min_val': float,
            'max_val': float,
            'normalized': bool
        }
    },
    'config': dict,                    # 处理配置
    'pde_types': list                  # PDE类型列表
}
```

## 使用示例

### 示例1: 快速原型开发
```bash
# 使用快速测试配置，处理少量数据进行原型验证
python run_pdebench_processor.py --config quick_test --pde_type darcy
```

### 示例2: 高质量模型训练
```bash
# 使用高分辨率配置，保持空间结构
python run_pdebench_processor.py --config high_resolution --pde_type darcy --output_file darcy_high_res.pt
```

### 示例3: 多PDE类型联合训练
```bash
# 处理多种PDE类型，用于多任务学习
python run_pdebench_processor.py --config basic --pde_type darcy,burgers,advection --output_file multi_pde.pt
```

### 示例4: 时间序列预测
```bash
# 生成长序列数据，适合时间序列预测任务
python run_pdebench_processor.py --config time_series --pde_type darcy --output_file time_series.pt
```

### 示例5: 区域关注分析
```bash
# 关注特定空间区域，减少计算复杂度
python run_pdebench_processor.py --config region_focus --pde_type darcy --output_file region_data.pt
```

## 性能优化

### 内存优化
- 使用`max_samples`限制样本数量
- 使用`spatial_downsample`减少空间分辨率
- 使用`spatial_crop`关注感兴趣区域
- 设置`flatten_spatial=true`减少维度复杂性

### 计算优化
- 使用较大的`time_step_interval`减少时间相关性
- 合理设置`sequence_length`平衡性能和效果
- 批量处理多个PDE类型

### 存储优化
- 使用适当的数据类型（float32）
- 压缩存储不必要的中间结果
- 定期清理临时文件

## 故障排除

### 常见问题

1. **数据文件不存在**
   ```
   错误: 文件不存在: xxx.hdf5
   解决: 检查PDEBench数据路径和文件名
   ```

2. **内存不足**
   ```
   错误: CUDA out of memory
   解决: 减少max_samples或增加spatial_downsample
   ```

3. **配置验证失败**
   ```
   错误: 配置验证失败
   解决: 检查配置文件格式和参数范围
   ```

4. **数据形状不匹配**
   ```
   错误: 堆叠数据时出错
   解决: 检查时间截取范围和空间处理参数
   ```

### 调试技巧

1. **使用详细输出**
   ```bash
   python run_pdebench_processor.py --verbose --pde_type darcy
   ```

2. **先运行快速测试**
   ```bash
   python run_pdebench_processor.py --quick_test
   ```

3. **检查数据可用性**
   ```bash
   python run_pdebench_processor.py --dataset_info
   ```

4. **查看日志文件**
   ```bash
   tail -f pdebench_processor.log
   ```

## 扩展功能

### 添加新的PDE类型

1. 在`pdebench_config.yaml`中添加新的PDE类型:
```yaml
supported_pdes:
  new_pde: "new_pde_file.hdf5"

dataset_info:
  new_pde:
    description: "新PDE类型描述"
    dimensions: "维度信息"
    physics: "物理背景"
```

2. 确保数据文件存在于指定路径

### 自定义配置

创建新的配置项:
```yaml
custom_config:
  pdebench_root: "your_path"
  max_samples: 500
  # ... 其他参数
```

### 集成到现有项目

```python
# 在现有项目中使用
from generate_data.pdebench_data_processor import PDEBenchProcessor
from generate_data.config_loader import ConfigLoader

# 加载和处理数据
loader = ConfigLoader('path/to/config.yaml')
config = loader.create_config('your_config')
processor = PDEBenchProcessor(config)

# 处理数据并获取结果
processor.process_pde_dataset('darcy')
data = processor.processed_data['darcy']

# 使用处理后的数据
inputs = data['inputs']
outputs = data['outputs']
```

## 文件结构

```
generate_data/
├── pdebench_data_processor.py     # 核心处理器
├── config_loader.py               # 配置加载器
├── pdebench_config.yaml          # 配置文件
├── run_pdebench_processor.py     # 主运行脚本
├── example_pdebench_usage.py     # 使用示例
├── README_PDEBench_Processor.md  # 本文档
└── pdebench_processor.log        # 日志文件
```

## 许可证

本项目遵循MIT许可证。

## 贡献

欢迎提交问题报告和功能请求。在提交代码之前，请确保:
1. 代码通过所有测试
2. 添加适当的文档
3. 遵循现有的代码风格

## 联系方式

如有问题或建议，请通过以下方式联系:
- 创建GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本工具专门为PDEBench数据集设计，确保在使用前已正确下载和配置PDEBench数据。