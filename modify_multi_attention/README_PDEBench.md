# PDEBench数据集集成指南

本项目已成功集成PDEBench数据集，支持HDF5格式的PDE数据加载和训练。

## 功能特性

- ✅ 自动检测数据格式（PyTorch .pt 或 HDF5 .h5/.hdf5）
- ✅ 支持单个HDF5文件或包含多个HDF5文件的目录
- ✅ 数据预处理和展平以适配现有模型
- ✅ 样本数量限制功能
- ✅ 详细的数据统计信息
- ✅ 兼容原有的压力数据集

## 快速开始

### 1. 数据准备

确保PDEBench数据文件位于正确路径：
```
x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5
```

### 2. 配置文件

使用PDEBench数据集的配置示例（`configs/config_pdebench.yaml`）：

```yaml
data:
  path: "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
  batch_size: 32
  dataset_type: "pdebench"  # 明确指定使用PDEBench数据集
  max_samples: 1000  # 限制样本数量以加快训练

model:
  input_dim: 16384  # PDEBench数据展平后的维度 (128*128)
  output_dim: 16384  # 与输入维度相同
```

### 3. 运行测试

快速测试PDEBench数据加载：
```bash
python run_pdebench_demo.py --mode test
```

### 4. 开始训练

使用PDEBench数据集进行训练：
```bash
python main.py --config configs/config_pdebench.yaml
```

或使用演示脚本：
```bash
python run_pdebench_demo.py --mode train
```

## 数据集配置选项

### dataset_type 参数

- `"auto"`: 自动检测数据格式（默认）
- `"pressure"`: 强制使用压力数据集格式
- `"pdebench"`: 强制使用PDEBench数据集格式

### max_samples 参数

- `null`: 使用所有可用样本（默认）
- `数字`: 限制使用的样本数量，用于快速测试或调试

## 数据格式说明

### PDEBench数据结构

- **输入格式**: HDF5文件，包含多个时间步的PDE解
- **数据形状**: 原始数据为 (时间步, 高度, 宽度)，如 (10000, 128, 128)
- **处理后**: 展平为 (16384,) 向量以适配现有模型
- **时间步**: 作为额外信息保留

### 数据加载流程

1. 检测文件格式（.h5/.hdf5 vs .pt）
2. 读取HDF5文件中的所有组和数据集
3. 将2D/3D数据展平为1D向量
4. 创建输入-目标对（当前时间步 → 下一时间步）
5. 分割为训练/验证/测试集（70%/15%/15%）

## 文件结构

```
modify_multi_attention/
├── data/
│   ├── pdebench_dataset.py      # PDEBench数据集类
│   └── dataloader.py            # 统一数据加载器
├── configs/
│   ├── config.yaml              # 原始配置（压力数据）
│   └── config_pdebench.yaml     # PDEBench配置示例
├── test_pdebench_integration.py # 集成测试脚本
├── run_pdebench_demo.py         # 演示脚本
└── README_PDEBench.md           # 本文档
```

## 性能建议

1. **内存优化**: 使用 `max_samples` 参数限制样本数量
2. **批处理大小**: PDEBench数据较大，建议使用较小的batch_size（16-32）
3. **GPU内存**: 监控CUDA内存使用，必要时减小batch_size

## 故障排除

### 常见问题

1. **路径错误**: 确保数据文件路径正确且文件存在
2. **内存不足**: 减小batch_size或使用max_samples限制
3. **格式不支持**: 确保文件扩展名为.h5或.hdf5

### 调试命令

```bash
# 测试数据加载
python test_pdebench_integration.py

# 快速验证
python run_pdebench_demo.py --mode test
```

## 扩展功能

- 支持更多PDEBench数据集类型
- 自定义数据预处理管道
- 多GPU训练支持
- 数据增强功能

---

**注意**: 确保安装了必要的依赖包：`h5py`, `torch`, `numpy`