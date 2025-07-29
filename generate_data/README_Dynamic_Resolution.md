# 动态分辨率训练器使用指南

## 概述

动态分辨率训练器 (`dynamic_resolution_trainer.py`) 是一个灵活的训练系统，允许您通过配置参数直接控制输入输出分辨率，无需预先生成大型数据文件。系统支持实时数据生成、多种注意力机制和完整的训练流程。

## 主要特性

✅ **动态分辨率配置**: 支持任意输入输出分辨率组合  
✅ **实时数据生成**: 无需预生成大型数据文件  
✅ **多种配置方式**: 命令行参数 + YAML配置文件  
✅ **多种注意力机制**: SGE, CBAM, SE, ECA等  
✅ **完整训练流程**: 训练、验证、测试、可视化  
✅ **断点恢复**: 自动保存和恢复训练进度  
✅ **详细日志**: 完整的训练日志记录  

## 快速开始

### 1. 基本命令行使用

```bash
# 基本训练: 32x32 → 128x128
python dynamic_resolution_trainer.py \
    --input_resolution 32 32 \
    --output_resolution 128 128 \
    --epochs 10 \
    --num_samples 100

# 使用CBAM注意力机制
python dynamic_resolution_trainer.py \
    --input_resolution 16 16 \
    --output_resolution 64 64 \
    --attention_type cbam \
    --epochs 5

# 自定义批次大小和学习率
python dynamic_resolution_trainer.py \
    --input_resolution 24 24 \
    --output_resolution 96 96 \
    --batch_size 8 \
    --learning_rate 0.0001
```

### 2. 使用YAML配置文件

```bash
# 使用默认配置文件
python dynamic_resolution_trainer.py --config dynamic_config.yaml

# 配置文件 + 命令行参数覆盖
python dynamic_resolution_trainer.py \
    --config dynamic_config.yaml \
    --epochs 20 \
    --num_samples 200
```

### 3. 运行示例

```bash
# 运行交互式示例
python quick_start_examples.py

# 运行演示脚本
python demo_dynamic_resolution.py
```

## 配置参数详解

### 数据配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` | str | `../modify_multi_attention/data/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5` | 原始数据路径 |
| `--input_resolution` | int int | `32 32` | 输入分辨率 (高 宽) |
| `--output_resolution` | int int | `128 128` | 输出分辨率 (高 宽) |
| `--num_samples` | int | `100` | 使用的样本数量 |
| `--crop_mode` | str | `center` | 裁剪模式: center/random |

### 训练配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | `10` | 训练轮数 |
| `--batch_size` | int | `8` | 批次大小 |
| `--learning_rate` | float | `0.001` | 学习率 |
| `--patience` | int | `5` | 早停耐心值 |
| `--train_ratio` | float | `0.7` | 训练集比例 |
| `--valid_ratio` | float | `0.15` | 验证集比例 |

### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--attention_type` | str | `sge` | 注意力类型: sge/cbam/se/eca |
| `--num_layers` | int | `6` | Transformer层数 |
| `--d_model` | int | `512` | 模型维度 |
| `--nhead` | int | `8` | 注意力头数 |
| `--dim_feedforward` | int | `2048` | 前馈网络维度 |

### 其他配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--device` | str | `auto` | 设备: auto/cpu/cuda |
| `--seed` | int | `42` | 随机种子 |
| `--log_level` | str | `INFO` | 日志级别 |
| `--save_model` | bool | `True` | 是否保存模型 |

## YAML配置文件示例

```yaml
# dynamic_config.yaml
data:
  data_path: "../modify_multi_attention/data/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
  input_resolution: [32, 32]
  output_resolution: [128, 128]
  num_samples: 100
  crop_mode: "center"
  batch_size: 8
  train_ratio: 0.7
  valid_ratio: 0.15

training:
  epochs: 10
  learning_rate: 0.001
  patience: 5
  save_model: true

model:
  attention_type: "sge"
  num_layers: 6
  d_model: 512
  nhead: 8
  dim_feedforward: 2048

logging:
  level: "INFO"
  save_logs: true
  log_dir: "./logs"

device:
  type: "auto"
  
seed: 42

visualization:
  enabled: true
  save_plots: true
```

## 使用场景示例

### 场景1: 超分辨率重建
```bash
# 16x16 → 128x128 (8倍超分辨率)
python dynamic_resolution_trainer.py \
    --input_resolution 16 16 \
    --output_resolution 128 128 \
    --epochs 20 \
    --attention_type cbam
```

### 场景2: 降采样训练
```bash
# 64x64 → 32x32 (降采样)
python dynamic_resolution_trainer.py \
    --input_resolution 64 64 \
    --output_resolution 32 32 \
    --crop_mode random
```

### 场景3: 非正方形分辨率
```bash
# 32x16 → 128x64 (矩形分辨率)
python dynamic_resolution_trainer.py \
    --input_resolution 32 16 \
    --output_resolution 128 64
```

### 场景4: 快速原型验证
```bash
# 小数据集快速验证
python dynamic_resolution_trainer.py \
    --input_resolution 8 8 \
    --output_resolution 32 32 \
    --num_samples 50 \
    --epochs 3 \
    --batch_size 4
```

## 输出文件说明

训练完成后，系统会生成以下文件：

```
generate_data/
├── models/
│   ├── best_model_{attention_type}.pt     # 最佳模型
│   ├── checkpoint_{attention_type}.pth    # 训练断点
│   └── loss_logs/                         # 损失日志
├── logs/
│   └── dynamic_resolution_training.log    # 训练日志
├── dynamic_resolution_losses.png          # 损失曲线图
└── final_config.yaml                      # 最终使用的配置
```

## 性能优化建议

1. **GPU内存优化**:
   - 大分辨率时减小batch_size
   - 使用梯度累积: `--gradient_accumulation_steps`

2. **训练速度优化**:
   - 使用混合精度训练
   - 适当减少模型层数和维度

3. **数据处理优化**:
   - 使用`center`裁剪模式（比`random`更快）
   - 合理设置`num_samples`

## 故障排除

### 常见问题

1. **CUDA内存不足**:
   ```bash
   # 减小批次大小
   --batch_size 4
   # 或使用CPU
   --device cpu
   ```

2. **数据路径错误**:
   ```bash
   # 检查数据文件是否存在
   ls -la ../modify_multi_attention/data/2D/CFD/2D_Train_Rand/
   ```

3. **配置文件格式错误**:
   ```bash
   # 验证YAML格式
   python -c "import yaml; yaml.safe_load(open('dynamic_config.yaml'))"
   ```

### 调试模式

```bash
# 启用详细日志
python dynamic_resolution_trainer.py \
    --log_level DEBUG \
    --num_samples 10 \
    --epochs 1
```

## 扩展功能

系统支持以下扩展：

1. **自定义数据加载器**: 修改`DynamicResolutionDataset`类
2. **新的注意力机制**: 在模型配置中添加新类型
3. **自定义损失函数**: 修改训练循环中的criterion
4. **数据增强**: 在数据预处理中添加变换

## 联系支持

如有问题或建议，请查看：
- 训练日志: `./logs/dynamic_resolution_training.log`
- 示例脚本: `quick_start_examples.py`
- 演示脚本: `demo_dynamic_resolution.py`