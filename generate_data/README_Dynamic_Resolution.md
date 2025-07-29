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
| `--patience` | int | `15` | 早停耐心值 |
| `--train_ratio` | float | `0.7` | 训练集比例 |
| `--valid_ratio` | float | `0.15` | 验证集比例 |

### 早停配置 (YAML配置文件)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_early_stopping` | bool | `true` | 是否启用早停功能 |
| `patience` | int | `15` | 早停耐心值（验证损失不改善的epoch数） |
| `min_delta` | float | `0.000001` | 最小改善阈值 |
| `monitor` | str | `"val_loss"` | 监控指标: 'val_loss', 'train_loss', 'test_loss' |
| `mode` | str | `"min"` | 监控模式: 'min'(损失), 'max'(准确率) |
| `restore_best_weights` | bool | `true` | 是否恢复最佳权重 |

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

### 场景5: 早停功能配置
```bash
# 使用默认早停配置
python dynamic_resolution_trainer.py \
    --config dynamic_config.yaml \
    --input_resolution 32 32 \
    --output_resolution 128 128

# 运行早停功能演示
python demo_early_stopping.py
```

## 早停功能详解

### 基本概念

早停（Early Stopping）是一种防止过拟合的技术，当监控指标在一定epoch内没有改善时，自动停止训练并恢复到最佳模型权重。

### 配置示例

#### 1. 基本早停配置
```yaml
training:
  # 早停配置
  enable_early_stopping: true  # 启用早停
  patience: 10                 # 10个epoch没有改善就停止
  min_delta: 0.001            # 改善阈值
  monitor: "val_loss"         # 监控验证损失
  mode: "min"                 # 损失越小越好
  restore_best_weights: true  # 恢复最佳权重
```

#### 2. 监控训练损失
```yaml
training:
  enable_early_stopping: true
  patience: 5
  min_delta: 0.0001
  monitor: "train_loss"       # 监控训练损失
  mode: "min"
  restore_best_weights: true
```

#### 3. 禁用早停
```yaml
training:
  enable_early_stopping: false  # 禁用早停
  # 其他早停参数会被忽略
```

### 监控指标说明

- `val_loss`: 验证损失（推荐，防止过拟合）
- `train_loss`: 训练损失（用于快速收敛检测）
- `test_loss`: 测试损失（用于最终性能评估）

### 最佳实践

1. **推荐配置**:
   - `monitor: "val_loss"` - 监控验证损失
   - `patience: 10-20` - 根据数据集大小调整
   - `min_delta: 0.0001-0.001` - 根据损失量级调整
   - `restore_best_weights: true` - 始终启用

2. **调试技巧**:
   - 使用 `monitor: "train_loss"` 快速检测模型是否学习
   - 较小的 `patience` 值用于快速原型验证
   - 较大的 `patience` 值用于最终训练

3. **性能优化**:
   - 早停可以显著减少训练时间
   - 自动找到最佳训练轮数
   - 防止过拟合，提高泛化能力

## 损失函数配置详解

### 基本概念
本项目支持多种损失函数配置，包括基础MSE损失和SVD模态损失的组合，可以更好地捕捉流场的主要特征。

### 损失函数类型

#### 1. 基础损失配置
```yaml
loss:
  base_weight: 0.8              # MSE损失权重
  svd_weights: [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
  topk: 10                      # SVD模态数量
  loss_type: "mse_svd"          # 损失类型
  svd_loss_enabled: true        # 启用SVD损失
  normalize_svd_weights: true   # 归一化SVD权重
```

#### 2. 仅MSE损失
```yaml
loss:
  base_weight: 1.0
  svd_weights: [0.0, 0.0, .0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  topk: 10
  svd_loss_enabled: false
```

#### 3. SVD主导配置
```yaml
loss:
  base_weight: 0.3
  svd_weights: [0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]
  topk: 10
  svd_loss_enabled: true
```

### 多损失配置实验

配置文件中的`loss_configs`部分提供了5种预设配置：

1. **基础配置**: 仅使用MSE损失
2. **平衡配置**: MSE + SVD平衡组合
3. **SVD主导**: 以SVD损失为主
4. **前3模态**: 只关注前3个主要模态
5. **高频模态**: 重点关注高频模态

### 参数说明

- **base_weight**: 基础MSE损失的权重，范围0-1
- **svd_weights**: 各SVD模态的权重列表，长度应等于topk
- **topk**: 使用的SVD主模态数量，建议3-10
- **loss_type**: 损失函数类型，支持'mse', 'mse_svd', 'l1', 'huber'
- **svd_loss_enabled**: 是否启用SVD损失计算
- **normalize_svd_weights**: 是否对SVD权重进行归一化

### 使用建议

1. **初学者**: 使用基础配置或平衡配置
2. **流场重建**: 重点关注前几个主模态
3. **细节保持**: 适当增加高频模态权重
4. **快速训练**: 减少topk值，使用较少模态
5. **精细调优**: 根据具体问题调整各模态权重

### 实验流程

```bash
# 使用默认损失配置
python dynamic_resolution_trainer.py

# 使用特定损失配置进行批量实验
# (需要修改代码支持loss_configs遍历)
```

## 验证和测试配置详解

### 验证配置

验证配置控制训练过程中的模型评估：

```yaml
validation:
  enabled: true                   # 是否启用验证
  interval: 1                     # 验证间隔（每N个epoch验证一次）
  metrics: ["mse", "mae", "rmse", "psnr"]  # 验证指标
  save_predictions: false         # 是否保存验证预测结果
  max_samples_to_save: 10         # 最大保存样本数
```

### 测试配置

测试配置控制最终的模型评估：

```yaml
test:
  enabled: true                   # 是否启用测试
  metrics: ["mse", "mae", "rmse", "psnr", "ssim"]  # 测试指标
  save_predictions: true          # 是否保存测试预测结果
  save_visualizations: true       # 是否保存可视化结果
  output_dir: "./results/test"    # 测试结果输出目录
```

## 评估指标配置详解

### 指标分类

评估指标分为三大类，全面评估模型性能：

#### 1. 回归指标
```yaml
regression_metrics:
  mse: true                     # 均方误差
  mae: true                     # 平均绝对误差
  rmse: true                    # 均方根误差
  r2_score: true                # R²决定系数
```

#### 2. 图像质量指标
```yaml
image_metrics:
  psnr: true                    # 峰值信噪比
  ssim: true                    # 结构相似性指数
  lpips: false                  # 感知图像补丁相似性
```

#### 3. 物理指标（针对流体力学）
```yaml
physics_metrics:
  energy_conservation: true     # 能量守恒误差
  mass_conservation: true       # 质量守恒误差
  vorticity_error: true         # 涡度误差
```

### 指标说明

- **MSE/MAE/RMSE**: 基础数值误差指标
- **R²**: 模型解释方差的比例
- **PSNR**: 图像重建质量指标
- **SSIM**: 结构相似性，更符合人眼感知
- **物理指标**: 评估是否满足物理约束

## 数据增强配置详解

### 增强类型

数据增强分为三大类，提高模型泛化能力：

#### 1. 几何变换
```yaml
geometric:
  rotation: false               # 旋转增强
  rotation_range: [-10, 10]     # 旋转角度范围（度）
  flip_horizontal: false        # 水平翻转
  flip_vertical: false          # 垂直翻转
  scale: false                  # 缩放增强
  scale_range: [0.9, 1.1]       # 缩放范围
```

#### 2. 噪声增强
```yaml
noise:
  gaussian_noise: false         # 高斯噪声
  noise_std: 0.01              # 噪声标准差
  salt_pepper: false           # 椒盐噪声
  noise_ratio: 0.01            # 噪声比例
```

#### 3. 物理增强（针对流体力学）
```yaml
physics:
  reynolds_perturbation: false  # 雷诺数扰动
  boundary_condition_noise: false  # 边界条件噪声
  initial_condition_noise: false   # 初始条件噪声
```

### 使用建议

1. **数据充足**: 关闭数据增强，避免过度正则化
2. **数据不足**: 启用几何变换和适度噪声
3. **鲁棒性要求高**: 启用物理增强，模拟真实条件变化
4. **快速训练**: 仅使用轻量级增强（翻转、小幅旋转）

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