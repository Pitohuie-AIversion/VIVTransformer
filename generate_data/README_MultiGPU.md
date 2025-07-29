# 多GPU训练功能说明

本文档介绍如何在 `dynamic_resolution_trainer.py` 中使用多GPU训练功能。

## 功能概述

新增的多GPU训练功能支持：
- **DataParallel模式**: 适用于单机多卡训练
- **自动GPU检测**: 自动检测可用的GPU数量
- **内存监控**: 显示每张GPU的内存使用情况
- **灵活配置**: 支持命令行参数和配置文件两种方式

## 使用方法

### 1. 命令行方式

```bash
# 启用多GPU训练
python dynamic_resolution_trainer.py --use_dataparallel

# 结合其他参数使用
python dynamic_resolution_trainer.py \
    --input_resolution 32 32 \
    --output_resolution 128 128 \
    --use_dataparallel \
    --batch_size 32 \
    --epochs 50
```

### 2. 配置文件方式

在 `dynamic_config.yaml` 中设置：

```yaml
# 启用多GPU训练
use_dataparallel: true

# 推荐的多GPU训练配置
data:
  batch_size: 32  # 增加批次大小以充分利用多GPU

dataloader:
  num_workers: 8      # 增加数据加载进程数
  pin_memory: true    # 启用内存固定
  persistent_workers: true  # 保持工作进程
  prefetch_factor: 4  # 增加预取因子

# 混合精度训练（可选，提升性能）
mixed_precision:
  enabled: true
  opt_level: "O1"
```

然后运行：

```bash
python dynamic_resolution_trainer.py --config dynamic_config.yaml
```

## 性能优化建议

### 1. 批次大小调整

- **单GPU**: batch_size = 16
- **双GPU**: batch_size = 32 (16 × 2)
- **四GPU**: batch_size = 64 (16 × 4)

### 2. 数据加载优化

```yaml
dataloader:
  num_workers: 8          # 设置为 CPU核心数
  pin_memory: true        # 加速GPU数据传输
  persistent_workers: true # 避免重复创建进程
  prefetch_factor: 4      # 预取更多批次
```

### 3. 内存管理

```yaml
device:
  max_memory_fraction: 0.8  # 限制每张GPU的内存使用
```

### 4. 混合精度训练

```yaml
mixed_precision:
  enabled: true
  opt_level: "O1"  # 或 "O2" 以获得更高性能
```

## 系统要求

### 硬件要求
- 2张或更多NVIDIA GPU
- 每张GPU至少4GB显存
- 足够的系统内存（推荐32GB+）

### 软件要求
- PyTorch >= 1.8.0 (支持CUDA)
- CUDA >= 10.2
- 正确安装的GPU驱动

## 测试多GPU功能

运行测试脚本检查多GPU环境：

```bash
python test_multi_gpu.py
```

测试脚本会检查：
- GPU可用性和数量
- DataParallel功能
- GPU内存使用情况

## 常见问题

### Q1: 为什么多GPU训练没有启用？

**可能原因：**
- 系统只有1张GPU或没有GPU
- CUDA不可用
- 没有设置 `use_dataparallel: true`

**解决方法：**
```bash
# 检查GPU状态
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')"

# 运行测试脚本
python test_multi_gpu.py
```

### Q2: 多GPU训练速度没有明显提升？

**可能原因：**
- 批次大小太小
- 数据加载成为瓶颈
- 模型太小，通信开销大

**解决方法：**
- 增加批次大小（建议按GPU数量倍数增加）
- 优化数据加载器配置
- 启用混合精度训练

### Q3: 出现CUDA内存不足错误？

**解决方法：**
```yaml
# 减少批次大小
data:
  batch_size: 16  # 从32减少到16

# 限制内存使用
device:
  max_memory_fraction: 0.7  # 从0.8减少到0.7

# 启用混合精度
mixed_precision:
  enabled: true
```

## 性能监控

训练过程中会显示以下信息：

```
🚀 启用多GPU训练: 检测到 2 张GPU
   使用的GPU设备: [0, 1]
   GPU 0: NVIDIA GeForce RTX 3080 (10.0GB)
   GPU 1: NVIDIA GeForce RTX 3080 (10.0GB)
```

## 注意事项

1. **DataParallel vs DistributedDataParallel**:
   - 当前实现使用DataParallel，适合单机多卡
   - 如需多机训练，建议使用DistributedDataParallel

2. **批次大小**:
   - DataParallel会自动将批次分割到各GPU
   - 总批次大小 = 配置的batch_size
   - 每张GPU的实际批次 = batch_size / gpu_count

3. **模型保存**:
   - DataParallel模型保存时会包含`module.`前缀
   - 加载时需要注意模型结构匹配

4. **调试建议**:
   - 先在单GPU上验证模型正确性
   - 再启用多GPU训练
   - 使用测试脚本验证多GPU环境

## 示例配置

完整的多GPU训练配置示例：

```yaml
# dynamic_config.yaml
data:
  input_resolution: [32, 32]
  output_resolution: [128, 128]
  batch_size: 32  # 双GPU时每张GPU处理16个样本
  num_samples: 1000

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4

model:
  num_layers: 6
  d_model: 512
  num_heads: 8
  attention_type: 'sge'

# 多GPU配置
use_dataparallel: true

# 数据加载优化
dataloader:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

# 混合精度训练
mixed_precision:
  enabled: true
  opt_level: "O1"

device: 'auto'
seed: 42
```

运行命令：

```bash
python dynamic_resolution_trainer.py --config dynamic_config.yaml
```