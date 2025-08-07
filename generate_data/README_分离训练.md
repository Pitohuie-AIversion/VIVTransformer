# 分离式训练方案使用说明

## 概述

为了解决服务器CPU资源竞争和锁定问题，我们将原来的训练过程分为两个独立的步骤：

1. **数据预处理阶段** (`data_preprocessor.py`) - 一次性处理所有数据
2. **模型训练阶段** (`simple_trainer.py`) - 专注于模型训练

这种分离方案的优势：
- ✅ 避免训练过程中的CPU密集型数据处理
- ✅ 减少内存占用和资源竞争
- ✅ 提高训练稳定性
- ✅ 支持多次训练实验而无需重复数据处理
- ✅ 更好的资源利用率

## 使用步骤

### 步骤1: 数据预处理

首先运行数据预处理脚本，将原始数据转换为训练就绪的格式：

```bash
# 基本用法
python data_preprocessor.py --data_path /path/to/your/data.h5 --output_dir ./preprocessed_data

# 指定配置文件和样本数量
python data_preprocessor.py \
    --config dynamic_config_server_optimized_final.yaml \
    --data_path /path/to/your/data.h5 \
    --output_dir ./preprocessed_data \
    --num_samples 1000
```

**参数说明：**
- `--config`: 配置文件路径（默认：`dynamic_config_server_optimized_final.yaml`）
- `--data_path`: 原始数据文件路径（必需）
- `--output_dir`: 预处理数据输出目录（默认：`./preprocessed_data`）
- `--num_samples`: 处理的样本数量（可选，覆盖配置文件设置）

**输出文件：**
```
preprocessed_data/
├── train_data.h5           # 训练集数据
├── valid_data.h5           # 验证集数据
├── test_data.h5            # 测试集数据
├── normalization_info.json # 归一化信息
└── preprocessing_config.yaml # 预处理配置
```

### 步骤2: 模型训练

使用预处理好的数据进行训练：

```bash
# 基本训练
python simple_trainer.py --data_dir ./preprocessed_data

# 指定配置文件
python simple_trainer.py \
    --config dynamic_config_server_optimized_final.yaml \
    --data_dir ./preprocessed_data

# 从检查点恢复训练
python simple_trainer.py \
    --config dynamic_config_server_optimized_final.yaml \
    --data_dir ./preprocessed_data \
    --resume checkpoint_epoch_50.pth
```

**参数说明：**
- `--config`: 配置文件路径（默认：`dynamic_config_server_optimized_final.yaml`）
- `--data_dir`: 预处理数据目录（默认：`./preprocessed_data`）
- `--resume`: 恢复训练的检查点路径（可选）

## Linux服务器运行命令

### 方案1: 分步执行（推荐）

```bash
# 1. 数据预处理（一次性执行）
nohup python data_preprocessor.py \
    --data_path /path/to/your/data.h5 \
    --output_dir ./preprocessed_data \
    --num_samples 2000 > preprocess.log 2>&1 &

# 等待预处理完成后，查看日志
tail -f preprocess.log

# 2. 模型训练
nohup python simple_trainer.py \
    --data_dir ./preprocessed_data > training.log 2>&1 &

# 监控训练日志
tail -f training.log
```

### 方案2: 使用screen会话（推荐）

```bash
# 创建预处理会话
screen -S preprocess
python data_preprocessor.py --data_path /path/to/your/data.h5 --output_dir ./preprocessed_data
# Ctrl+A+D 分离会话

# 预处理完成后，创建训练会话
screen -S training
python simple_trainer.py --data_dir ./preprocessed_data
# Ctrl+A+D 分离会话

# 重新连接会话
screen -r preprocess  # 查看预处理进度
screen -r training    # 查看训练进度
```

### 方案3: PyCharm Terminal中运行

在PyCharm的Terminal中：

```bash
# 1. 预处理数据
python generate_data/data_preprocessor.py \
    --data_path /path/to/your/data.h5 \
    --output_dir ./preprocessed_data

# 2. 训练模型
python generate_data/simple_trainer.py \
    --data_dir ./preprocessed_data
```

## 资源监控

### GPU监控
```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 系统资源监控
```bash
# CPU和内存使用情况
htop

# 或者
top

# 磁盘使用情况
df -h

# 进程监控
ps aux | grep python
```

## 配置优化建议

### 针对不同服务器配置的优化

**低内存服务器（<16GB）：**
```yaml
dataloader:
  batch_size: 2
  num_workers: 1
  pin_memory: false
  prefetch_factor: 1

model:
  d_model: 128
  num_layers: 2
  dim_feedforward: 512
```

**中等配置服务器（16-32GB）：**
```yaml
dataloader:
  batch_size: 4
  num_workers: 2
  pin_memory: false
  prefetch_factor: 2

model:
  d_model: 256
  num_layers: 4
  dim_feedforward: 1024
```

**高配置服务器（>32GB）：**
```yaml
dataloader:
  batch_size: 8
  num_workers: 4
  pin_memory: true
  prefetch_factor: 4

model:
  d_model: 512
  num_layers: 6
  dim_feedforward: 2048
```

## 故障排除

### 常见问题

**1. 预处理数据文件不存在**
```
错误: 预处理数据目录不存在
解决: 确保先运行 data_preprocessor.py
```

**2. GPU内存不足**
```
错误: CUDA out of memory
解决: 减小 batch_size 或模型参数
```

**3. CPU占用过高**
```
错误: 系统响应缓慢
解决: 减少 num_workers 或使用预处理数据
```

**4. 磁盘空间不足**
```
错误: No space left on device
解决: 清理临时文件或增加磁盘空间
```

### 性能优化技巧

1. **预处理阶段优化：**
   - 使用SSD存储预处理数据
   - 适当调整样本数量
   - 启用数据压缩

2. **训练阶段优化：**
   - 使用混合精度训练
   - 适当的批次大小
   - 梯度累积

3. **内存管理：**
   - 定期清理GPU缓存
   - 关闭不必要的pin_memory
   - 减少数据加载器的worker数量

## 预期性能

### 数据预处理阶段
- **时间**: 10-30分钟（取决于数据量）
- **内存**: 2-8GB
- **CPU**: 中等使用率
- **磁盘**: 需要足够空间存储预处理数据

### 训练阶段
- **GPU内存**: 2-6GB（取决于配置）
- **训练速度**: 比原方案提升20-40%
- **CPU使用**: 显著降低
- **稳定性**: 大幅提升

## 总结

通过将数据处理和训练分离，我们可以：

1. **解决CPU锁定问题** - 训练时不再进行数据处理
2. **提高训练效率** - 专注于模型优化
3. **增强稳定性** - 减少资源竞争
4. **便于实验** - 可以多次训练而无需重复数据处理
5. **更好的资源管理** - 分阶段使用不同类型的资源

这种方案特别适合在资源受限的服务器环境中进行深度学习训练。