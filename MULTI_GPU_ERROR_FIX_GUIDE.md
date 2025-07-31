# 多GPU训练设备错误修复指南

## 错误描述

在多GPU训练时遇到以下错误：
```
RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cuda:1
```

## 错误原因

这是一个典型的多GPU训练中的设备不匹配问题。具体原因是：

1. **DataParallel设备冲突**：在 `dynamic_resolution_trainer.py` 中，模型被正确包装为 `DataParallel`
2. **重复设备设置**：在 `trainer.py` 中，模型被重新移动到单个设备，破坏了 `DataParallel` 的多GPU设置
3. **设备分布不一致**：导致模型参数分布在不同GPU上，引发运行时错误

## 快速修复方法

### 方法1：使用自动修复脚本（推荐）

```bash
# 在项目根目录下运行
python fix_multi_gpu_error.py
```

### 方法2：手动修复

编辑 `modify_multi_attention/training/trainer.py` 文件，找到第32行左右的代码：

**修复前：**
```python
model.to(device)
```

**修复后：**
```python
# 只有在模型不是DataParallel时才移动到device
# DataParallel模型已经在主程序中正确设置了设备
if not isinstance(model, torch.nn.DataParallel):
    model.to(device)
```

## 修复验证

修复后，重新运行多GPU训练：

```bash
python dynamic_resolution_trainer.py --use_dataparallel --config dynamic_config.yaml
```

应该看到类似输出：
```
🚀 启用多GPU训练: 检测到 2 张GPU
   使用的GPU设备: [0, 1]
   GPU 0: NVIDIA L40 (44.3GB)
   GPU 1: NVIDIA L40 (44.3GB)
```

## 多GPU配置参数说明

### 核心参数

在 `dynamic_config.yaml` 中确保以下设置：

```yaml
# 设备配置
device: "auto"                    # 自动检测设备
use_dataparallel: true           # 启用多GPU训练
max_memory_fraction: 0.8         # GPU显存使用限制

# 数据加载优化（多GPU推荐设置）
dataloader:
  num_workers: 4                 # 数据加载进程数
  pin_memory: true               # 启用内存锁定
  persistent_workers: true       # 保持工作进程
  prefetch_factor: 2             # 预取因子

# 批次大小（多GPU时可适当增大）
data:
  batch_size: 4000               # 根据GPU显存调整
```

### 性能优化参数

```yaml
# 混合精度训练（可选）
mixed_precision:
  enabled: true
  loss_scale: "dynamic"

# 分布式训练配置（高级用法）
distributed:
  enabled: false                # 单机多卡使用DataParallel
  backend: "nccl"               # 通信后端
```

## 常见问题排查

### 1. 只检测到1张GPU

**现象：**
```
⚠️ 只检测到 1 张GPU，无法启用多GPU训练
```

**解决方案：**
- 检查 `nvidia-smi` 确认GPU可见性
- 设置 `CUDA_VISIBLE_DEVICES` 环境变量
- 确认PyTorch能检测到多GPU：`torch.cuda.device_count()`

### 2. CUDA内存不足

**现象：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
- 减小 `batch_size`
- 调整 `max_memory_fraction`
- 启用混合精度训练

### 3. 数据加载速度慢

**解决方案：**
- 增加 `num_workers`
- 启用 `pin_memory`
- 使用 `persistent_workers`

## 性能监控

### GPU使用率监控

```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 或使用gpustat
gpustat -i 1
```

### 训练日志检查

查看训练日志确认多GPU正常工作：
```
2025-07-31 22:57:19,403 - __main__ - INFO - 🚀 启用多GPU训练: 检测到 2 张GPU
2025-07-31 22:57:19,404 - __main__ - INFO -    使用的GPU设备: [0, 1]
```

## 最佳实践

1. **批次大小调整**：多GPU时可以成比例增加批次大小
2. **学习率调整**：可能需要相应调整学习率
3. **显存管理**：合理设置 `max_memory_fraction`
4. **数据并行**：确保数据加载不成为瓶颈
5. **模型同步**：DataParallel会自动处理梯度同步

## 技术细节

### DataParallel工作原理

1. **模型复制**：在每个GPU上复制模型
2. **数据分发**：将batch数据分发到各GPU
3. **前向传播**：各GPU并行计算
4. **梯度收集**：收集各GPU梯度到主GPU
5. **参数更新**：在主GPU上更新参数并广播

### 修复原理

- **问题**：`trainer.py` 中的 `model.to(device)` 破坏了DataParallel的设备设置
- **解决**：检查模型类型，只对非DataParallel模型执行设备移动
- **效果**：保持DataParallel模型的正确设备分布

## 联系支持

如果修复后仍有问题，请提供：
1. 完整的错误日志
2. GPU配置信息（`nvidia-smi`输出）
3. PyTorch版本和CUDA版本
4. 配置文件内容