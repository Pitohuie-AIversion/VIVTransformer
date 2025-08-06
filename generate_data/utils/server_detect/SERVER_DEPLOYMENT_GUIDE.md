# 🚀 服务器部署指南

## 📋 概述

本指南帮助您在Linux服务器上正确配置和运行VIVTransformer训练任务。

## 🔧 服务器环境信息

根据您的服务器环境检测结果：
- **CPU**: 192核心
- **内存**: 1007.1GB (883.6GB可用)
- **GPU**: 2x NVIDIA L40 (每个44.3GB)
- **平台**: Linux
- **PyTorch**: 2.7.0+cu118
- **CUDA**: 11.8

## 📁 数据路径配置

您的PDEBench数据位于：
```
/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codex/pdebench_extended/PDEBench/pdebench/data_download/2D_DarcyFlow_beta10.0_Train.hdf5
```

## 🚀 快速开始

### 1. 上传文件到服务器

将以下文件上传到服务器：
```bash
# 主要文件
dynamic_resolution_trainer.py
dynamic_config_server_corrected.yaml

# 模型和工具文件
modify_multi_attention/
utils/
```

### 2. 验证环境

```bash
# 检查Python环境
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# 检查数据文件
ls -la "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codex/pdebench_extended/PDEBench/pdebench/data_download/2D_DarcyFlow_beta10.0_Train.hdf5"
```

### 3. 运行训练

#### 快速测试 (推荐先运行)
```bash
# 1轮训练，100样本，验证配置
python dynamic_resolution_trainer.py --config dynamic_config_server_corrected.yaml --epochs 1 --num_samples 100
```

#### 完整训练
```bash
# 后台运行完整训练
nohup python dynamic_resolution_trainer.py --config dynamic_config_server_corrected.yaml > training.log 2>&1 &
```

#### 监控训练
```bash
# 查看训练日志
tail -f training.log

# 查看GPU使用情况
nvidia-smi

# 查看进程
ps aux | grep python
```

## ⚙️ 配置文件说明

`dynamic_config_server_corrected.yaml` 已针对您的服务器环境优化：

### 数据配置
- **数据路径**: 使用您提供的正确路径
- **输入分辨率**: 32x32
- **输出分辨率**: 128x128
- **样本数量**: 20,000 (可根据需要调整)
- **批次大小**: 128 (利用GPU大内存)

### 性能优化
- **工作进程**: 48 (利用192核CPU)
- **多GPU训练**: 启用
- **混合精度**: 启用
- **内存固定**: 启用

### 训练参数
- **训练轮数**: 500
- **学习率**: 0.0005
- **早停**: 50轮无改善停止

## 🔧 故障排除

### 常见问题

#### 1. 数据文件未找到
```bash
# 检查文件是否存在
ls -la "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codex/pdebench_extended/PDEBench/pdebench/data_download/"

# 检查权限
ls -la "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codex/pdebench_extended/PDEBench/pdebench/data_download/2D_DarcyFlow_beta10.0_Train.hdf5"
```

#### 2. GPU内存不足
```bash
# 减小批次大小
# 在配置文件中修改 batch_size: 64 或 32

# 或使用单GPU
# 在配置文件中设置 use_data_parallel: false
```

#### 3. 权限问题
```bash
# 检查当前用户
whoami

# 检查文件权限
ls -la dynamic_config_server_corrected.yaml

# 如需要，修改权限
chmod 644 dynamic_config_server_corrected.yaml
```

### 性能调优

#### 根据实际情况调整配置

1. **内存充足时**：
   - 增加 `batch_size` 到 256 或更高
   - 增加 `num_samples` 到 50,000+

2. **GPU内存不足时**：
   - 减少 `batch_size` 到 64 或 32
   - 禁用 `use_data_parallel`

3. **训练速度优化**：
   - 调整 `num_workers` (建议 CPU核心数的1/4)
   - 启用 `mixed_precision`
   - 调整 `prefetch_factor`

## 📊 监控和日志

### 训练进度监控
```bash
# 实时查看训练日志
tail -f training.log

# 查看最近的错误
grep -i error training.log | tail -10

# 查看损失变化
grep -i "loss" training.log | tail -20
```

### 系统资源监控
```bash
# GPU使用情况
nvidia-smi -l 1

# CPU和内存使用
htop

# 磁盘使用
df -h
```

## 🎯 预期结果

训练成功后，您将在 `./results/` 目录下看到：
- `models/`: 训练好的模型文件
- `logs/`: 详细训练日志
- `loss_logs/`: 损失变化记录
- `final_config.yaml`: 最终使用的配置
- `normalization_info.json`: 数据归一化信息

## 📞 技术支持

如遇到问题，请提供：
1. 完整的错误信息
2. 训练日志 (`training.log`)
3. 系统资源使用情况 (`nvidia-smi`, `htop`)
4. 配置文件内容

---

**祝您训练顺利！** 🎉