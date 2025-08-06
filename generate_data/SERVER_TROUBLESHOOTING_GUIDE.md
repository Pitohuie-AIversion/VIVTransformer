# 服务器训练问题诊断和解决指南

## 🚨 常见问题快速诊断

### 问题1: 服务器任务无响应/卡住

**症状**: 
- 测试脚本显示所有功能正常
- 实际运行训练时服务器没有响应
- 任务提交后长时间无输出

**可能原因**:
1. **路径问题**: 配置文件中使用了Windows路径格式
2. **数据文件不存在**: 数据路径不正确
3. **权限问题**: 没有读取数据文件的权限
4. **环境变量**: CUDA/PyTorch环境配置问题
5. **资源竞争**: 其他任务占用了GPU/内存

**解决方案**:

#### 步骤1: 检查和修复路径问题
```bash
# 运行环境检测脚本
python server_environment_setup.py

# 这将自动:
# 1. 检测系统资源
# 2. 查找正确的数据路径
# 3. 生成适合的配置文件
# 4. 提供运行建议
```

#### 步骤2: 使用Linux适配的配置文件
```bash
# 使用新生成的配置文件
python dynamic_resolution_trainer.py --config dynamic_config_server_auto.yaml

# 或者使用预设的Linux配置
python dynamic_resolution_trainer.py --config dynamic_config_server_linux.yaml
```

#### 步骤3: 快速测试验证
```bash
# 先运行快速测试确保环境正常
python dynamic_resolution_trainer.py --config dynamic_config_server_auto.yaml --epochs 1 --num_samples 10
```

### 问题2: 数据路径错误

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'X:\\...\\2D_DarcyFlow_beta0.1_Train.hdf5'
```

**解决方案**:

1. **查找正确的数据路径**:
```bash
# 查找PDEBench数据文件
find /share -name "2D_DarcyFlow_beta0.1_Train.hdf5" 2>/dev/null
find /data -name "2D_DarcyFlow_beta0.1_Train.hdf5" 2>/dev/null
find /home -name "2D_DarcyFlow_beta0.1_Train.hdf5" 2>/dev/null
```

2. **更新配置文件中的路径**:
```yaml
data:
  path: "/实际的/数据/路径/2D_DarcyFlow_beta0.1_Train.hdf5"
```

3. **验证文件存在**:
```bash
ls -la /path/to/your/data/2D_DarcyFlow_beta0.1_Train.hdf5
```

### 问题3: GPU内存不足

**症状**:
```
CUDA out of memory. Tried to allocate XXX MiB
```

**解决方案**:

1. **减小批次大小**:
```yaml
data:
  batch_size: 32  # 从128减少到32
training:
  batch_size: 32
```

2. **启用梯度累积**:
```yaml
gradient:
  accumulation_steps: 4  # 累积4个批次再更新
```

3. **使用混合精度训练**:
```yaml
amp:
  enabled: true
  opt_level: "O1"
```

### 问题4: 权限问题

**症状**:
```
PermissionError: [Errno 13] Permission denied
```

**解决方案**:
```bash
# 检查文件权限
ls -la /path/to/data/file.hdf5

# 如果需要，修改权限
chmod 644 /path/to/data/file.hdf5

# 或者联系管理员获取访问权限
```

### 问题5: 环境变量问题

**症状**:
- CUDA不可用
- 找不到GPU
- 库版本冲突

**解决方案**:
```bash
# 检查CUDA环境
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# 检查PyTorch安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 如果需要，设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🛠️ 推荐的服务器运行流程

### 1. 环境准备和验证
```bash
# 步骤1: 运行环境检测
python server_environment_setup.py

# 步骤2: 验证生成的配置
cat dynamic_config_server_auto.yaml
```

### 2. 快速测试
```bash
# 步骤3: 运行快速测试
python dynamic_resolution_trainer.py --config dynamic_config_server_auto.yaml --epochs 1 --num_samples 100
```

### 3. 正式训练
```bash
# 步骤4: 后台运行完整训练
nohup python dynamic_resolution_trainer.py --config dynamic_config_server_auto.yaml > training.log 2>&1 &

# 步骤5: 监控训练进度
tail -f training.log

# 步骤6: 检查GPU使用情况
watch -n 1 nvidia-smi
```

## 📊 性能优化建议

### 针对192核CPU + 1TB内存 + 2GPU的服务器

**推荐配置**:
```yaml
dataloader:
  num_workers: 64        # 充分利用CPU核心
  pin_memory: true       # 启用内存固定
  persistent_workers: true  # 持久化工作进程
  prefetch_factor: 16    # 大预取因子

training:
  batch_size: 128        # 大批次大小
  num_samples: 20000     # 大样本数量

device: "cuda"
use_dataparallel: true   # 启用多GPU并行
max_memory_fraction: 0.95  # 高显存使用率
```

**监控指标**:
- CPU使用率应该在80-90%
- 内存使用率应该在60-80%
- GPU使用率应该在90%+
- GPU内存使用率应该在80-95%

## 🔧 调试工具和命令

### 系统资源监控
```bash
# CPU和内存监控
htop

# GPU监控
nvidia-smi -l 1

# 磁盘IO监控
iotop

# 网络监控
iftop
```

### 进程管理
```bash
# 查看Python进程
ps aux | grep python

# 杀死卡住的进程
kill -9 <PID>

# 查看进程资源使用
top -p <PID>
```

### 日志分析
```bash
# 实时查看训练日志
tail -f training.log

# 搜索错误信息
grep -i error training.log
grep -i "out of memory" training.log

# 查看最后的输出
tail -n 100 training.log
```

## 🚀 快速修复脚本

创建一个快速修复脚本 `quick_fix.sh`:

```bash
#!/bin/bash
# 服务器训练快速修复脚本

echo "🔍 检测服务器环境..."
python server_environment_setup.py

echo "🧪 运行快速测试..."
python dynamic_resolution_trainer.py --config dynamic_config_server_auto.yaml --epochs 1 --num_samples 10

if [ $? -eq 0 ]; then
    echo "✅ 环境正常，可以开始训练"
    echo "🚀 启动完整训练..."
    nohup python dynamic_resolution_trainer.py --config dynamic_config_server_auto.yaml > training.log 2>&1 &
    echo "📊 训练已在后台启动，使用 'tail -f training.log' 监控进度"
else
    echo "❌ 环境测试失败，请检查上述错误信息"
fi
```

使用方法:
```bash
chmod +x quick_fix.sh
./quick_fix.sh
```

## 📞 获取帮助

如果以上解决方案都无法解决问题，请提供以下信息:

1. **系统信息**:
```bash
uname -a
python --version
pip list | grep torch
nvidia-smi
```

2. **错误日志**:
```bash
tail -n 50 training.log
```

3. **配置文件**:
```bash
cat dynamic_config_server_auto.yaml
```

4. **数据路径验证**:
```bash
ls -la /path/to/your/data/
```

## 🎯 成功运行的标志

当看到以下输出时，说明训练正在正常进行:

```
2025-XX-XX XX:XX:XX,XXX - INFO - 🚀 开始训练...
2025-XX-XX XX:XX:XX,XXX - INFO - 📊 数据加载完成
2025-XX-XX XX:XX:XX,XXX - INFO - 🔄 Epoch 1/500
2025-XX-XX XX:XX:XX,XXX - INFO - 📈 训练损失: X.XXXX
2025-XX-XX XX:XX:XX,XXX - INFO - 📉 验证损失: X.XXXX
```

同时GPU监控应该显示:
- GPU使用率 > 80%
- GPU内存使用率 > 60%
- 温度在正常范围内 (< 85°C)

---

**记住**: 服务器环境的关键是正确的路径配置和资源优化。使用提供的自动化脚本可以大大简化配置过程！