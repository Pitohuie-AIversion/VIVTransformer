# 服务器多GPU错误快速修复指南

## 🚨 问题描述
服务器上仍然出现多GPU设备错误：
```
RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cuda:1
```

## 🔧 立即修复方案

### 方案1: 自动修复脚本（推荐）

1. **上传修复脚本到服务器**
   ```bash
   # 将 server_fix_multi_gpu.py 上传到项目根目录
   ```

2. **在服务器上运行修复脚本**
   ```bash
   cd /share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench
   python server_fix_multi_gpu.py
   ```

3. **重新运行训练**
   ```bash
   python generate_data/dynamic_resolution_trainer.py --config dynamic_config.yaml
   ```

### 方案2: 手动修复（如果自动脚本失败）

1. **清理Python缓存**
   ```bash
   find . -name "__pycache__" -type d -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```

2. **手动编辑trainer.py文件**
   ```bash
   nano modify_multi_attention/training/trainer.py
   ```
   
   找到第38行左右的：
   ```python
   model.to(device)
   ```
   
   替换为：
   ```python
   # 只有在模型不是DataParallel时才移动到device
   # DataParallel模型已经在主程序中正确设置了设备
   if not isinstance(model, torch.nn.DataParallel):
       model.to(device)
   ```

3. **验证修复**
   ```bash
   grep -n "isinstance.*DataParallel" modify_multi_attention/training/trainer.py
   ```
   应该看到包含检查代码的行

4. **重新运行训练**
   ```bash
   python generate_data/dynamic_resolution_trainer.py --config dynamic_config.yaml
   ```

## 🔍 验证修复成功

修复成功后，训练日志应该显示：
```
🚀 启用多GPU训练: 检测到 2 张GPU
   使用的GPU设备: [0, 1]
   GPU 0: NVIDIA L40 (44.3GB)
   GPU 1: NVIDIA L40 (44.3GB)
```

并且训练正常进行，不再出现设备错误。

## 🚨 如果问题仍然存在

### 检查环境一致性
```bash
# 检查Python环境
which python
python --version

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 检查CUDA版本
nvcc --version
```

### 强制重新导入模块
```bash
# 重启Python进程
pkill -f python

# 或者在Python中强制重新加载
python -c "
import sys
if 'modify_multi_attention.training.trainer' in sys.modules:
    del sys.modules['modify_multi_attention.training.trainer']
print('模块缓存已清理')
"
```

### 检查文件权限
```bash
# 确保文件可写
chmod 644 modify_multi_attention/training/trainer.py
ls -la modify_multi_attention/training/trainer.py
```

## 📋 完整的修复验证流程

1. **运行自动修复脚本**
   ```bash
   python server_fix_multi_gpu.py
   ```

2. **验证修复代码存在**
   ```bash
   grep -A 3 -B 1 "isinstance.*DataParallel" modify_multi_attention/training/trainer.py
   ```

3. **清理缓存并重启**
   ```bash
   find . -name "__pycache__" -type d -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```

4. **测试训练**
   ```bash
   python generate_data/dynamic_resolution_trainer.py --config dynamic_config.yaml
   ```

5. **确认多GPU正常工作**
   - 查看日志中的GPU检测信息
   - 确认训练正常进行
   - 监控GPU使用情况：`nvidia-smi`

## 🎯 预期结果

修复成功后，您应该看到：
- ✅ 多GPU训练正常启动
- ✅ 无设备冲突错误
- ✅ 训练损失正常下降
- ✅ GPU利用率均衡

## 📞 技术支持

如果以上方案都无法解决问题，请提供：
1. 服务器Python环境信息
2. PyTorch和CUDA版本
3. 完整的错误堆栈跟踪
4. trainer.py文件的相关代码段