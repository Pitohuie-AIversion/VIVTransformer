# 内存优化配置指南

## 概述

本指南介绍了如何在动态分辨率训练器中配置和使用内存优化功能，以提高训练效率并避免内存不足问题。

## 问题背景

在深度学习训练过程中，常见的内存相关问题包括：
- 内存使用率过高导致系统卡顿
- 内存不足导致训练中断
- 内存泄漏导致长时间训练失败
- 大数据集加载时内存峰值过高
- 多进程数据加载器内存开销过大

## 解决方案

### 1. 配置文件设置

在 `dynamic_config_server_downsampling.yaml` 中添加内存优化配置：

```yaml
dataloader:
  # ... 其他配置 ...
  
  # 内存优化配置
  memory_optimization:
    enable_memory_monitoring: true    # 启用内存监控
    max_memory_usage_gb: 64          # 最大内存使用限制（GB）
    memory_threshold_warning: 0.8    # 内存使用警告阈值（80%）
    memory_threshold_critical: 0.9   # 内存使用临界阈值（90%）
    enable_memory_cleanup: true      # 启用自动内存清理
    cleanup_interval: 100            # 内存清理间隔（每N个batch）
    enable_shared_memory: true       # 启用共享内存（多进程间共享数据）
    shared_memory_size_mb: 1024      # 共享内存大小（MB）
    enable_memory_mapping: true      # 启用内存映射（大文件处理）
    memory_cache_size_mb: 2048       # 内存缓存大小（MB）
    enable_lazy_loading: true        # 启用懒加载（按需加载数据）
    preload_ratio: 0.1               # 预加载比例（10%的数据预加载到内存）
```

### 2. 内存监控功能

#### 实时内存监控
- **功能**: 实时跟踪系统内存使用情况
- **输出**: 显示当前内存使用量、总内存、使用率
- **示例输出**: `💾 内存监控: 当前使用 19.55GB / 39.75GB (49.1%)`

#### 阈值检查
- **警告阈值**: 内存使用率超过80%时发出警告
- **临界阈值**: 内存使用率超过90%时自动调整参数
- **自动调整**: 减少 `num_workers` 数量以节省内存

### 3. 自动内存优化

#### Worker数量自动调整
```python
# 根据内存使用率自动调整worker数量
if memory_usage_ratio > critical_threshold:
    # 内存使用率过高，减少worker数量
    if num_workers > 4:
        num_workers = max(2, num_workers // 2)
        logger.info(f"🔧 自动调整: 减少num_workers到{num_workers}以节省内存")
elif memory_usage_ratio > warning_threshold:
    # 内存使用率较高，发出警告
    logger.warning(f"⚠️ 内存使用率较高 ({memory_usage_ratio:.1%})，请注意监控")
```

#### 内存清理
- **自动垃圾回收**: 定期执行 `gc.collect()` 清理无用对象
- **清理间隔**: 可配置清理频率（默认每100个batch）
- **效果**: 释放不必要的内存占用

### 4. 高级内存优化

#### 共享内存
- **用途**: 多进程间共享数据，减少内存重复
- **配置**: `shared_memory_size_mb` 设置共享内存大小
- **适用场景**: 多worker数据加载

#### 内存映射
- **用途**: 大文件处理时减少内存占用
- **原理**: 将文件映射到虚拟内存，按需加载
- **适用场景**: 大型数据集处理

#### 懒加载
- **用途**: 按需加载数据，减少内存峰值
- **配置**: `preload_ratio` 设置预加载比例
- **效果**: 降低初始内存使用量

## 使用方法

### 1. 启用内存优化

```bash
# 运行训练时自动启用内存优化
python dynamic_resolution_trainer.py --config dynamic_config_server_downsampling.yaml
```

### 2. 测试内存优化功能

```bash
# 运行内存优化测试
python test_memory_optimization.py
```

### 3. 监控训练过程

训练过程中会显示内存监控信息：
```
💾 内存监控: 当前使用 19.55GB / 39.75GB (49.1%)
🔗 启用共享内存: 1024MB
🗺️ 启用内存映射以优化大文件处理
💾 内存缓存大小: 2048MB
⏳ 启用懒加载，预加载比例: 10.0%
```

## 配置建议

### 根据系统配置调整

#### 小内存系统（< 32GB）
```yaml
memory_optimization:
  max_memory_usage_gb: 24
  memory_threshold_warning: 0.7
  memory_threshold_critical: 0.85
  shared_memory_size_mb: 512
  memory_cache_size_mb: 1024
  preload_ratio: 0.05
```

#### 中等内存系统（32-64GB）
```yaml
memory_optimization:
  max_memory_usage_gb: 48
  memory_threshold_warning: 0.8
  memory_threshold_critical: 0.9
  shared_memory_size_mb: 1024
  memory_cache_size_mb: 2048
  preload_ratio: 0.1
```

#### 大内存系统（> 64GB）
```yaml
memory_optimization:
  max_memory_usage_gb: 96
  memory_threshold_warning: 0.85
  memory_threshold_critical: 0.95
  shared_memory_size_mb: 2048
  memory_cache_size_mb: 4096
  preload_ratio: 0.15
```

### 根据数据集大小调整

#### 小数据集（< 1GB）
- 禁用懒加载：`enable_lazy_loading: false`
- 增加预加载比例：`preload_ratio: 0.5`
- 减少缓存大小：`memory_cache_size_mb: 512`

#### 大数据集（> 10GB）
- 启用懒加载：`enable_lazy_loading: true`
- 减少预加载比例：`preload_ratio: 0.05`
- 增加缓存大小：`memory_cache_size_mb: 4096`
- 启用内存映射：`enable_memory_mapping: true`

## 性能测试结果

### 测试环境
- **系统**: Windows 11
- **CPU**: 32核心
- **内存**: 40GB
- **配置**: `dynamic_config_server_downsampling.yaml`

### 测试结果

#### 内存监控测试
```
📊 初始内存状态:
   总内存: 39.75GB
   已使用: 19.55GB
   可用内存: 20.20GB
   使用率: 49.1%
   ✅ 内存使用率正常 (49.1%)
```

#### 内存压力测试
```
🔥 内存压力测试
测试前内存使用: 19.55GB (49.2%)
创建张量 1: 内存使用 19.65GB (49.4%)
创建张量 2: 内存使用 19.75GB (49.6%)
创建张量 3: 内存使用 19.81GB (49.8%)
创建张量 4: 内存使用 19.90GB (50.0%)
创建张量 5: 内存使用 19.98GB (50.2%)
清理后内存使用: 19.98GB (50.2%)
```

#### Worker调整测试
```
📊 场景: 高内存使用 (85.0%)
   建议: 适度减少worker数量 16 → 12
   CPU核心数: 32
   Worker利用率: 37.5%

📊 场景: 极高内存使用 (95.0%)
   建议: 减少worker数量 16 → 8
   CPU核心数: 32
   Worker利用率: 25.0%
```

## 故障排除

### 常见问题

#### 1. 内存监控不工作
**问题**: 没有显示内存监控信息
**解决**: 
- 检查配置文件中 `enable_memory_monitoring: true`
- 确保安装了 `psutil` 库：`pip install psutil`

#### 2. 内存使用率仍然过高
**问题**: 启用优化后内存使用率仍然很高
**解决**:
- 降低 `batch_size`
- 减少 `max_workers`
- 降低 `memory_cache_size_mb`
- 减少 `preload_ratio`

#### 3. 训练速度变慢
**问题**: 启用内存优化后训练速度下降
**解决**:
- 适当增加 `shared_memory_size_mb`
- 启用 `enable_memory_mapping`
- 调整 `cleanup_interval` 到更大值

#### 4. 内存清理过于频繁
**问题**: 内存清理影响训练性能
**解决**:
- 增加 `cleanup_interval` 值
- 设置 `enable_memory_cleanup: false` 禁用自动清理

### 调试命令

```bash
# 查看系统内存使用
python -c "import psutil; print(f'内存使用: {psutil.virtual_memory().percent:.1f}%')"

# 测试内存优化功能
python test_memory_optimization.py

# 监控训练过程内存使用
python dynamic_resolution_trainer.py --config dynamic_config_server_downsampling.yaml --verbose
```

## 最佳实践

### 1. 渐进式优化
- 从默认配置开始
- 根据实际内存使用情况逐步调整
- 监控训练性能变化

### 2. 定期监控
- 使用内存监控功能跟踪使用情况
- 记录不同配置下的性能表现
- 建立最优配置档案

### 3. 环境适配
- 根据硬件配置调整参数
- 考虑其他程序的内存占用
- 预留足够的系统内存

### 4. 测试验证
- 在正式训练前进行内存压力测试
- 验证长时间训练的内存稳定性
- 测试不同数据集大小的适应性

## 总结

内存优化功能通过以下方式提升训练效率：

1. **实时监控**: 跟踪内存使用情况，及时发现问题
2. **自动调整**: 根据内存压力自动优化参数
3. **智能清理**: 定期清理无用内存，防止泄漏
4. **高效共享**: 多进程间共享数据，减少重复
5. **按需加载**: 懒加载策略减少内存峰值

通过合理配置这些功能，可以显著改善训练过程中的内存使用效率，避免内存不足问题，提升整体训练性能。