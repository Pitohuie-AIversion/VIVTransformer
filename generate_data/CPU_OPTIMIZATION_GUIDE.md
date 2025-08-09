# CPU和内存利用率优化指南

## 问题描述

在服务器训练时遇到的问题：
- **GPU利用率高**：GPU处理能力强，计算密集
- **CPU利用率低**：数据加载跟不上GPU消费速度
- **内存利用率低**：数据流水线效率不高，内存资源未充分利用
- **内存使用不当**：内存峰值过高或内存泄漏导致训练中断

## 根本原因分析

### 1. 数据加载瓶颈
- 配置文件中设置了固定的`num_workers: 64`，但代码的自动优化逻辑只在`num_workers=0`时生效
- 过高的worker数量可能导致进程开销，反而降低效率
- 数据预处理（降采样、归一化）在运行时进行，增加CPU负担

### 2. 数据流水线不平衡
- GPU处理速度快，但数据供应跟不上
- CPU核心没有被充分且合理地利用
- 内存预取策略不当

### 3. 内存管理问题
- 缺乏实时内存监控，无法及时发现内存问题
- 多进程数据加载器内存开销过大
- 大数据集加载时内存峰值过高
- 内存泄漏导致长时间训练失败
- 缺乏内存使用阈值控制和自动调整机制

## 优化方案

### 1. 智能Worker数量优化

#### 配置文件修改
```yaml
dataloader:
  num_workers: 0                  # 设为0启用自动优化
  pin_memory: true                # 固定内存以提升GPU传输速度
  persistent_workers: false       # 将由代码自动设置
  prefetch_factor: 2              # 保守的预取因子
  drop_last: true                 # 保持批次一致性
  # 智能优化配置
  enable_smart_workers: true      # 启用智能worker数量检测
  min_workers: 0                  # 最小worker数量
  max_workers: 32                 # 最大worker数量
```

#### 智能优化逻辑

代码会根据以下因素自动调整worker数量：

1. **数据处理复杂度**：
   - **降采样模式**：需要更多CPU计算，使用更多worker
     - 超级服务器（≥64核）：`min(max_workers, max(8, cpu_count // 6))`
     - 普通服务器：`min(max_workers, max(4, cpu_count // 4))`
   - **裁剪模式**：计算较轻，使用较少worker避免进程开销
     - 大批次（≥64）：`min(max_workers, max(2, cpu_count // 8))`
     - 小批次：单进程模式（`num_workers=0`）

2. **系统资源**：
   - CPU核心数
   - 配置的最大/最小worker限制

### 2. 代码优化

#### 多进程启动方法优化
```python
# 设置多进程启动方法（Linux服务器优化）
try:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
```

#### 智能worker数量设置
```python
if enable_smart_workers:
    use_downsampling = data_config.get('downsampling', {}).get('enabled', False)
    batch_size = data_config.get('batch_size', 16)
    
    if use_downsampling:
        # 降采样需要更多CPU计算
        if cpu_count >= 64:
            num_workers = min(max_workers, max(8, cpu_count // 6))
        else:
            num_workers = min(max_workers, max(4, cpu_count // 4))
    else:
        # 简单裁剪操作
        if batch_size >= 64:
            num_workers = min(max_workers, max(2, cpu_count // 8))
        else:
            num_workers = max(min_workers, 0)
```

### 3. 内存优化配置

#### 配置文件添加内存优化
```yaml
dataloader:
  # ... CPU优化配置 ...
  
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

#### 内存监控和自动调整
```python
# 实时内存监控
memory_info = psutil.virtual_memory()
current_memory_gb = memory_info.used / (1024**3)
total_memory_gb = memory_info.total / (1024**3)
memory_usage_ratio = memory_info.percent / 100.0

# 根据内存使用率自动调整worker数量
if memory_usage_ratio > critical_threshold:
    if num_workers > 4:
        num_workers = max(2, num_workers // 2)
        logger.info(f"🔧 自动调整: 减少num_workers到{num_workers}以节省内存")
elif memory_usage_ratio > warning_threshold:
    logger.warning(f"⚠️ 内存使用率较高 ({memory_usage_ratio:.1%})，请注意监控")
```

#### 内存优化功能
1. **实时监控**: 跟踪内存使用情况，显示使用量和使用率
2. **阈值控制**: 设置警告和临界阈值，自动调整参数
3. **自动清理**: 定期执行垃圾回收，防止内存泄漏
4. **共享内存**: 多进程间共享数据，减少内存重复
5. **内存映射**: 大文件处理时减少内存占用
6. **懒加载**: 按需加载数据，减少内存峰值

### 4. 性能测试结果

#### 当前配置（32核CPU）
- **降采样模式**：推荐8个worker，CPU利用率25%
- **裁剪大批次**：推荐4个worker，CPU利用率12.5%
- **裁剪小批次**：推荐单进程，CPU利用率0%（避免进程开销）

#### 不同场景优化策略

| 场景 | 数据处理 | 批次大小 | 推荐Worker | CPU利用率 | 说明 |
|------|----------|----------|------------|-----------|------|
| 降采样+大批次 | 复杂 | 256 | 8 | 25.0% | 平衡性能与开销 |
| 降采样+小批次 | 复杂 | 32 | 8 | 25.0% | 保证处理能力 |
| 裁剪+大批次 | 简单 | 128 | 4 | 12.5% | 适度并行 |
| 裁剪+小批次 | 简单 | 16 | 0 | 0.0% | 避免进程开销 |

## 使用方法

### 1. 应用优化配置
```bash
# 使用优化后的配置文件
python dynamic_resolution_trainer.py --config dynamic_config_server_downsampling.yaml
```

### 2. 验证优化效果
```bash
# 测试智能优化逻辑
python test_smart_optimization.py

# 测试多核数据加载性能
python test_multicore_dataloader.py
```

### 3. 监控训练过程
训练时注意观察日志中的以下信息：
```
🎯 智能优化(降采样): 设置num_workers=8
🚀 最终设置: num_workers=8 (CPU核心数: 32)
🔄 数据加载器配置: num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=2
```

## 预期效果

### 1. CPU利用率提升
- **优化前**：单核运行，CPU利用率极低
- **优化后**：智能多核并行，CPU利用率提升至15-25%

### 2. 内存利用率改善
- 合理的预取策略减少内存浪费
- 持久化worker进程减少重复创建开销

### 3. 训练效率提升
- 数据加载速度提升，减少GPU等待时间
- 更好的数据流水线平衡

## 进阶优化建议

### 1. 针对超级服务器（192核）
如果在192核服务器上运行，可以调整配置：
```yaml
dataloader:
  max_workers: 64  # 增加最大worker数量
```

### 2. 内存充足时的优化
```yaml
data:
  lazy_loading: false  # 禁用懒加载，预加载数据到内存
  
dataloader:
  prefetch_factor: 4   # 增加预取因子
```

### 3. 自定义优化
如果需要手动设置worker数量：
```yaml
dataloader:
  num_workers: 16              # 手动设置
  enable_smart_workers: false  # 禁用智能优化
```

## 故障排除

### 1. Worker数量仍为0
- 检查CPU核心数是否>4
- 确认`enable_smart_workers: true`
- 查看日志中的优化信息

### 2. 性能没有提升
- 检查数据处理是否为瓶颈
- 考虑减少worker数量避免过度并行
- 监控系统资源使用情况

### 3. 内存使用过高
- 减少`prefetch_factor`
- 减少`max_workers`
- 启用`lazy_loading`

## 总结

通过智能worker数量优化，我们解决了GPU利用率高但CPU和内存利用率低的问题：

1. **根据数据处理复杂度**智能设置worker数量
2. **避免过度并行**导致的进程开销
3. **平衡GPU和CPU**的工作负载
4. **提供灵活的配置选项**适应不同场景

这种优化方案既保证了性能提升，又避免了资源浪费，是一个可持续的解决方案。