2025-08-08 21:41:19,116 - __main__ - INFO - 📝 日志系统已初始化，日志文件: ./results/logs/server_downsampling_training.log
2025-08-08 21:41:19,117 - __main__ - INFO - ℹ️ SVD损失已禁用，跳过SVD权重验证
2025-08-08 21:41:19,118 - __main__ - INFO - ✅ 配置验证通过
2025-08-08 21:41:19,118 - __main__ - INFO - === 配置信息 ===
2025-08-08 21:41:19,118 - __main__ - INFO - 输入分辨率: [32, 32] -> 1024维
2025-08-08 21:41:19,118 - __main__ - INFO - 输出分辨率: [128, 128] -> 16384维
2025-08-08 21:41:19,118 - __main__ - INFO - 样本数量: 5000
2025-08-08 21:41:19,118 - __main__ - INFO - 批次大小: 64
2025-08-08 21:41:19,118 - __main__ - INFO - 训练轮数: 550
2025-08-08 21:41:19,118 - __main__ - INFO - 注意力机制: sge
2025-08-08 21:41:19,121 - __main__ - INFO - 🎲 设置随机种子: 42
2025-08-08 21:41:19,121 - __main__ - INFO - 🚀 CPU优化设置: 检测到192个CPU核心，设置190个线程用于计算
2025-08-08 21:41:19,232 - __main__ - INFO - 🔍 CPU状态 初始状态:
2025-08-08 21:41:19,232 - __main__ - INFO -   内存使用: 117.70GB/1007.06GB (12.4%)
2025-08-08 21:41:19,232 - __main__ - INFO -   CPU使用率: 1.1%
2025-08-08 21:41:19,298 - __main__ - INFO - 🔍 GPU内存状态 初始状态:
2025-08-08 21:41:19,298 - __main__ - INFO -   已分配: 0.00 GB
2025-08-08 21:41:19,298 - __main__ - INFO -   已保留: 0.00 GB
2025-08-08 21:41:19,298 - __main__ - INFO -   峰值分配: 0.00 GB
2025-08-08 21:41:19,298 - __main__ - INFO - 🖥️ 使用设备: cuda
2025-08-08 21:41:19,701 - __main__ - INFO - 💾 设置GPU显存限制: 90.0% (应用于 2 张GPU)
2025-08-08 21:41:19,702 - __main__ - INFO - 📁 模型保存目录: ./results/models
2025-08-08 21:41:19,702 - __main__ - INFO - === 验证配置 ===
2025-08-08 21:41:19,702 - __main__ - INFO - ✅ 配置验证通过
2025-08-08 21:41:19,702 - __main__ - INFO - 配置验证详情: warnings=0, errors=0
2025-08-08 21:41:19,702 - __main__ - INFO - === 创建数据加载器 ===
2025-08-08 21:41:19,702 - __main__ - INFO - 🔽 使用降分辨率数据集 (DownsampledResolutionDataset)
2025-08-08 21:41:19,702 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 分辨率降采样器初始化完成:
2025-08-08 21:41:19,702 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-08 21:41:19,702 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   保持宽高比: True
2025-08-08 21:41:19,702 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   抗锯齿: True
2025-08-08 21:41:19,703 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载初始化: tensor
2025-08-08 21:41:19,703 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 数据形状: (10000, 1, 128, 128)
2025-08-08 21:41:19,703 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 🚀 启用懒加载模式，节省内存使用
2025-08-08 21:41:19,703 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 计算全局归一化参数...
2025-08-08 21:41:19,703 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载模式：使用采样数据计算归一化参数
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 基于 100 个样本计算归一化参数
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 全局数据范围: [0.001869, 7.435785]
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 降采样数据集初始化完成:
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输入分辨率: (32, 32) -> 1024维
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输出分辨率: (128, 128) -> 16384维
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   样本数量: 5000
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   数据归一化: True
2025-08-08 21:41:19,714 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   懒加载模式: True
2025-08-08 21:41:19,714 - __main__ - INFO - 降采样方法: bicubic
2025-08-08 21:41:19,714 - __main__ - INFO - 保持宽高比: True
2025-08-08 21:41:19,714 - __main__ - INFO - 抗锯齿: True
2025-08-08 21:41:19,714 - __main__ - INFO - 后端: opencv
2025-08-08 21:41:19,715 - __main__ - INFO - 🔧 设置多进程启动方法为spawn以提高稳定性
2025-08-08 21:41:19,715 - __main__ - INFO - 🔄 数据加载器配置: num_workers=16, pin_memory=True, drop_last=True, prefetch_factor=8
2025-08-08 21:41:19,715 - __main__ - INFO - 🔄 持久化工作进程: True, CPU核心数: 192
2025-08-08 21:41:19,715 - __main__ - INFO - 数据集分割: 训练集=4000, 验证集=500, 测试集=500
2025-08-08 21:41:19,717 - __main__ - INFO - 数据统计: {'num_samples': 5000, 'input_shape': (32, 32), 'output_shape': (128, 128), 'input_dim': 1024, 'output_dim': 16384, 'input_range': (0.01112553384155035, 0.6109797954559326), 'output_range': (0.0020235178526490927, 0.611142635345459), 'downsample_method': 'bicubic'}
2025-08-08 21:41:19,717 - __main__ - INFO - === 归一化信息 ===
2025-08-08 21:41:19,717 - __main__ - INFO - 归一化方法: min_max_global
2025-08-08 21:41:19,717 - __main__ - INFO - 全局数据范围: [0.001869, 7.435785]
2025-08-08 21:41:19,717 - __main__ - INFO - 原始数据形状: (10000, 1, 128, 128)
2025-08-08 21:41:19,717 - __main__ - INFO - 输入分辨率: (32, 32)
2025-08-08 21:41:19,717 - __main__ - INFO - 输出分辨率: (128, 128)
2025-08-08 21:41:19,717 - __main__ - INFO - ✅ 数据已归一化到[0,1]范围，可用于反归一化恢复物理信息
2025-08-08 21:41:19,717 - __main__ - INFO - === 创建模型 ===
2025-08-08 21:41:22,901 - __main__ - INFO - 模型参数数量: 869208192
2025-08-08 21:41:22,902 - __main__ - INFO - 🚀 启用多GPU训练: 检测到 2 张GPU
2025-08-08 21:41:22,913 - __main__ - INFO - ✅ 已将模型移动到主设备 cuda:0
2025-08-08 21:41:23,051 - __main__ - INFO - 🧹 已清理所有GPU缓存
2025-08-08 21:41:23,052 - __main__ - INFO -    使用的GPU设备: [0, 1]
2025-08-08 21:41:23,052 - __main__ - INFO -    GPU 0: NVIDIA L40 (44.3GB)
2025-08-08 21:41:23,052 - __main__ - INFO -    GPU 1: NVIDIA L40 (44.3GB)
2025-08-08 21:41:23,052 - __main__ - INFO - === 创建增强版损失函数 ===
2025-08-08 21:41:23,052 - __main__ - INFO - 损失函数配置:
2025-08-08 21:41:23,052 - __main__ - INFO -   - 基础权重 (base_weight): 1.0
2025-08-08 21:41:23,052 - __main__ - INFO -   - SVD权重 (svd_weights): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-08-08 21:41:23,052 - __main__ - INFO -   - SVD模态数 (topk): 10
2025-08-08 21:41:23,052 - __main__ - INFO -   - 损失类型: mse
2025-08-08 21:41:23,052 - __main__ - INFO -   - SVD损失启用: False
2025-08-08 21:41:23,052 - __main__ - INFO -   - 增强版SVD: False
2025-08-08 21:41:23,052 - __main__ - INFO -   - 混合精度优化: False
2025-08-08 21:41:23,052 - __main__ - INFO -   - 自适应权重: False
2025-08-08 21:41:23,052 - __main__ - INFO -   - Fallback级别: 2
2025-08-08 21:41:23,052 - __main__ - INFO -   - 性能监控: False
2025-08-08 21:41:23,052 - __main__ - INFO -   - 强制SVD: False
2025-08-08 21:41:23,052 - __main__ - INFO - ⚠️ 禁用SVD损失，使用标准MSE损失函数
2025-08-08 21:41:23,052 - __main__ - INFO - === 使用标准MSE损失函数 ===
2025-08-08 21:41:23,052 - __main__ - INFO - SVD计算已完全跳过，节省计算资源
2025-08-08 21:41:23,052 - __main__ - INFO - ================================
2025-08-08 21:41:23,053 - __main__ - INFO - === 高级训练功能状态 ===
2025-08-08 21:41:23,053 - __main__ - INFO - 梯度裁剪: 启用
2025-08-08 21:41:23,053 - __main__ - INFO -   裁剪阈值: 0.5
2025-08-08 21:41:23,053 - __main__ - INFO - 混合精度训练: 启用
2025-08-08 21:41:23,053 - __main__ - INFO -   损失缩放: dynamic
2025-08-08 21:41:23,053 - __main__ - INFO - 梯度累积: 启用
2025-08-08 21:41:23,053 - __main__ - INFO -   累积步数: 4
2025-08-08 21:41:23,053 - __main__ - INFO - 🔍 GPU内存状态 训练开始前:
2025-08-08 21:41:23,053 - __main__ - INFO -   已分配: 3.24 GB
2025-08-08 21:41:23,053 - __main__ - INFO -   已保留: 3.26 GB
2025-08-08 21:41:23,053 - __main__ - INFO -   峰值分配: 3.24 GB
2025-08-08 21:41:23,053 - __main__ - INFO - === 开始训练 ===
2025-08-08 21:45:47,068 - __main__ - WARNING - ⚠️ 训练被用户中断
2025-08-08 21:45:59,623 - __main__ - INFO - 📝 日志系统已初始化，日志文件: ./results/logs/server_downsampling_training.log
2025-08-08 21:45:59,623 - __main__ - INFO - ℹ️ SVD损失已禁用，跳过SVD权重验证
2025-08-08 21:45:59,624 - __main__ - INFO - ✅ 配置验证通过
2025-08-08 21:45:59,624 - __main__ - INFO - === 配置信息 ===
2025-08-08 21:45:59,624 - __main__ - INFO - 输入分辨率: [32, 32] -> 1024维
2025-08-08 21:45:59,624 - __main__ - INFO - 输出分辨率: [128, 128] -> 16384维
2025-08-08 21:45:59,624 - __main__ - INFO - 样本数量: 5000
2025-08-08 21:45:59,624 - __main__ - INFO - 批次大小: 64
2025-08-08 21:45:59,624 - __main__ - INFO - 训练轮数: 550
2025-08-08 21:45:59,624 - __main__ - INFO - 注意力机制: sge
2025-08-08 21:45:59,624 - __main__ - INFO - 🎲 设置随机种子: 42
2025-08-08 21:45:59,624 - __main__ - INFO - 🚀 CPU优化设置: 检测到192个CPU核心，设置190个线程用于计算
2025-08-08 21:45:59,733 - __main__ - INFO - 🔍 CPU状态 初始状态:
2025-08-08 21:45:59,733 - __main__ - INFO -   内存使用: 117.75GB/1007.06GB (12.4%)
2025-08-08 21:45:59,733 - __main__ - INFO -   CPU使用率: 1.4%
2025-08-08 21:45:59,810 - __main__ - INFO - 🔍 GPU内存状态 初始状态:
2025-08-08 21:45:59,810 - __main__ - INFO -   已分配: 0.00 GB
2025-08-08 21:45:59,810 - __main__ - INFO -   已保留: 0.00 GB
2025-08-08 21:45:59,810 - __main__ - INFO -   峰值分配: 0.00 GB
2025-08-08 21:45:59,810 - __main__ - INFO - 🖥️ 使用设备: cuda
2025-08-08 21:46:00,205 - __main__ - INFO - 💾 设置GPU显存限制: 90.0% (应用于 2 张GPU)
2025-08-08 21:46:00,206 - __main__ - INFO - 📁 模型保存目录: ./results/models
2025-08-08 21:46:00,206 - __main__ - INFO - === 验证配置 ===
2025-08-08 21:46:00,206 - __main__ - INFO - ✅ 配置验证通过
2025-08-08 21:46:00,206 - __main__ - INFO - 配置验证详情: warnings=0, errors=0
2025-08-08 21:46:00,206 - __main__ - INFO - === 创建数据加载器 ===
2025-08-08 21:46:00,206 - __main__ - INFO - 🔽 使用降分辨率数据集 (DownsampledResolutionDataset)
2025-08-08 21:46:00,206 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 分辨率降采样器初始化完成:
2025-08-08 21:46:00,206 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-08 21:46:00,206 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   保持宽高比: True
2025-08-08 21:46:00,206 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   抗锯齿: True
2025-08-08 21:46:00,207 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载初始化: tensor
2025-08-08 21:46:00,207 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 数据形状: (10000, 1, 128, 128)
2025-08-08 21:46:00,207 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 🚀 启用懒加载模式，节省内存使用
2025-08-08 21:46:00,207 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 计算全局归一化参数...
2025-08-08 21:46:00,207 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载模式：使用采样数据计算归一化参数
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 基于 100 个样本计算归一化参数
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 全局数据范围: [0.001869, 7.435785]
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 降采样数据集初始化完成:
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输入分辨率: (32, 32) -> 1024维
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输出分辨率: (128, 128) -> 16384维
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   样本数量: 5000
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   数据归一化: True
2025-08-08 21:46:00,218 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   懒加载模式: True
2025-08-08 21:46:00,218 - __main__ - INFO - 降采样方法: bicubic
2025-08-08 21:46:00,218 - __main__ - INFO - 保持宽高比: True
2025-08-08 21:46:00,218 - __main__ - INFO - 抗锯齿: True
2025-08-08 21:46:00,218 - __main__ - INFO - 后端: opencv
2025-08-08 21:46:00,218 - __main__ - INFO - 🔧 设置多进程启动方法为spawn以提高稳定性
2025-08-08 21:46:00,219 - __main__ - INFO - 🎯 智能优化(降采样): 设置num_workers=32
2025-08-08 21:46:00,219 - __main__ - INFO - 🚀 最终设置: num_workers=32 (CPU核心数: 192)
2025-08-08 21:46:00,219 - __main__ - INFO - 🔄 数据加载器配置: num_workers=32, pin_memory=True, drop_last=True, prefetch_factor=8
2025-08-08 21:46:00,219 - __main__ - INFO - 🔄 持久化工作进程: True, CPU核心数: 192
2025-08-08 21:46:00,219 - __main__ - INFO - 数据集分割: 训练集=4000, 验证集=500, 测试集=500
2025-08-08 21:46:00,221 - __main__ - INFO - 数据统计: {'num_samples': 5000, 'input_shape': (32, 32), 'output_shape': (128, 128), 'input_dim': 1024, 'output_dim': 16384, 'input_range': (0.01112553384155035, 0.6109797954559326), 'output_range': (0.0020235178526490927, 0.611142635345459), 'downsample_method': 'bicubic'}
2025-08-08 21:46:00,221 - __main__ - INFO - === 归一化信息 ===
2025-08-08 21:46:00,221 - __main__ - INFO - 归一化方法: min_max_global
2025-08-08 21:46:00,221 - __main__ - INFO - 全局数据范围: [0.001869, 7.435785]
2025-08-08 21:46:00,221 - __main__ - INFO - 原始数据形状: (10000, 1, 128, 128)
2025-08-08 21:46:00,221 - __main__ - INFO - 输入分辨率: (32, 32)
2025-08-08 21:46:00,221 - __main__ - INFO - 输出分辨率: (128, 128)
2025-08-08 21:46:00,221 - __main__ - INFO - ✅ 数据已归一化到[0,1]范围，可用于反归一化恢复物理信息
2025-08-08 21:46:00,221 - __main__ - INFO - === 创建模型 ===
2025-08-08 21:46:03,419 - __main__ - INFO - 模型参数数量: 869208192
2025-08-08 21:46:03,419 - __main__ - INFO - 🚀 启用多GPU训练: 检测到 2 张GPU
2025-08-08 21:46:03,430 - __main__ - INFO - ✅ 已将模型移动到主设备 cuda:0
2025-08-08 21:46:03,568 - __main__ - INFO - 🧹 已清理所有GPU缓存
2025-08-08 21:46:03,568 - __main__ - INFO -    使用的GPU设备: [0, 1]
2025-08-08 21:46:03,568 - __main__ - INFO -    GPU 0: NVIDIA L40 (44.3GB)
2025-08-08 21:46:03,568 - __main__ - INFO -    GPU 1: NVIDIA L40 (44.3GB)
2025-08-08 21:46:03,568 - __main__ - INFO - === 创建增强版损失函数 ===
2025-08-08 21:46:03,568 - __main__ - INFO - 损失函数配置:
2025-08-08 21:46:03,568 - __main__ - INFO -   - 基础权重 (base_weight): 1.0
2025-08-08 21:46:03,568 - __main__ - INFO -   - SVD权重 (svd_weights): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-08-08 21:46:03,568 - __main__ - INFO -   - SVD模态数 (topk): 10
2025-08-08 21:46:03,568 - __main__ - INFO -   - 损失类型: mse
2025-08-08 21:46:03,568 - __main__ - INFO -   - SVD损失启用: False
2025-08-08 21:46:03,568 - __main__ - INFO -   - 增强版SVD: False
2025-08-08 21:46:03,568 - __main__ - INFO -   - 混合精度优化: False
2025-08-08 21:46:03,568 - __main__ - INFO -   - 自适应权重: False
2025-08-08 21:46:03,568 - __main__ - INFO -   - Fallback级别: 2
2025-08-08 21:46:03,569 - __main__ - INFO -   - 性能监控: False
2025-08-08 21:46:03,569 - __main__ - INFO -   - 强制SVD: False
2025-08-08 21:46:03,569 - __main__ - INFO - ⚠️ 禁用SVD损失，使用标准MSE损失函数
2025-08-08 21:46:03,569 - __main__ - INFO - === 使用标准MSE损失函数 ===
2025-08-08 21:46:03,569 - __main__ - INFO - SVD计算已完全跳过，节省计算资源
2025-08-08 21:46:03,569 - __main__ - INFO - ================================
2025-08-08 21:46:03,569 - __main__ - INFO - === 高级训练功能状态 ===
2025-08-08 21:46:03,569 - __main__ - INFO - 梯度裁剪: 启用
2025-08-08 21:46:03,569 - __main__ - INFO -   裁剪阈值: 0.5
2025-08-08 21:46:03,570 - __main__ - INFO - 混合精度训练: 启用
2025-08-08 21:46:03,570 - __main__ - INFO -   损失缩放: dynamic
2025-08-08 21:46:03,570 - __main__ - INFO - 梯度累积: 启用
2025-08-08 21:46:03,570 - __main__ - INFO -   累积步数: 4
2025-08-08 21:46:03,570 - __main__ - INFO - 🔍 GPU内存状态 训练开始前:
2025-08-08 21:46:03,570 - __main__ - INFO -   已分配: 3.24 GB
2025-08-08 21:46:03,570 - __main__ - INFO -   已保留: 3.26 GB
2025-08-08 21:46:03,570 - __main__ - INFO -   峰值分配: 3.24 GB
2025-08-08 21:46:03,570 - __main__ - INFO - === 开始训练 ===
2025-08-09 12:35:46,881 - __main__ - INFO - === 开始测试 ===
2025-08-09 12:35:57,901 - __main__ - INFO - 最终测试损失: 0.001522
2025-08-09 12:35:58,125 - __main__ - INFO - 🎉 === 训练完成 ===
2025-08-09 12:35:58,226 - __main__ - INFO - 🔍 CPU状态 训练完成后:
2025-08-09 12:35:58,226 - __main__ - INFO -   内存使用: 151.76GB/1007.06GB (15.8%)
2025-08-09 12:35:58,226 - __main__ - INFO -   CPU使用率: 1.5%
2025-08-09 12:35:58,226 - __main__ - INFO - 🔍 GPU内存状态 训练完成后:
2025-08-09 12:35:58,226 - __main__ - INFO -   已分配: 12.97 GB
2025-08-09 12:35:58,226 - __main__ - INFO -   已保留: 37.31 GB
2025-08-09 12:35:58,226 - __main__ - INFO -   峰值分配: 29.55 GB
2025-08-09 12:35:58,226 - __main__ - WARNING - ⚠️  GPU内存使用较高，建议启用懒加载或减少批次大小
2025-08-09 12:35:58,784 - __main__ - INFO - 🧹 已清理GPU缓存
2025-08-09 12:35:58,784 - __main__ - INFO - 
=== 服务器资源使用总结 ===
2025-08-09 12:35:58,784 - __main__ - INFO - 💾 CPU内存: 充分利用多进程数据加载 (num_workers=0)
2025-08-09 12:35:58,784 - __main__ - INFO - 🚀 GPU计算: 核心模型训练和推理
2025-08-09 12:35:58,784 - __main__ - INFO - 📊 数据流水线: pin_memory=True, prefetch_factor=8
2025-08-09 12:35:58,784 - __main__ - INFO - ⚡ 持久化工作进程: persistent_workers=True
2025-08-09 12:35:58,786 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 归一化信息已保存到: results/normalization_info.json
2025-08-09 12:35:58,786 - __main__ - INFO - 📊 归一化信息已保存: results/normalization_info.json
2025-08-09 12:35:58,786 - __main__ - INFO - 💡 使用提示:
2025-08-09 12:35:58,786 - __main__ - INFO -    - 使用 dataset.denormalize_predictions(predictions) 反归一化预测结果
2025-08-09 12:35:58,786 - __main__ - INFO -    - 使用 DynamicResolutionDataset.load_normalization_info(path) 加载归一化信息
2025-08-09 12:35:58,786 - __main__ - INFO -    - 参考 demo_global_normalization.py 了解完整用法
2025-08-09 12:35:58,792 - __main__ - INFO - 💾 最终配置已保存: results/final_config.yaml
