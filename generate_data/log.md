/share/fandixiaLab/suguangsheng/anaconda3/bin/python /share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/generate_data/dynamic_resolution_trainer.py --config dynamic_config_server_downsampling.yaml 
2025-08-06 02:56:49,874 - INFO - ✅ 成功导入降分辨率模块
/share/fandixiaLab/suguangsheng/anaconda3/lib/python3.12/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
✅ 成功加载配置文件: dynamic_config_server_downsampling.yaml
🔄 配置文件已合并，epochs设置为: 500
2025-08-06 02:56:51,589 - __main__ - INFO - 📝 日志系统已初始化，日志文件: ./results/logs/server_downsampling_training.log
2025-08-06 02:56:51,590 - __main__ - INFO - ℹ️ SVD损失已禁用，跳过SVD权重验证
2025-08-06 02:56:51,591 - __main__ - INFO - ✅ 配置验证通过
2025-08-06 02:56:51,591 - __main__ - INFO - === 配置信息 ===
2025-08-06 02:56:51,591 - __main__ - INFO - 输入分辨率: [32, 32] -> 1024维
2025-08-06 02:56:51,591 - __main__ - INFO - 输出分辨率: [128, 128] -> 16384维
2025-08-06 02:56:51,591 - __main__ - INFO - 样本数量: 5000
2025-08-06 02:56:51,591 - __main__ - INFO - 批次大小: 64
2025-08-06 02:56:51,591 - __main__ - INFO - 训练轮数: 500
2025-08-06 02:56:51,591 - __main__ - INFO - 注意力机制: sge
2025-08-06 02:56:51,594 - __main__ - INFO - 🎲 设置随机种子: 42
2025-08-06 02:56:51,594 - __main__ - INFO - 🖥️ 使用设备: cuda
2025-08-06 02:56:52,092 - __main__ - INFO - 💾 设置GPU显存限制: 90.0% (应用于 2 张GPU)
2025-08-06 02:56:52,093 - __main__ - INFO - 📁 模型保存目录: ./results/models
2025-08-06 02:56:52,093 - __main__ - INFO - === 验证配置 ===
2025-08-06 02:56:52,093 - __main__ - INFO - ✅ 配置验证通过
2025-08-06 02:56:52,093 - __main__ - INFO - 配置验证详情: warnings=0, errors=0
2025-08-06 02:56:52,093 - __main__ - INFO - === 创建数据加载器 ===
2025-08-06 02:56:52,093 - __main__ - INFO - 🔽 使用降分辨率数据集 (DownsampledResolutionDataset)
2025-08-06 02:56:52,093 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 分辨率降采样器初始化完成:
2025-08-06 02:56:52,093 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-06 02:56:52,093 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   保持宽高比: True
2025-08-06 02:56:52,093 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   抗锯齿: True
2025-08-06 02:56:52,094 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载初始化: tensor
2025-08-06 02:56:52,094 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 数据形状: (10000, 1, 128, 128)
2025-08-06 02:56:52,094 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 🚀 启用懒加载模式，节省内存使用
2025-08-06 02:56:52,094 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 计算全局归一化参数...
2025-08-06 02:56:52,094 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载模式：使用采样数据计算归一化参数
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 基于 100 个样本计算归一化参数
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 全局数据范围: [0.001869, 7.435785]
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 降采样数据集初始化完成:
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输入分辨率: (32, 32) -> 1024维
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输出分辨率: (128, 128) -> 16384维
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   样本数量: 5000
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   数据归一化: True
2025-08-06 02:56:52,106 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   懒加载模式: True
2025-08-06 02:56:52,106 - __main__ - INFO - 降采样方法: bicubic
2025-08-06 02:56:52,106 - __main__ - INFO - 保持宽高比: True
2025-08-06 02:56:52,106 - __main__ - INFO - 抗锯齿: True
2025-08-06 02:56:52,106 - __main__ - INFO - 后端: opencv
2025-08-06 02:56:52,107 - __main__ - INFO - 🔄 数据加载器配置: num_workers=16, pin_memory=True, drop_last=True, prefetch_factor=8
2025-08-06 02:56:52,107 - __main__ - INFO - 数据集分割: 训练集=4000, 验证集=500, 测试集=500
2025-08-06 02:56:52,109 - __main__ - INFO - 数据统计: {'num_samples': 5000, 'input_shape': (32, 32), 'output_shape': (128, 128), 'input_dim': 1024, 'output_dim': 16384, 'input_range': (0.01112553384155035, 0.6109797954559326), 'output_range': (0.0020235178526490927, 0.611142635345459), 'downsample_method': 'bicubic'}
2025-08-06 02:56:52,109 - __main__ - INFO - === 归一化信息 ===
2025-08-06 02:56:52,109 - __main__ - INFO - 归一化方法: min_max_global
2025-08-06 02:56:52,109 - __main__ - INFO - 全局数据范围: [0.001869, 7.435785]
2025-08-06 02:56:52,109 - __main__ - INFO - 原始数据形状: (10000, 1, 128, 128)
2025-08-06 02:56:52,109 - __main__ - INFO - 输入分辨率: (32, 32)
2025-08-06 02:56:52,109 - __main__ - INFO - 输出分辨率: (128, 128)
2025-08-06 02:56:52,109 - __main__ - INFO - ✅ 数据已归一化到[0,1]范围，可用于反归一化恢复物理信息
2025-08-06 02:56:52,109 - __main__ - INFO - === 创建模型 ===
2025-08-06 02:56:55,848 - __main__ - INFO - 模型参数数量: 869208192
2025-08-06 02:56:55,849 - __main__ - INFO - 🚀 启用多GPU训练: 检测到 2 张GPU
2025-08-06 02:56:55,860 - __main__ - INFO - ✅ 已将模型移动到主设备 cuda:0
2025-08-06 02:56:55,998 - __main__ - INFO - 🧹 已清理所有GPU缓存
2025-08-06 02:56:55,998 - __main__ - INFO -    使用的GPU设备: [0, 1]
2025-08-06 02:56:55,998 - __main__ - INFO -    GPU 0: NVIDIA L40 (44.3GB)
2025-08-06 02:56:55,998 - __main__ - INFO -    GPU 1: NVIDIA L40 (44.3GB)
2025-08-06 02:56:55,998 - __main__ - INFO - === 创建增强版损失函数 ===
2025-08-06 02:56:55,998 - __main__ - INFO - 损失函数配置:
2025-08-06 02:56:55,998 - __main__ - INFO -   - 基础权重 (base_weight): 1.0
2025-08-06 02:56:55,998 - __main__ - INFO -   - SVD权重 (svd_weights): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2025-08-06 02:56:55,998 - __main__ - INFO -   - SVD模态数 (topk): 10
2025-08-06 02:56:55,999 - __main__ - INFO -   - 损失类型: mse
2025-08-06 02:56:55,999 - __main__ - INFO -   - SVD损失启用: False
2025-08-06 02:56:55,999 - __main__ - INFO -   - 增强版SVD: False
2025-08-06 02:56:55,999 - __main__ - INFO -   - 混合精度优化: False
2025-08-06 02:56:55,999 - __main__ - INFO -   - 自适应权重: False
2025-08-06 02:56:55,999 - __main__ - INFO -   - Fallback级别: 2
2025-08-06 02:56:55,999 - __main__ - INFO -   - 性能监控: False
2025-08-06 02:56:55,999 - __main__ - INFO -   - 强制SVD: False
2025-08-06 02:56:55,999 - __main__ - INFO - ⚠️ 禁用SVD损失，使用标准MSE损失函数
2025-08-06 02:56:55,999 - __main__ - INFO - === 使用标准MSE损失函数 ===
2025-08-06 02:56:55,999 - __main__ - INFO - SVD计算已完全跳过，节省计算资源
2025-08-06 02:56:55,999 - __main__ - INFO - ================================
🔧 优化器配置: 类型=adam, 学习率=0.0005, 权重衰减=0.0001
   Adam参数: betas=[0.9, 0.999], eps=1e-8, amsgrad=False
📈 学习率调度器配置: 类型=cosine, 启用=True
   Cosine参数: T_max=500
2025-08-06 02:56:56,000 - __main__ - INFO - === 高级训练功能状态 ===
2025-08-06 02:56:56,000 - __main__ - INFO - 梯度裁剪: 启用
2025-08-06 02:56:56,000 - __main__ - INFO -   裁剪阈值: 0.5
2025-08-06 02:56:56,000 - __main__ - INFO - 混合精度训练: 启用
2025-08-06 02:56:56,000 - __main__ - INFO -   损失缩放: dynamic
2025-08-06 02:56:56,000 - __main__ - INFO - 梯度累积: 启用
2025-08-06 02:56:56,000 - __main__ - INFO -   累积步数: 4
2025-08-06 02:56:56,000 - __main__ - INFO - 🔍 GPU内存状态 训练开始前:
2025-08-06 02:56:56,000 - __main__ - INFO -   已分配: 3.24 GB
2025-08-06 02:56:56,000 - __main__ - INFO -   已保留: 3.26 GB
2025-08-06 02:56:56,000 - __main__ - INFO -   峰值分配: 3.24 GB
2025-08-06 02:56:56,000 - __main__ - INFO - === 开始训练 ===
🚀 启用混合精度训练 (AMP)，损失缩放: dynamic
📊 启用梯度累积，累积步数: 4
📊 早停配置: 启用=False, 监控=val_loss, 模式=min, 耐心值=200
写入loss_log.txt到：./results/loss_logs/loss_log.txt
检测到断点文件，自动恢复：./results/checkpoint_sge.pth
/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/modify_multi_attention/training/trainer.py:33: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler() if use_amp else None
已恢复到 epoch 20，best_metric=inf，patience_counter=0
/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/modify_multi_attention/training/trainer.py:153: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
    🔄 Epoch [21/500], Batch [1/62], Loss: 0.016721
    🔄 Epoch [21/500], Batch [50/62], Loss: 0.015353
🎯 Epoch [21/500], Train Loss: 0.017167, Valid Loss: 0.019149, Test Loss: 0.015924
