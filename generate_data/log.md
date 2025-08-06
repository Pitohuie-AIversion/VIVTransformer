/share/fandixiaLab/suguangsheng/anaconda3/bin/python /share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/generate_data/dynamic_resolution_trainer.py --config dynamic_config_server_downsampling.yaml 
2025-08-07 05:24:33,683 - INFO - ✅ 成功导入降分辨率模块
/share/fandixiaLab/suguangsheng/anaconda3/lib/python3.12/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
✅ 成功加载配置文件: dynamic_config_server_downsampling.yaml
🔄 配置文件已合并，epochs设置为: 500
2025-08-07 05:24:35,742 - __main__ - INFO - 📝 日志系统已初始化，日志文件: ./results/logs/server_downsampling_training.log
2025-08-07 05:24:35,744 - __main__ - INFO - ✅ 配置验证通过
2025-08-07 05:24:35,744 - __main__ - INFO - === 配置信息 ===
2025-08-07 05:24:35,744 - __main__ - INFO - 输入分辨率: [32, 32] -> 1024维
2025-08-07 05:24:35,744 - __main__ - INFO - 输出分辨率: [128, 128] -> 16384维
2025-08-07 05:24:35,744 - __main__ - INFO - 样本数量: 100
2025-08-07 05:24:35,744 - __main__ - INFO - 批次大小: 8
2025-08-07 05:24:35,744 - __main__ - INFO - 训练轮数: 500
2025-08-07 05:24:35,744 - __main__ - INFO - 注意力机制: sge
2025-08-07 05:24:35,747 - __main__ - INFO - 🎲 设置随机种子: 42
2025-08-07 05:24:35,766 - __main__ - INFO - 🚀 CPU优化设置: 检测到192个CPU核心，设置190个线程用于计算
2025-08-07 05:24:35,930 - __main__ - INFO - 🔍 CPU状态 初始状态:
2025-08-07 05:24:35,931 - __main__ - INFO -   内存使用: 116.46GB/1007.06GB (12.3%)
2025-08-07 05:24:35,931 - __main__ - INFO -   CPU使用率: 45.9%
2025-08-07 05:24:36,013 - __main__ - INFO - 🔍 GPU内存状态 初始状态:
2025-08-07 05:24:36,013 - __main__ - INFO -   已分配: 0.00 GB
2025-08-07 05:24:36,013 - __main__ - INFO -   已保留: 0.00 GB
2025-08-07 05:24:36,013 - __main__ - INFO -   峰值分配: 0.00 GB
2025-08-07 05:24:36,013 - __main__ - INFO - 🖥️ 使用设备: cuda
2025-08-07 05:24:36,422 - __main__ - INFO - 💾 设置GPU显存限制: 95.0% (应用于 2 张GPU)
2025-08-07 05:24:36,423 - __main__ - INFO - 📁 模型保存目录: ./results/models
2025-08-07 05:24:36,423 - __main__ - INFO - === 验证配置 ===
2025-08-07 05:24:36,423 - __main__ - INFO - ✅ 配置验证通过
2025-08-07 05:24:36,423 - __main__ - INFO - 配置验证详情: warnings=0, errors=0
2025-08-07 05:24:36,423 - __main__ - INFO - === 创建数据加载器 ===
2025-08-07 05:24:36,423 - __main__ - INFO - 🔽 使用降分辨率数据集 (DownsampledResolutionDataset)
2025-08-07 05:24:36,423 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 分辨率降采样器初始化完成:
2025-08-07 05:24:36,423 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-07 05:24:36,423 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   保持宽高比: True
2025-08-07 05:24:36,423 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   抗锯齿: True
2025-08-07 05:24:36,424 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载初始化: tensor
2025-08-07 05:24:36,424 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 数据形状: (10000, 1, 128, 128)
2025-08-07 05:24:36,424 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 🚀 启用懒加载模式，节省内存使用
2025-08-07 05:24:36,424 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 计算全局归一化参数...
2025-08-07 05:24:36,424 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 懒加载模式：使用采样数据计算归一化参数
Traceback (most recent call last):
  File "/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/generate_data/dynamic_resolution_trainer.py", line 1347, in main
    train_loader, valid_loader, test_loader, dataset = get_dynamic_loaders(config)
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/generate_data/dynamic_resolution_trainer.py", line 854, in get_dynamic_loaders
    if num_workers > 8 and prefetch_factor < 4:
                           ^^^^^^^^^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 基于 100 个样本计算归一化参数
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 全局数据范围: [0.001869, 7.435785]
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO - 降采样数据集初始化完成:
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输入分辨率: (32, 32) -> 1024维
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   输出分辨率: (128, 128) -> 16384维
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   样本数量: 100
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   降采样方法: bicubic
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   数据归一化: True
2025-08-07 05:24:36,435 - pde_process.resolution_downsampler.resolution_downsampler - INFO -   懒加载模式: True
2025-08-07 05:24:36,435 - __main__ - INFO - 降采样方法: bicubic
2025-08-07 05:24:36,435 - __main__ - INFO - 保持宽高比: True
2025-08-07 05:24:36,435 - __main__ - INFO - 抗锯齿: True
2025-08-07 05:24:36,435 - __main__ - INFO - 后端: opencv
2025-08-07 05:24:36,435 - __main__ - INFO - 🚀 服务器优化: 自动设置num_workers=16 (CPU核心数: 192)
2025-08-07 05:24:36,436 - __main__ - ERROR - ❌ 训练过程中出错: '<' not supported between instances of 'NoneType' and 'int'
2025-08-07 05:24:36,436 - __main__ - ERROR - 详细错误信息请查看上方的堆栈跟踪

Process finished with exit code 0
