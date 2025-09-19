# PDE数据集全网络训练指南

## 核心需求分析
完成PDE数据集的全网络训练，并统一参数量配置，实现公平的模型对比实验。

## 推荐主脚本

### 1. 最佳选择：`generate_data/enhanced_pdebench_trainer.py`

**推荐理由：**
- ✅ 集成PDEBench原生模型（FNO, UNet, PINN）作为基线
- ✅ 支持稀疏到稠密的流场重建任务
- ✅ 统一的训练和评估框架
- ✅ 公平的横向对比实验
- ✅ 支持动态分辨率和SVD模态投影
- ✅ 增强的损失函数和优化策略

**运行命令：**
```bash
python generate_data/enhanced_pdebench_trainer.py --config modify_multi_attention/models/unified_comparison_config.yaml --output_dir results/unified_training
```

### 2. 备选方案：`generate_data/dynamic_resolution_trainer.py`

**适用场景：**
- 需要动态分辨率训练
- 实时生成训练数据
- SVD模态投影实验

**运行命令：**
```bash
python generate_data/dynamic_resolution_trainer.py --config config_training.yaml --output_dir results/dynamic_training
```

### 3. 基础方案：`modify_multi_attention/main.py`

**适用场景：**
- 基础Transformer模型训练
- 简单的配置文件加载

**运行命令：**
```bash
python modify_multi_attention/main.py --config config.yaml --output_dir results/basic_training
```

## 统一参数量配置方案

### 配置文件：`unified_comparison_config.yaml`

**关键参数设置：**

#### 数据配置
```yaml
data:
  input_resolution: 64      # 输入分辨率 [B, 1, 64] → [B, 1, 128]
  output_resolution: 128    # 输出分辨率
  input_channels: 1         # 输入通道数
  output_channels: 1        # 输出通道数
  batch_size: 4            # 批次大小
```

#### 模型参数量统一
```yaml
models:
  fno_1d:
    target_params: 287425   # 目标参数量
    modes: 16
    width: 64
    
  unet_1d:
    target_params: 500000
    init_features: 32
    
  mlp_1d:
    target_params: 200000
    hidden_dim: 256
    num_layers: 8
    
  pinn_1d:
    target_params: 300000
    hidden_dim: 256
    num_layers: 8
```

#### 训练配置
```yaml
training:
  epochs: 10               # 训练轮数
  learning_rate: 0.001     # 学习率
  weight_decay: 1e-5       # 权重衰减
  validation_split: 0.2    # 验证集比例
```

## 执行步骤

### 步骤1：环境准备
```bash
cd x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1
```

### 步骤2：运行训练
```bash
# 推荐方案
python generate_data/enhanced_pdebench_trainer.py --config modify_multi_attention/models/unified_comparison_config.yaml --output_dir results/unified_training

# 或者使用动态分辨率训练
python generate_data/dynamic_resolution_trainer.py --config config_training.yaml --output_dir results/dynamic_training
```

### 步骤3：结果分析
训练完成后，结果将保存在指定的输出目录中，包括：
- 模型权重文件
- 训练日志
- 性能指标
- 可视化结果

## 技术特点

### 数据流形状变换
- **输入形状：** [B, 1, 64] 或 [B, 1, 64, 64]
- **输出形状：** [B, 1, 128] 或 [B, 1, 128, 128]
- **批次大小：** B = 4（内存优化）

### 模型架构对比
1. **FNO模型：** 频域神经算子，适合PDE求解
2. **UNet模型：** 编码器-解码器架构，适合图像重建
3. **MLP模型：** 多层感知机，基线对比
4. **PINN模型：** 物理信息神经网络，融合物理约束

### 损失函数配置
```yaml
loss:
  weights:
    mse: 1.0        # 均方误差
    mae: 0.1        # 平均绝对误差
    relative: 0.1   # 相对误差
```

## 预期输出

### 训练结果
- 各模型的训练损失曲线
- 验证集性能指标
- 模型参数量统计
- 推理时间对比

### 可视化结果
- 重建质量对比图
- 损失函数收敛曲线
- 模型性能雷达图
- 参数量-性能散点图

## 注意事项

1. **内存管理：** 批次大小设置为4，避免GPU内存溢出
2. **路径配置：** 确保所有路径使用绝对路径
3. **依赖检查：** 确认PDEBench模块正确安装
4. **结果保存：** 训练结果自动保存到指定目录

## 总结

推荐使用 <mcfile name="enhanced_pdebench_trainer.py" path="generate_data/enhanced_pdebench_trainer.py"></mcfile> 配合 <mcfile name="unified_comparison_config.yaml" path="modify_multi_attention/models/unified_comparison_config.yaml"></mcfile> 进行PDE数据集的全网络训练，该方案提供了完整的模型对比框架和统一的参数量配置。