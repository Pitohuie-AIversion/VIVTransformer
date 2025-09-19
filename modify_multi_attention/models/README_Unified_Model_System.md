# 统一模型系统使用指南

## 概述

统一模型系统是一个灵活、可扩展的深度学习模型管理框架，专为PDE求解和流体动力学建模设计。该系统提供了统一的接口来创建、配置和管理多种类型的神经网络模型。

### 核心组件

1. **统一模型工厂 (UnifiedModelFactory)** - 负责创建和管理所有模型类型
2. **统一配置管理器 (UnifiedConfigManager)** - 处理所有配置参数的验证和管理
3. **Trainer兼容性适配器 (TrainerCompatibilityAdapter)** - 与现有训练系统的兼容性桥梁

### 支持的模型类型

- **Enhanced Transformer 1D/2D/3D** - 增强版Transformer模型，支持多种注意力机制
- **Enhanced FNO 1D/2D/3D** - 增强版傅里叶神经算子
- **Enhanced MLP 1D/2D/3D** - 增强版多层感知机
- **Enhanced PINN 1D/2D/3D** - 增强版物理信息神经网络
- **Enhanced UNet 1D/2D/3D** - 增强版U-Net架构

## 快速开始

### 1. 基本导入

```python
from unified_model_factory import get_global_factory
from unified_config_manager import get_global_config_manager
from trainer_compatibility_adapter import get_global_adapter

# 获取全局实例
model_factory = get_global_factory()
config_manager = get_global_config_manager()
compatibility_adapter = get_global_adapter()
```

### 2. 创建模型

#### 方法一：直接使用模型工厂

```python
# 配置Transformer模型
config = {
    'input_dim': 64,
    'output_dim': 32,
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 6,
    'attention_type': 'sge',  # 可选: 'standard', 'sge', 'cross'
    'pe_type': 'learnable_1d',
    'max_time_steps': 1,
    'seq_len': 32
}

# 创建模型
model = model_factory.create_model('enhanced_transformer_1d', config)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
```

#### 方法二：使用兼容性适配器

```python
# 使用trainer格式的配置
trainer_config = {
    'model': {
        'input_dim': 64,
        'output_dim': 32,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'attention_type': 'sge'
    },
    'training': {
        'epochs': 100,
        'learning_rate': 1e-3,
        'batch_size': 32
    }
}

# 验证配置
is_valid, errors = compatibility_adapter.validate_trainer_config(trainer_config)
if is_valid:
    model = compatibility_adapter.create_model_from_trainer_config(trainer_config)
else:
    print(f"配置错误: {errors}")
```

### 3. 配置管理

#### 获取配置模板

```python
# 获取模型配置模板
model_template = config_manager.get_config_template(['model'])
print("模型配置模板:")
for key, value in model_template['model'].items():
    print(f"  {key}: {value}")

# 获取完整配置模板
full_template = config_manager.get_config_template([
    'model', 'training', 'data', 'loss', 'optimizer'
])
```

#### 配置验证和合并

```python
# 验证配置
user_config = {
    'model': {
        'input_dim': 128,
        'd_model': 512
    }
}

validated_config = config_manager.validate_full_config(user_config)

# 合并配置
base_config = {'model': {'num_heads': 8}}
override_config = {'model': {'num_heads': 16, 'num_layers': 8}}
merged_config = config_manager.merge_configs(base_config, override_config)
```

## 详细API参考

### UnifiedModelFactory

#### 主要方法

```python
# 获取支持的模型列表
supported_models = model_factory.get_supported_models()
print(f"支持的模型: {supported_models}")

# 获取模型兼容性信息
compatibility = model_factory.get_compatibility_info()
for model_type, info in compatibility.items():
    print(f"{model_type}: 维度 {info['dimensions']}, 特性 {info['features']}")

# 按维度列出模型
models_1d = model_factory.list_models_by_dimension('1d')
models_2d = model_factory.list_models_by_dimension('2d')
models_3d = model_factory.list_models_by_dimension('3d')

# 获取配置模板
template = model_factory.get_config_template('enhanced_transformer_1d')
```

#### 模型创建参数

**Transformer模型参数:**
- `input_dim`: 输入维度
- `output_dim`: 输出维度
- `d_model`: 模型隐藏维度
- `num_heads`: 注意力头数
- `num_layers`: 层数
- `attention_type`: 注意力类型 ('standard', 'sge', 'cross')
- `pe_type`: 位置编码类型 ('learnable_1d', 'sinusoidal', 'none')
- `max_time_steps`: 最大时间步数
- `seq_len`: 序列长度
- `dropout`: Dropout率 (默认: 0.1)

**FNO模型参数:**
- `input_dim`: 输入维度
- `output_dim`: 输出维度
- `modes`: 傅里叶模式数
- `width`: 网络宽度
- `num_layers`: 层数

**MLP模型参数:**
- `input_dim`: 输入维度
- `output_dim`: 输出维度
- `hidden_dims`: 隐藏层维度列表
- `activation`: 激活函数 ('relu', 'gelu', 'tanh')
- `dropout`: Dropout率

### UnifiedConfigManager

#### 配置类型

```python
from unified_config_manager import ConfigType

# 可用的配置类型
config_types = [
    ConfigType.MODEL,
    ConfigType.TRAINING,
    ConfigType.DATA,
    ConfigType.LOSS,
    ConfigType.OPTIMIZER,
    ConfigType.SCHEDULER,
    ConfigType.DEVICE,
    ConfigType.MIXED_PRECISION,
    ConfigType.SVD_PROJECTION,
    ConfigType.DOWNSAMPLING,
    ConfigType.VISUALIZATION,
    ConfigType.LOGGING
]
```

#### 配置验证

```python
# 验证单个配置部分
model_config = {'input_dim': 64, 'd_model': 256}
validated = config_manager.validate_config(model_config, ConfigType.MODEL)

# 验证完整配置
full_config = {
    'model': model_config,
    'training': {'epochs': 100, 'learning_rate': 1e-3}
}
validated_full = config_manager.validate_full_config(full_config)
```

### TrainerCompatibilityAdapter

#### 主要功能

```python
# 获取模型信息
model_info = compatibility_adapter.get_model_info()
print(f"可用模型: {model_info['available_models']}")
print(f"默认配置: {model_info['default_configs']}")

# 增强配置
enhanced_config = compatibility_adapter.enhance_trainer_config(trainer_config)

# 从trainer配置创建模型
model = compatibility_adapter.create_model_from_trainer_config(trainer_config)
```

## 高级用法示例

### 1. 批量模型比较

```python
import torch
import time

def compare_models():
    """比较不同模型的性能"""
    models_to_test = [
        ('enhanced_transformer_1d', {
            'input_dim': 256, 'output_dim': 128,
            'd_model': 256, 'num_heads': 8, 'num_layers': 4
        }),
        ('enhanced_fno_1d', {
            'input_dim': 256, 'output_dim': 128,
            'modes': 16, 'width': 64, 'num_layers': 4
        }),
        ('enhanced_mlp_1d', {
            'input_dim': 256, 'output_dim': 128,
            'hidden_dims': [512, 1024, 512]
        })
    ]
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_type, config in models_to_test:
        try:
            # 创建模型
            model = model_factory.create_model(model_type, config)
            model = model.to(device)
            model.eval()
            
            # 测试性能
            test_input = torch.randn(32, config['input_dim']).to(device)
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            # 计时
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    output = model(test_input)
            end_time = time.time()
            
            results.append({
                'model_type': model_type,
                'parameters': sum(p.numel() for p in model.parameters()),
                'inference_time': (end_time - start_time) / 100,
                'memory_usage': torch.cuda.memory_allocated() if device.type == 'cuda' else 0
            })
            
        except Exception as e:
            print(f"模型 {model_type} 测试失败: {e}")
    
    return results

# 运行比较
comparison_results = compare_models()
for result in comparison_results:
    print(f"{result['model_type']}: {result['parameters']} 参数, "
          f"{result['inference_time']:.4f}s 推理时间")
```

### 2. 动态配置调整

```python
def dynamic_model_scaling():
    """根据数据大小动态调整模型配置"""
    data_sizes = [64, 128, 256, 512]
    
    for data_size in data_sizes:
        # 根据数据大小调整配置
        if data_size <= 64:
            config = {
                'input_dim': data_size,
                'output_dim': data_size // 2,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2
            }
        elif data_size <= 256:
            config = {
                'input_dim': data_size,
                'output_dim': data_size // 2,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4
            }
        else:
            config = {
                'input_dim': data_size,
                'output_dim': data_size // 2,
                'd_model': 512,
                'num_heads': 16,
                'num_layers': 6
            }
        
        # 验证配置
        validated_config = config_manager.validate_config(config, ConfigType.MODEL)
        
        # 创建模型
        model = model_factory.create_model('enhanced_transformer_1d', validated_config)
        print(f"数据大小 {data_size}: 模型参数 {sum(p.numel() for p in model.parameters())}")

# 运行动态缩放
dynamic_model_scaling()
```

### 3. 自定义训练流程

```python
import torch.nn as nn
import torch.optim as optim

def custom_training_pipeline():
    """自定义训练流程示例"""
    # 配置
    config = {
        'model': {
            'input_dim': 128,
            'output_dim': 64,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'attention_type': 'sge'
        },
        'training': {
            'epochs': 50,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'weight_decay': 1e-4
        },
        'loss': {
            'type': 'mse',
            'reduction': 'mean'
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4
        }
    }
    
    # 验证配置
    is_valid, errors = compatibility_adapter.validate_trainer_config(config)
    if not is_valid:
        print(f"配置错误: {errors}")
        return
    
    # 创建模型
    model = compatibility_adapter.create_model_from_trainer_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['learning_rate'],
                          weight_decay=config['training']['weight_decay'])
    criterion = nn.MSELoss()
    
    # 模拟训练数据
    batch_size = config['training']['batch_size']
    input_dim = config['model']['input_dim']
    output_dim = config['model']['output_dim']
    
    # 训练循环
    model.train()
    for epoch in range(config['training']['epochs']):
        # 模拟批次数据
        inputs = torch.randn(batch_size, input_dim).to(device)
        targets = torch.randn(batch_size, output_dim).to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    print("训练完成!")
    return model

# 运行自定义训练
trained_model = custom_training_pipeline()
```

## 故障排除

### 常见问题

1. **模型创建失败**
   ```python
   # 检查支持的模型类型
   supported = model_factory.get_supported_models()
   print(f"支持的模型: {supported}")
   
   # 检查配置是否有效
   try:
       model = model_factory.create_model(model_type, config)
   except Exception as e:
       print(f"创建失败: {e}")
       # 获取配置模板
       template = model_factory.get_config_template(model_type)
       print(f"正确的配置模板: {template}")
   ```

2. **配置验证错误**
   ```python
   # 详细的配置验证
   try:
       validated = config_manager.validate_full_config(config)
   except ValueError as e:
       print(f"配置验证失败: {e}")
       # 获取配置模板
       template = config_manager.get_config_template(['model', 'training'])
       print(f"配置模板: {template}")
   ```

3. **设备兼容性问题**
   ```python
   # 检查设备可用性
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"使用设备: {device}")
   
   # 确保模型和数据在同一设备
   model = model.to(device)
   inputs = inputs.to(device)
   ```

### 性能优化建议

1. **模型大小优化**
   - 根据数据大小选择合适的模型配置
   - 使用较小的`d_model`和`num_layers`进行快速原型设计
   - 考虑使用混合精度训练

2. **内存优化**
   - 使用梯度累积减少批次大小
   - 启用梯度检查点
   - 定期清理GPU内存

3. **训练加速**
   - 使用多GPU训练
   - 启用编译优化
   - 使用适当的数据加载器

## 扩展和自定义

### 添加新模型类型

要添加新的模型类型，需要：

1. 在相应的模型文件中实现模型类
2. 在`UnifiedModelFactory`中注册新模型
3. 在`UnifiedConfigManager`中添加配置模式
4. 更新兼容性适配器

### 自定义配置验证

```python
# 扩展配置管理器
class CustomConfigManager(UnifiedConfigManager):
    def validate_custom_config(self, config):
        # 自定义验证逻辑
        pass
```

## 版本信息

- **版本**: 1.0.0
- **Python要求**: >= 3.7
- **PyTorch要求**: >= 1.8.0
- **支持的CUDA版本**: 10.2, 11.0+

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

---

如有问题或建议，请提交Issue或Pull Request。