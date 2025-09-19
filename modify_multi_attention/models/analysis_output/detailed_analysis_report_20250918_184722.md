
# 模型横向对比详细分析报告

**生成时间**: 2025-09-18 18:47:22
**数据源**: unified_comparison_results_20250918_184500.yaml

## 1. 总体概况

- **总测试数**: 52
- **成功测试数**: 0
- **失败测试数**: 52
- **总体成功率**: 0.0%

## 2. 模型类型分析

### 2.1 各模型类型成功率

| 模型类型 | 测试数量 | 成功数量 | 成功率 | 包含模型 |
|---------|---------|---------|--------|----------|
| fno | 8 | 0 | 0.0% | fno_2d, fno_1d |
| unet | 8 | 0 | 0.0% | unet_1d, unet_2d |
| mlp | 8 | 0 | 0.0% | mlp_1d, mlp_2d |
| pinn | 8 | 0 | 0.0% | pinn_2d, pinn_1d |
| transformer | 20 | 0 | 0.0% | transformer_medium, transformer_1d, transformer_light 等5个 |

### 2.2 数据集适配性分析

| 数据集 | 测试数量 | 成功数量 | 成功率 |
|--------|---------|---------|--------|
| time_series_short | 13 | 0 | 0.0% |
| time_series_long | 13 | 0 | 0.0% |
| image_small | 13 | 0 | 0.0% |
| physics_diffusion | 13 | 0 | 0.0% |

## 3. 错误分析

### 3.1 主要错误类型


**张量尺寸不匹配** (21次)
- 影响模型: fno_2d, mlp_1d, unet_2d, pinn_1d, fno_1d 等7个模型
- 典型错误示例: The size of tensor a (128) must match the size of tensor b (100) at non-singleton dimension 1

**解包错误** (3次)
- 影响模型: fno_2d
- 典型错误示例: not enough values to unpack (expected 4, got 2)

**通道数错误** (3次)
- 影响模型: unet_1d
- 典型错误示例: Given groups=1, weight of size [32, 1, 3], expected input[1, 4, 100] to have 1 channels, but got 4 channels instead

**维度错误** (4次)
- 影响模型: unet_2d, mlp_1d
- 典型错误示例: Input and output must have the same number of spatial dimensions, but got input with spatial dimensions of [] and output size of (100, 100). Please provide input tensor in (N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format.

**参数错误** (20次)
- 影响模型: transformer_medium, transformer_1d, transformer_light, transformer_2d, transformer_micro
- 典型错误示例: EnhancedTransformer1d.forward() takes 2 positional arguments but 3 were given

**其他错误** (1次)
- 影响模型: unet_1d
- 典型错误示例: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [4, 1, 32, 32]

### 3.2 模型特定问题

- **fno_1d**: 张量尺寸不匹配
- **fno_2d**: 解包错误, 张量尺寸不匹配
- **unet_1d**: 通道数错误, 其他错误
- **unet_2d**: 维度错误, 张量尺寸不匹配
- **mlp_1d**: 维度错误, 张量尺寸不匹配
- **mlp_2d**: 张量尺寸不匹配
- **pinn_1d**: 张量尺寸不匹配
- **pinn_2d**: 张量尺寸不匹配
- **transformer_micro**: 参数错误
- **transformer_light**: 参数错误
- **transformer_medium**: 参数错误
- **transformer_1d**: 参数错误
- **transformer_2d**: 参数错误

## 5. 问题总结与建议

### 5.1 主要问题

1. **张量尺寸不匹配**: 这是最常见的问题，主要原因是模型输入输出尺寸配置与数据集不匹配
2. **通道数错误**: 模型期望的输入通道数与实际数据不符
3. **参数传递错误**: 某些模型的forward方法参数定义与调用不一致
4. **维度处理错误**: 1D和2D模型在处理不同维度数据时出现问题

### 5.2 改进建议

1. **统一接口设计**: 为所有模型实现统一的输入输出接口
2. **动态尺寸适配**: 实现自动的尺寸适配机制
3. **参数验证**: 在模型创建时验证参数的有效性
4. **错误处理**: 增加更详细的错误信息和恢复机制

### 5.3 下一步工作

1. 修复识别出的接口不一致问题
2. 实现自适应的尺寸处理机制
3. 优化模型配置参数
4. 增加更多的测试用例和边界条件检查

---

**报告生成完成**: 2025-09-18 18:47:22
