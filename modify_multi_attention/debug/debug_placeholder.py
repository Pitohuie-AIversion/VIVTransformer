import torch
import torch.nn as nn
from models.enhanced_transformer import EnhancedTransformer2d

print("=== 测试占位符类参数传递 ===")

# 创建EnhancedTransformer2d实例
enhanced_model = EnhancedTransformer2d(
    input_channels=1,
    output_channels=1,
    d_model=256,  # 使用run_crop_model_test.py中的默认值
    num_heads=4,
    num_layers=3,
    input_resolution=(32, 32),
    output_resolution=(128, 128)
)

print(f"EnhancedTransformer2d内部transformer类型: {type(enhanced_model.transformer)}")
print(f"EnhancedTransformer2d内部transformer的fc层: {enhanced_model.transformer.fc}")
print(f"fc层输入维度: {enhanced_model.transformer.fc.in_features}")
print(f"fc层输出维度: {enhanced_model.transformer.fc.out_features}")

# 测试输入
test_input = torch.randn(1, 1024)  # 32*32*1 = 1024
time_steps = torch.zeros(1, 1, dtype=torch.long)

print(f"\n测试输入形状: {test_input.shape}")
print(f"时间步形状: {time_steps.shape}")

try:
    output = enhanced_model.transformer(test_input, time_steps)
    print(f"输出形状: {output.shape}")
    print("✅ 占位符类工作正常")
except Exception as e:
    print(f"❌ 占位符类出错: {e}")
    print(f"错误类型: {type(e)}")