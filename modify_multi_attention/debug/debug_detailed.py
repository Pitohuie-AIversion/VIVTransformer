import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mymodels.transformer import TransformerFlowReconstructionModel
    from models.enhanced_transformer import create_enhanced_transformer2d
    print("✅ 成功导入TransformerFlowReconstructionModel和EnhancedTransformer2d")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

print("\n=== 比较两个TransformerFlowReconstructionModel实例 ===")

# 直接创建的实例
print("\n--- 直接创建的TransformerFlowReconstructionModel ---")
direct_transformer = TransformerFlowReconstructionModel(
    input_dim=1024,
    output_dim=32,
    d_model=32,
    num_heads=8,
    num_layers=6,
    seq_len=1024,
    input_hw=(32, 32),
    pe_type='learnable_2d',
    output_head_type='global',
    time_encoding='embedding',
    max_time_steps=100,
    attention_type='simplified_self',
    use_memory_film=False,
    use_memory_concat=False
)

print(f"直接创建的参数:")
print(f"  input_dim: {direct_transformer.input_dim}")
print(f"  output_dim: {direct_transformer.output_dim}")
print(f"  d_model: {direct_transformer.d_model}")
print(f"  seq_len: {direct_transformer.seq_len}")
print(f"  fc_out: {direct_transformer.fc_out}")

# 通过EnhancedTransformer2d创建的实例
print("\n--- 通过EnhancedTransformer2d创建的实例 ---")
enhanced_model = create_enhanced_transformer2d(
    input_channels=1,
    output_channels=1,
    d_model=32,
    num_heads=8,
    num_layers=6,
    input_resolution=(32, 32),
    output_resolution=(128, 128),
    attention_type='simplified_self',
    pe_type='learnable_2d',
    time_encoding='embedding',
    max_time_steps=100,
    use_upsampling=True
)

print(f"EnhancedTransformer2d参数:")
print(f"  input_dim: {enhanced_model.input_dim}")
print(f"  transformer_output_dim: {enhanced_model.transformer_output_dim}")
print(f"  final_output_dim: {enhanced_model.final_output_dim}")

print(f"EnhancedTransformer2d内部transformer类型: {type(enhanced_model.transformer)}")
print(f"EnhancedTransformer2d内部transformer属性:")
for attr in ['output_head_type', 'attention_type', 'd_model', 'output_dim', 'fc_out', 'token_head']:
    if hasattr(enhanced_model.transformer, attr):
        print(f"  {attr}: {getattr(enhanced_model.transformer, attr)}")
    else:
        print(f"  {attr}: NOT FOUND")

# 测试两个实例
test_input = torch.randn(1, 1024)
time_steps = torch.zeros(1, 1, dtype=torch.long)

print(f"\n=== 测试两个实例的输出 ===")
print(f"测试输入形状: {test_input.shape}")
print(f"时间步形状: {time_steps.shape}")

with torch.no_grad():
    # 测试直接创建的实例
    try:
        direct_output = direct_transformer(test_input, time_steps)
        print(f"\n直接创建的实例输出形状: {direct_output.shape}")
    except Exception as e:
        print(f"❌ 直接创建的实例测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试EnhancedTransformer2d内部的transformer
    try:
        enhanced_output = enhanced_model.transformer(test_input, time_steps)
        print(f"EnhancedTransformer2d内部transformer输出形状: {enhanced_output.shape}")
    except Exception as e:
        print(f"❌ EnhancedTransformer2d内部transformer测试失败: {e}")
        import traceback
        traceback.print_exc()

# 检查两个实例的fc_out层是否相同
print(f"\n=== 比较fc_out层 ===")
print(f"直接创建的fc_out: {direct_transformer.fc_out}")
print(f"EnhancedTransformer2d内部的fc_out: {enhanced_model.transformer.fc_out}")

# 检查两个实例的其他关键参数
print(f"\n=== 比较其他关键参数 ===")
print(f"直接创建的output_head_type: {getattr(direct_transformer, 'output_head_type', 'None')}")
print(f"EnhancedTransformer2d内部的output_head_type: {getattr(enhanced_model.transformer, 'output_head_type', 'None')}")

print(f"\n直接创建的attention_type: {getattr(direct_transformer, 'attention_type', 'None')}")
print(f"EnhancedTransformer2d内部的attention_type: {getattr(enhanced_model.transformer, 'attention_type', 'None')}")