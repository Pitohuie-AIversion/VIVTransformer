import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.enhanced_transformer import create_enhanced_transformer2d
    from mymodels.transformer import TransformerFlowReconstructionModel
    print("✅ 成功导入enhanced_transformer和TransformerFlowReconstructionModel")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试transformer维度
print("\n=== 测试Transformer维度 ===")

# 直接测试TransformerFlowReconstructionModel
print("\n--- 直接测试TransformerFlowReconstructionModel ---")
transformer_model = TransformerFlowReconstructionModel(
    input_dim=1024,
    output_dim=32,  # 期望输出32维
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

print(f"TransformerFlowReconstructionModel:")
print(f"  input_dim: {transformer_model.input_dim}")
print(f"  output_dim: {transformer_model.output_dim}")
print(f"  d_model: {transformer_model.d_model}")
print(f"  seq_len: {transformer_model.seq_len}")
if hasattr(transformer_model, 'fc_out'):
    print(f"  fc_out: {transformer_model.fc_out}")

# 测试直接调用
test_input = torch.randn(1, 1024)
time_steps = torch.zeros(1, 1, dtype=torch.long)
print(f"\n测试输入形状: {test_input.shape}")
print(f"时间步形状: {time_steps.shape}")

with torch.no_grad():
    try:
        direct_output = transformer_model(test_input, time_steps)
        print(f"TransformerFlowReconstructionModel直接输出形状: {direct_output.shape}")
    except Exception as e:
        print(f"❌ TransformerFlowReconstructionModel测试失败: {e}")
        import traceback
        traceback.print_exc()

# 创建EnhancedTransformer2d模型
print("\n--- 测试EnhancedTransformer2d ---")
model = create_enhanced_transformer2d(
    input_channels=1,
    output_channels=1,
    d_model=32,  # 从配置文件
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

print(f"EnhancedTransformer2d:")
print(f"  输入维度: {model.input_dim}")
print(f"  transformer输出维度: {model.transformer_output_dim}")
print(f"  最终输出维度: {model.final_output_dim}")
print(f"  实际输出维度: {model.output_dim}")
# print(f"  内部transformer的output_dim: {model.transformer.output_dim}")  # 这个属性不存在

# 创建测试输入
test_input = torch.randn(1, 32, 32, 1)  # [batch_size, H, W, C]
print(f"\n测试输入形状: {test_input.shape}")

# 测试transformer部分
with torch.no_grad():
    try:
        # 展平输入
        x_flat = test_input.view(1, -1)
        print(f"展平后输入形状: {x_flat.shape}")
        
        # 创建时间步
        time_steps = torch.zeros(1, 1, dtype=torch.long)
        print(f"时间步形状: {time_steps.shape}")
        
        # 通过transformer
        transformer_output = model.transformer(x_flat, time_steps)
        print(f"Transformer输出形状: {transformer_output.shape}")
        
        # 如果有upsampler，测试它
        if hasattr(model, 'upsampler'):
            print(f"\nUpsampler结构:")
            for i, layer in enumerate(model.upsampler):
                print(f"  层 {i}: {layer}")
            
            print(f"\n尝试通过upsampler...")
            print(f"输入到upsampler的形状: {transformer_output.shape}")
            upsampler_output = model.upsampler(transformer_output)
            print(f"Upsampler输出形状: {upsampler_output.shape}")
        
        # 测试完整前向传播
        full_output = model(test_input)
        print(f"\n完整模型输出形状: {full_output.shape}")
        
        print("\n✅ 所有测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()