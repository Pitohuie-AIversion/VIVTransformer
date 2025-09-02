import torch
import argparse
from .transformer import TransformerFlowReconstructionModel

# ===== 用户可配置区域 =====
# 定义模型基础参数（根据你实际情况填写）
input_dim = 128   # 示例输入维度，若表示 H*W，请确保为可分解的平方或提供 input_hw
num_heads = 8
num_layers = 6
d_model = 512
max_time_steps = 100

# 输出头配置：'global' 或 'per_token'
output_head_type = 'global'   # 改为 'per_token' 导出逐 token 头
out_channels_per_token = 3    # 仅在 per_token 有效，表示每个 token 的通道数 C

# ===== 命令行参数（可覆盖上面的默认值）=====
parser = argparse.ArgumentParser(description='Export TransformerFlowReconstructionModel to ONNX with configurable output head.')
parser.add_argument('--head', '--output-head-type', dest='output_head_type', choices=['global', 'per_token'], default=output_head_type, help='Output head type: global or per_token')
parser.add_argument('--out-channels-per-token', dest='out_channels_per_token', type=int, default=out_channels_per_token, help='C per token when head is per_token')
parser.add_argument('--input-dim', dest='input_dim', type=int, default=input_dim, help='Model input dimension (e.g., H*W)')
parser.add_argument('--num-heads', dest='num_heads', type=int, default=num_heads)
parser.add_argument('--num-layers', dest='num_layers', type=int, default=num_layers)
parser.add_argument('--d-model', dest='d_model', type=int, default=d_model)
parser.add_argument('--max-time-steps', dest='max_time_steps', type=int, default=max_time_steps)
args = parser.parse_args()

# 使用参数覆盖默认配置
input_dim = args.input_dim
num_heads = args.num_heads
num_layers = args.num_layers
d_model = args.d_model
max_time_steps = args.max_time_steps
output_head_type = args.output_head_type
out_channels_per_token = args.out_channels_per_token if output_head_type == 'per_token' else out_channels_per_token

# 根据输出头类型自动计算 output_dim
if output_head_type == 'per_token':
    # 当 per-token 时，output_dim = (H*W) * C，其中 L=H*W
    # 自动推断 input_hw；若不是完全平方数，则回退为 (seq_len, seq_len) 的默认设置
    if int(input_dim ** 0.5) ** 2 == input_dim:
        H = W = int(input_dim ** 0.5)
        output_dim = input_dim * out_channels_per_token
        seq_len = H  # 与当前实现对齐（内部使用 input_hw 推导 seq_len）
        input_hw = (H, W)
    else:
        # 若非平方，仍允许导出；内部会根据 input_hw/seq_len 做合理处理
        # 这里简化处理，仍将 output_dim 设为 input_dim * C
        output_dim = input_dim * out_channels_per_token
        seq_len = 32
        input_hw = None
else:
    # global 头仅需要最终标量或指定维度
    output_dim = 256  # 示例输出维度（可按需修改）
    seq_len = int(input_dim ** 0.5) if int(input_dim ** 0.5) ** 2 == input_dim else 32
    input_hw = (int(input_dim ** 0.5), int(input_dim ** 0.5)) if int(input_dim ** 0.5) ** 2 == input_dim else None

# ===== 初始化模型 =====
model = TransformerFlowReconstructionModel(
    input_dim=input_dim,
    output_dim=output_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    d_model=d_model,
    max_time_steps=max_time_steps,
    seq_len=seq_len,
    input_hw=input_hw,
    pe_type='learnable_1d' if output_head_type == 'global' else 'learnable_2d',
    output_head_type=output_head_type,
    out_channels_per_token=out_channels_per_token if output_head_type == 'per_token' else None
)

model.eval()  # 设置为评估模式

# ===== 创建示例输入数据 =====
batch_size = 1
example_x_in = torch.randn(batch_size, input_dim)
example_x_time_steps = torch.randint(0, max_time_steps, (batch_size,))

# 预跑一次，打印输出形状用于确认
with torch.no_grad():
    y = model(example_x_in, example_x_time_steps)
print(f"Model output shape: {tuple(y.shape)} (head={output_head_type})")

# ===== 导出到 ONNX =====
onnx_path = f"transformer_flow_model_{output_head_type}.onnx"
torch.onnx.export(
    model,
    (example_x_in, example_x_time_steps),  # 模型输入示例
    onnx_path,
    input_names=["x_in_pressures_flat", "x_time_steps"],
    output_names=["out_pressure_flat_pred"],
    dynamic_axes={
        "x_in_pressures_flat": {0: "batch_size"},
        "x_time_steps": {0: "batch_size"},
        "out_pressure_flat_pred": {0: "batch_size"}
    },
    opset_version=14
)

print(f"Model successfully exported to ONNX format: {onnx_path}")
