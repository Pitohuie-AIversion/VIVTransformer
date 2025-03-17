import torch
from transformer import TransformerFlowReconstructionModel

# 定义模型参数（根据你实际情况填写）
input_dim = 128  # 示例输入维度
output_dim = 256  # 示例输出维度
num_heads = 8
num_layers = 6
d_model = 512
max_time_steps = 100

# 初始化模型
model = TransformerFlowReconstructionModel(
    input_dim=input_dim,
    output_dim=output_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    d_model=d_model,
    max_time_steps=max_time_steps
)

model.eval()  # 设置为评估模式

# 创建示例输入数据
batch_size = 1
example_x_in = torch.randn(batch_size, input_dim)
example_x_time_steps = torch.randint(0, max_time_steps, (batch_size,))

# 导出到ONNX
torch.onnx.export(
    model,
    (example_x_in, example_x_time_steps),  # 模型输入示例
    "transformer_flow_model.onnx",
    input_names=["x_in_pressures_flat", "x_time_steps"],
    output_names=["out_pressure_flat_pred"],
    dynamic_axes={
        "x_in_pressures_flat": {0: "batch_size"},
        "x_time_steps": {0: "batch_size"},
        "out_pressure_flat_pred": {0: "batch_size"}
    },
    opset_version=14
)

print("Model successfully exported to ONNX format!")
