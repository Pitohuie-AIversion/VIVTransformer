import os
import sys
import numpy as np
import onnxruntime as ort


def verify_onnx(onnx_path: str, out_channels_per_token: int = 3):
    if not os.path.exists(onnx_path):
        print(f"[SKIP] {onnx_path} not found")
        return

    print(f"[LOAD] {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # CPU is enough
    inputs = sess.get_inputs()
    assert len(inputs) == 2, f"Expected 2 inputs, got {len(inputs)}"
    x_name = inputs[0].name
    t_name = inputs[1].name

    # Try to infer input_dim from model input shape
    # Expected shape: [batch, input_dim] for x; [batch] for t
    input_shape = inputs[0].shape
    if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 2):
        raise RuntimeError(f"Unexpected input shape for {x_name}: {input_shape}")
    input_dim = int(input_shape[1])

    batch = 2
    x = np.random.randn(batch, input_dim).astype(np.float32)
    # time steps is int64 in ONNX
    t = np.random.randint(0, 100, size=(batch,), dtype=np.int64)

    y_list = sess.run(None, {x_name: x, t_name: t})
    assert len(y_list) == 1, f"Expected 1 output, got {len(y_list)}"
    y = y_list[0]
    print(f"[RUN] output shape = {y.shape}")

    # Check shape according to filename convention
    if "per_token" in os.path.basename(onnx_path):
        expected_dim = input_dim * out_channels_per_token
    else:
        expected_dim = 256  # must be aligned with export_to_onnx.py default

    assert y.shape == (batch, expected_dim), \
        f"Unexpected output shape: {y.shape}, expected {(batch, expected_dim)}"
    print(f"[OK] {os.path.basename(onnx_path)} passes shape check: {(batch, expected_dim)}")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    global_path = os.path.join(base_dir, "transformer_flow_model_global.onnx")
    per_token_path = os.path.join(base_dir, "transformer_flow_model_per_token.onnx")

    verify_onnx(global_path, out_channels_per_token=3)
    verify_onnx(per_token_path, out_channels_per_token=3)