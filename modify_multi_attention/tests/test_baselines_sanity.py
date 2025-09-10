import torch
import torch.nn as nn
from modify_multi_attention.mymodels.baselines import (
    LinearBaselineModel,
    MLPBaselineModel,
    CNNBaselineModel,
    UNetBaselineModel,
    CNNResizeBaselineModel,
    UNetResizeBaselineModel,
    PODLSEModel,
)


def assert_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        raise AssertionError(f"{name} contains non-finite values: NaN={n_nan}, Inf={n_inf}, shape={tuple(t.shape)}")


def run_sanity_checks():
    torch.manual_seed(0)

    B = 4
    H_in, W_in = 20, 20
    input_dim = H_in * W_in  # 400
    H_out, W_out = 32, 32
    output_dim_same = input_dim  # 400
    output_dim_square = H_out * W_out  # 1024

    # Inputs
    x_in = torch.randn(B, input_dim, dtype=torch.float32)
    t_steps = torch.randint(low=0, high=100, size=(B,), dtype=torch.long)

    # 1) LinearBaselineModel
    model = LinearBaselineModel(input_dim=input_dim, output_dim=output_dim_square, max_time_steps=100, time_encoding="none")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_square), f"LinearBaselineModel shape {y.shape} != {(B, output_dim_square)}"
    assert_finite(y, "LinearBaselineModel output")

    # 2) MLPBaselineModel
    model = MLPBaselineModel(input_dim=input_dim, output_dim=output_dim_square, hidden_dim=128, max_time_steps=100, time_encoding="mlp")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_square)
    assert_finite(y, "MLPBaselineModel output")

    # 3) CNNBaselineModel (same output_dim)
    model = CNNBaselineModel(input_dim=input_dim, output_dim=output_dim_same, input_hw=(H_in, W_in), max_time_steps=100, time_encoding="mlp")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_same)
    assert_finite(y, "CNNBaselineModel(same) output")

    # 4) CNNBaselineModel (square but different output_dim)
    model = CNNBaselineModel(input_dim=input_dim, output_dim=output_dim_square, input_hw=(H_in, W_in), max_time_steps=100, time_encoding="mlp")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_square)
    assert_finite(y, "CNNBaselineModel(resize) output")

    # 5) UNetBaselineModel (same)
    model = UNetBaselineModel(input_dim=input_dim, output_dim=output_dim_same, input_hw=(H_in, W_in), base_ch=16, max_time_steps=100, time_encoding="mlp")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_same)
    assert_finite(y, "UNetBaselineModel(same) output")

    # 6) UNetBaselineModel (square but different output_dim)
    model = UNetBaselineModel(input_dim=input_dim, output_dim=output_dim_square, input_hw=(H_in, W_in), base_ch=16, max_time_steps=100, time_encoding="mlp")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_square)
    assert_finite(y, "UNetBaselineModel(resize) output")

    # 7) CNNResizeBaselineModel
    model = CNNResizeBaselineModel(input_dim=input_dim, output_dim=output_dim_square, input_hw=(H_in, W_in), output_hw=(H_out, W_out), max_time_steps=100, time_encoding="mlp")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_square)
    assert_finite(y, "CNNResizeBaselineModel output")

    # 8) UNetResizeBaselineModel
    model = UNetResizeBaselineModel(input_dim=input_dim, output_dim=output_dim_square, input_hw=(H_in, W_in), output_hw=(H_out, W_out), base_ch=16, max_time_steps=100, time_encoding="mlp")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_square)
    assert_finite(y, "UNetResizeBaselineModel output")

    # 9) PODLSEModel
    k = 8
    # Create an orthonormal-ish basis U: [output_dim_square, k]
    U_rand = torch.randn(output_dim_square, output_dim_square)
    Q, _ = torch.linalg.qr(U_rand)
    U = Q[:, :k].contiguous()
    model = PODLSEModel(input_dim=input_dim, output_dim=output_dim_square, pod_basis=U, max_time_steps=100, d_model=128, time_encoding="concat")
    y = model(x_in, t_steps)
    assert y.shape == (B, output_dim_square)
    assert_finite(y, "PODLSEModel output")

    print("[OK] All baselines forward sanity checks passed (outputs finite, shapes correct).")


if __name__ == "__main__":
    run_sanity_checks()
