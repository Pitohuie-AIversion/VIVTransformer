import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from .mymodels.baselines import (
    LinearBaselineModel,
    MLPBaselineModel,
    CNNBaselineModel,
    UNetBaselineModel,
    CNNResizeBaselineModel,
    UNetResizeBaselineModel,
    PODLSEModel,
)


def run_case(model_ctor, kwargs, name, B=4, max_time_steps=50):
    input_dim = kwargs.get("input_dim")
    output_dim = kwargs.get("output_dim")
    x = torch.randn(B, input_dim)
    x_t = torch.randint(low=0, high=max_time_steps, size=(B,))
    m = model_ctor(**kwargs)
    with torch.no_grad():
        y = m(x, x_t)
    assert y.shape == (B, output_dim), f"{name}: bad shape {y.shape} expected {(B, output_dim)}"
    assert torch.isfinite(y).all(), f"{name}: got NaN/Inf in output"
    print(f"[PASS] {name}: output shape {tuple(y.shape)}")


def main():
    B = 4
    max_time_steps = 50
    H, W = 64, 64
    input_dim = H * W

    # 1) CNN baseline 64->64
    run_case(
        CNNBaselineModel,
        dict(input_dim=input_dim, output_dim=input_dim, input_hw=(H, W), max_time_steps=max_time_steps, time_encoding="mlp"),
        name="CNNBaseline 64->64",
        B=B, max_time_steps=max_time_steps,
    )

    # 2) CNN baseline 64->128 (square upsample by internal interpolate)
    run_case(
        CNNBaselineModel,
        dict(input_dim=input_dim, output_dim=128*128, input_hw=(H, W), max_time_steps=max_time_steps, time_encoding="mlp"),
        name="CNNBaseline 64->128",
        B=B, max_time_steps=max_time_steps,
    )

    # 3) UNet baseline 64->64
    run_case(
        UNetBaselineModel,
        dict(input_dim=input_dim, output_dim=input_dim, input_hw=(H, W), base_ch=8, max_time_steps=max_time_steps, time_encoding="mlp"),
        name="UNetBaseline 64->64",
        B=B, max_time_steps=max_time_steps,
    )

    # 4) UNet baseline 64->128 (square upsample by internal interpolate)
    run_case(
        UNetBaselineModel,
        dict(input_dim=input_dim, output_dim=128*128, input_hw=(H, W), base_ch=8, max_time_steps=max_time_steps, time_encoding="mlp"),
        name="UNetBaseline 64->128",
        B=B, max_time_steps=max_time_steps,
    )

    # 5) CNN resize 64->(96x80)
    run_case(
        CNNResizeBaselineModel,
        dict(input_dim=input_dim, output_dim=96*80, input_hw=(H, W), output_hw=(96, 80), max_time_steps=max_time_steps, time_encoding="mlp"),
        name="CNNResize 64->96x80",
        B=B, max_time_steps=max_time_steps,
    )

    # 6) UNet resize 64->(96x80)
    run_case(
        UNetResizeBaselineModel,
        dict(input_dim=input_dim, output_dim=96*80, input_hw=(H, W), output_hw=(96, 80), base_ch=8, max_time_steps=max_time_steps, time_encoding="mlp"),
        name="UNetResize 64->96x80",
        B=B, max_time_steps=max_time_steps,
    )

    # 7) Linear baseline
    run_case(
        LinearBaselineModel,
        dict(input_dim=input_dim, output_dim=5000, max_time_steps=max_time_steps, time_encoding="none", d_model=64),
        name="Linear",
        B=B, max_time_steps=max_time_steps,
    )

    # 8) MLP baseline
    run_case(
        MLPBaselineModel,
        dict(input_dim=input_dim, output_dim=5000, hidden_dim=128, max_time_steps=max_time_steps, time_encoding="none"),
        name="MLP",
        B=B, max_time_steps=max_time_steps,
    )

    # 9) PODLSE
    output_dim_pod = 2048
    k = 32
    pod_basis = torch.randn(output_dim_pod, k)
    run_case(
        PODLSEModel,
        dict(input_dim=input_dim, output_dim=output_dim_pod, pod_basis=pod_basis, max_time_steps=max_time_steps, d_model=128, time_encoding="concat"),
        name="PODLSE",
        B=B, max_time_steps=max_time_steps,
    )

    print("ALL BASELINE SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()