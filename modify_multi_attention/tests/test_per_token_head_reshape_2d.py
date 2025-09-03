import os
import sys
import torch
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel  # noqa: E402


def test_per_token_output_to_2d_success():
    torch.manual_seed(0)
    B, H, W, C = 2, 3, 5, 4
    L = H * W
    in_dim = L
    out_dim = L * C

    m = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=out_dim,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=L,
        input_hw=(H, W),
        pe_type='learnable_2d',
        output_head_type='per_token',
        out_channels_per_token=C,
        time_encoding='embedding',
        use_memory_film=False,
    )
    m.eval()

    x = torch.randn(B, in_dim)
    t = torch.randint(0, 10, (B,))
    with torch.no_grad():
        y = m(x, t)
    y_2d = m.per_token_output_to_2d(y)
    assert y_2d.shape == (B, H, W, C)


def test_per_token_output_to_2d_raises_on_mismatch():
    torch.manual_seed(0)
    B, H, W, C = 1, 2, 2, 3
    L = H * W
    in_dim = L
    out_dim = L * C

    m = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=out_dim,
        num_heads=2,
        num_layers=1,
        d_model=16,
        max_time_steps=10,
        attention_type='relative',
        seq_len=L,
        input_hw=(H, W),
        pe_type='learnable_2d',
        output_head_type='per_token',
        out_channels_per_token=C,
        time_encoding='embedding',
        use_memory_film=False,
    )
    m.eval()

    x = torch.randn(B, in_dim)
    t = torch.randint(0, 10, (B,))
    with torch.no_grad():
        y = m(x, t)
    # Tamper the tensor to mismatch
    y_bad = y[:, :-1]
    with pytest.raises(AssertionError):
        _ = m.per_token_output_to_2d(y_bad)