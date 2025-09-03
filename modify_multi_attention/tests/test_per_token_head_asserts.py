import os
import sys
import torch
import pytest

# Ensure project root on sys.path so that package-style imports work when running pytest from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel  # noqa: E402


def test_per_token_head_valid_output_shape():
    torch.manual_seed(0)
    B, H, W, C = 2, 4, 4, 3
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
        seq_len=H,
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
    assert tuple(y.shape) == (B, out_dim)


def test_per_token_head_mismatch_fallback_to_global():
    torch.manual_seed(0)
    B, H, W, C_decl, out_dim = 2, 4, 4, 2, 7  # declare C=2, but output_dim is 7, so L*C != output_dim -> fallback
    L = H * W
    in_dim = L

    m = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=out_dim,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=H,
        input_hw=(H, W),
        pe_type='learnable_2d',
        output_head_type='per_token',
        out_channels_per_token=C_decl,
        time_encoding='embedding',
        use_memory_film=False,
    )
    m.eval()

    x = torch.randn(B, in_dim)
    t = torch.randint(0, 10, (B,))
    with torch.no_grad():
        y = m(x, t)
    # Should fallback to global head with output_dim features
    assert tuple(y.shape) == (B, out_dim)