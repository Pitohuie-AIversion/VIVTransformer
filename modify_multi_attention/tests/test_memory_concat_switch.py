import os
import sys
import torch
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel  # noqa: E402


def test_memory_concat_enables_and_shapes():
    torch.manual_seed(0)
    B, L, D = 2, 6, 32
    in_dim = L
    out_dim = L  # use global head for simplicity

    m = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=out_dim,
        num_heads=2,
        num_layers=1,
        d_model=D,
        max_time_steps=10,
        attention_type='relative',
        seq_len=L,
        input_hw=None,
        pe_type='learnable_1d',
        output_head_type='global',
        time_encoding='embedding',
        use_memory_film=False,
        use_memory_concat=True,
    )
    m.eval()

    x = torch.randn(B, in_dim)
    t = torch.randint(0, 10, (B,))
    with torch.no_grad():
        y = m(x, t)
    assert y.shape == (B, out_dim)


def test_memory_concat_off_default_behavior():
    torch.manual_seed(0)
    B, L, D = 2, 6, 32
    in_dim = L
    out_dim = L

    m = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=out_dim,
        num_heads=2,
        num_layers=1,
        d_model=D,
        max_time_steps=10,
        attention_type='relative',
        seq_len=L,
        input_hw=None,
        pe_type='learnable_1d',
        output_head_type='global',
        time_encoding='embedding',
        use_memory_film=False,
        use_memory_concat=False,
    )
    m.eval()

    x = torch.randn(B, in_dim)
    t = torch.randint(0, 10, (B,))
    with torch.no_grad():
        y = m(x, t)
    assert y.shape == (B, out_dim)