import os
import sys
import torch
import pytest

# Ensure project root is on sys.path so that package-style imports work when running pytest from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modify_multi_attention.mymodels.transformer import CustomDecoderLayer  # noqa: E402


def test_decoder_cross_attention_is_memory_sensitive():
    torch.manual_seed(42)

    # Model hyperparams (keep it small and deterministic)
    B, L, D, H = 2, 16, 64, 8

    # Instantiate a decoder layer with standard relative attention for self-attn
    # Disable dropout and FiLM gating to isolate cross-attn behavior
    layer = CustomDecoderLayer(
        d_model=D,
        num_heads=H,
        dim_feedforward=4 * D,
        dropout=0.0,
        attention_type="relative",
        seq_len=L,
        input_hw=None,
        use_memory_film=False,
    )
    layer.eval()

    # Create a base tgt and two different memory tensors
    tgt = torch.randn(B, L, D)
    # memory_1 intentionally equals tgt to detect the bug of ignoring memory and using tgt as K/V
    memory_1 = tgt.detach().clone()
    # memory_2 is different enough from tgt
    memory_2 = torch.randn(B, L, D) * 1.7 + 0.3

    with torch.no_grad():
        out1 = layer(tgt, memory_1)
        out2 = layer(tgt, memory_2)

    # If cross-attn really uses provided memory for K/V, out1 and out2 should differ noticeably
    diff = (out1 - out2).abs().mean().item()

    # Threshold chosen to be robust across seeds; if memory is ignored (e.g., using tgt for K/V), diff ~ 0
    assert diff > 1e-4, f"Cross-attn appears insensitive to memory; |out1-out2| mean={diff:.6e}"


if __name__ == "__main__":
    # Allow running as a script for quick local checks
    pytest.main([__file__, "-q"])