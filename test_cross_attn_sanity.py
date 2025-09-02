import torch
import sys


def run_test_base():
    # Lazy import to avoid heavy import time if torch not available
    from modify_multi_attention.mymodels.transformer import CustomDecoderLayer

    torch.manual_seed(0)
    device = torch.device("cpu")

    # Use square length to avoid padding ambiguity
    B, H, W = 2, 7, 7
    L, D, heads = H * W, 64, 8

    # Instantiate decoder layer with explicit grid (relative attention, no FiLM gating)
    layer = CustomDecoderLayer(d_model=D, num_heads=heads, seq_len=L, input_hw=(H, W)).to(device)
    layer.eval()  # disable dropout

    # Construct tgt fixed, memory varies
    tgt = torch.randn(B, L, D, device=device)
    memory_zero = torch.zeros(B, L, D, device=device)
    memory_rand = torch.randn(B, L, D, device=device) * 3.0

    with torch.no_grad():
        out_zero = layer(tgt.clone(), memory_zero)
        out_rand = layer(tgt.clone(), memory_rand)

    diff = (out_zero - out_rand).pow(2).mean().item()
    print(f"[Base] Cross-attention sensitivity L2-mean diff: {diff:.6e}")
    # Threshold chosen to be robust; if cross-attn ignores memory, diff ~ 0
    return diff > 1e-7


def run_test_cnn_like(attn_type: str = "se"):
    # Lazy import to avoid heavy import time if torch not available
    from modify_multi_attention.mymodels.transformer import CustomDecoderLayer

    torch.manual_seed(0)
    device = torch.device("cpu")

    # Use square length to avoid padding ambiguity
    B, H, W = 2, 7, 7
    L, D, heads = H * W, 64, 8

    # Instantiate decoder layer with CNN-like attention to trigger FiLM gating
    layer = CustomDecoderLayer(d_model=D, num_heads=heads, seq_len=L, input_hw=(H, W), attention_type=attn_type).to(device)
    layer.eval()  # disable dropout

    # Construct tgt fixed, memory varies
    tgt = torch.randn(B, L, D, device=device)
    memory_zero = torch.zeros(B, L, D, device=device)
    memory_rand = torch.randn(B, L, D, device=device) * 3.0

    with torch.no_grad():
        out_zero = layer(tgt.clone(), memory_zero)
        out_rand = layer(tgt.clone(), memory_rand)

    diff = (out_zero - out_rand).pow(2).mean().item()
    print(f"[CNN-like:{attn_type}] Cross-attention sensitivity L2-mean diff: {diff:.6e}")
    # Threshold chosen to be robust; if FiLM or cross-attn ignores memory, diff ~ 0
    return diff > 1e-7


def run_test_output_heads():
    """
    Sanity check for TransformerFlowReconstructionModel output heads:
    - global head should output shape [B, 1]
    - per-token head should output shape [B, L * C]
    """
    from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel

    torch.manual_seed(0)
    device = torch.device("cpu")

    B, H, W = 2, 4, 4
    L = H * W
    in_dim = L

    # Global head
    m1 = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=1,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=H,
        input_hw=(H, W),
        pe_type='learnable_1d',
        output_head_type='global'
    ).to(device)
    m1.eval()

    x = torch.randn(B, in_dim, device=device)
    t = torch.randint(0, 10, (B,), device=device)
    with torch.no_grad():
        y1 = m1(x, t)
    print('[Head] Global shape:', tuple(y1.shape))
    ok_global = tuple(y1.shape) == (B, 1)

    # Per-token head
    per_ch = 3
    m2 = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=L * per_ch,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=H,
        input_hw=(H, W),
        pe_type='learnable_2d',
        output_head_type='per_token',
        out_channels_per_token=per_ch
    ).to(device)
    m2.eval()

    with torch.no_grad():
        y2 = m2(x, t)
    print('[Head] Per-token shape:', tuple(y2.shape))
    ok_token = tuple(y2.shape) == (B, L * per_ch)

    return ok_global and ok_token


if __name__ == "__main__":
    ok1 = run_test_base()
    ok2 = run_test_cnn_like("se")
    ok3 = run_test_output_heads()

    all_ok = ok1 and ok2 and ok3
    if not all_ok:
        print("[FAIL] One or more tests failed. Decoder may not be sensitive to memory changes or output heads mismatch.")
        sys.exit(1)
    else:
        print("[OK] Decoder sensitivity and output head shapes are correct across tested paths.")
        sys.exit(0)