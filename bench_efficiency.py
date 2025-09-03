import sitecustomize
import os
import time
import argparse
import torch

# 建议减少碎片并提升稳定性
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel


def parse_input_hw(s: str | None):
    if not s:
        return None
    s = s.lower().replace('x', ',')
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--input_hw 需要形如 'H,W' 或 'HxW'，收到: {s}")
    H, W = int(parts[0]), int(parts[1])
    if H <= 0 or W <= 0:
        raise ValueError("--input_hw 的 H/W 必须为正整数")
    return (H, W)


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark(device, batch_sizes, iters_warmup, iters_run, model_kwargs, dtype: torch.dtype | None):
    model = TransformerFlowReconstructionModel(**model_kwargs).to(device)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()

    total, trainable = count_parameters(model)
    print(f"[Params] total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    results = []
    for B in batch_sizes:
        x = torch.randn(B, model_kwargs['input_dim'], device=device)
        t = torch.randint(low=0, high=100, size=(B,), device=device)
        if dtype is not None:
            x = x.to(dtype)

        # Warmup
        with torch.inference_mode():
            for _ in range(iters_warmup):
                _ = model(x, t)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

        # Timing
        times = []
        with torch.inference_mode():
            for _ in range(iters_run):
                start = time.perf_counter()
                _ = model(x, t)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

        times_sorted = sorted(times)
        avg = sum(times) / len(times)
        p50 = times_sorted[len(times_sorted) // 2]
        p90 = times_sorted[max(0, int(len(times_sorted) * 0.9) - 1)]
        throughput = B / avg

        max_mem = None
        if device.type == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
            torch.cuda.reset_peak_memory_stats(device)

        results.append((B, avg, p50, p90, throughput, max_mem))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--output_dim', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--seq_len', type=int, default=32)  # 仅保留以兼容；真实以 input_hw 为准
    parser.add_argument('--input_hw', type=str, default='8x8', help="形如 HxW 或 H,W（控制序列长度 H*W）")
    parser.add_argument('--pe_type', type=str, default='learnable_1d')
    parser.add_argument('--output_head_type', type=str, default='global')
    parser.add_argument('--out_channels_per_token', type=int, default=None)
    # Memory fusion toggles
    parser.add_argument('--use_memory_concat', dest='use_memory_concat', action='store_true', help='Enable memory concat fusion in decoder cross pathways')
    parser.add_argument('--no_use_memory_concat', dest='use_memory_concat', action='store_false')
    parser.set_defaults(use_memory_concat=False)
    parser.add_argument('--use_memory_film', dest='use_memory_film', action='store_true', help='Enable FiLM gating using memory in decoder cross pathways')
    parser.add_argument('--no_use_memory_film', dest='use_memory_film', action='store_false')
    parser.set_defaults(use_memory_film=True)
    parser.add_argument('--batches', type=str, default='8,16,32')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda')
    parser.add_argument('--dtype', type=str, default='float32', help='float32|float16|bfloat16')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype.lower(), torch.float32)
    if device.type == 'cpu' and dtype != torch.float32:
        print('[Warn] CPU 上将强制使用 float32')
        dtype = torch.float32

    input_hw = parse_input_hw(args.input_hw)

    model_kwargs = dict(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_model=args.d_model,
        max_time_steps=100,
        attention_type='relative',
        seq_len=args.seq_len,  # 模型内部不会直接使用该字段
        input_hw=input_hw,
        pe_type=args.pe_type,
        output_head_type=args.output_head_type,
        out_channels_per_token=args.out_channels_per_token,
        use_memory_concat=args.use_memory_concat,
        use_memory_film=args.use_memory_film,
    )

    batch_sizes = [int(s) for s in args.batches.split(',') if s.strip()]

    warmup = args.warmup
    iters = args.iters
    if device.type == 'cpu':
        warmup = max(3, min(warmup, 5))
        iters = max(10, min(iters, 20))

    print(f"[Device] {device}")
    print(f"[Precision] {dtype}")
    print(f"[Model] heads={args.num_heads}, layers={args.num_layers}, d_model={args.d_model}, input_dim={args.input_dim}, output_dim={args.output_dim}, input_hw={input_hw}, pe_type={args.pe_type}, output_head_type={args.output_head_type}, use_memory_concat={args.use_memory_concat}, use_memory_film={args.use_memory_film}")
    print(f"[Benchmark] batch_sizes={batch_sizes}, warmup={warmup}, iters={iters}")

    results = benchmark(device, batch_sizes, warmup, iters, model_kwargs, dtype)
    for B, avg, p50, p90, tput, max_mem in results:
        mem_str = f", peak_mem={max_mem:.1f} MB" if max_mem is not None else ""
        print(f"B={B}  avg={avg*1000:.2f} ms  p50={p50*1000:.2f} ms  p90={p90*1000:.2f} ms  throughput={tput:.2f} samples/s{mem_str}")


if __name__ == '__main__':
    main()