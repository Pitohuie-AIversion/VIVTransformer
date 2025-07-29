import os
import matplotlib.pyplot as plt

# 两个目录和它们的label
dirs = [
    ("attention_results", "A"),
    (r"F:\Zhaoyang\VIVTransformer\attention_results3covr_dif", "B")
]
loss_log_subpath = "loss_logs/loss_log.txt"

plt.figure(figsize=(14, 8))
linestyles = ["-", "--"]  # 不同目录用不同线型

all_attn_types = set()

# 1. 先收集所有attention类型（子目录名），便于对齐
for dir_path, _ in dirs:
    for attn_type in os.listdir(dir_path):
        log_file = os.path.join(dir_path, attn_type, loss_log_subpath)
        if os.path.isfile(log_file):
            all_attn_types.add(attn_type)

all_attn_types = sorted(list(all_attn_types))

# 2. 再绘图
for attn_type in all_attn_types:
    for idx, (dir_path, dir_label) in enumerate(dirs):
        log_file = os.path.join(dir_path, attn_type, loss_log_subpath)
        if not os.path.isfile(log_file):
            continue
        epochs, valid_loss = [], []
        with open(log_file, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3 or not parts[0].strip().isdigit():
                    continue
                epochs.append(int(parts[0]))
                valid_loss.append(float(parts[2]))
        if epochs and valid_loss:
            plt.plot(epochs, valid_loss, label=f"{attn_type} ({dir_label})", linestyle=linestyles[idx % len(linestyles)])

plt.xlabel("Epoch")
plt.ylabel("Valid Loss (log10 scale)")
plt.title("Validation Loss Curve Comparison (Two Results Folders)")
plt.yscale("log", base=10)
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
