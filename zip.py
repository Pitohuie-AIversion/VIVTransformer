import os
import zipfile

# 根目录
ROOT = r"F:\Zhaoyang\VIVTransformer\attention_results_2"
# 输出 ZIP 文件
OUT_ZIP = os.path.join(ROOT, "all_attention_results.zip")

with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    # 遍历所有注意力机制子目录
    for attn in os.listdir(ROOT):
        attn_dir = os.path.join(ROOT, attn)
        if not os.path.isdir(attn_dir):
            continue
        # 针对 difference_results 和 visualization_results 两个子目录
        for sub in ("difference_results", "visualization_results"):
            subdir = os.path.join(attn_dir, sub)
            if not os.path.isdir(subdir):
                continue
            # 递归加入文件
            for root, _, files in os.walk(subdir):
                for fname in files:
                    fullpath = os.path.join(root, fname)
                    # arcname 保留相对于 ROOT 的目录结构
                    arcname = os.path.relpath(fullpath, ROOT)
                    zf.write(fullpath, arcname)
    print(f"打包完成：{OUT_ZIP}")
