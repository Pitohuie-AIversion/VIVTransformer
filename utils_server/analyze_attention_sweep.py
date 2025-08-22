#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析批量注意力扫描结果，判定哪些注意力机制为“异常”。

判定依据（任一满足则记为异常）：
- 日志缺失或为空
- 日志中出现错误关键词（Traceback/RuntimeError/Out of memory/Exception/❌ 等）
- 日志中出现 NaN/Inf（大小写不敏感，按单词匹配）
- 未能解析出任何损失曲线（无 Epoch 或损失数字）
- 最终 Valid Loss 为 NaN/Inf
- 过早 Early Stopping（默认 <3 epoch，可通过参数调整）
- 未生成模型文件（models/*.pt 或 *.pth）

输入：
- root（默认 ./results/attention_sweep）
输出：
- sweep_analysis.json ：包含 normal / abnormal 列表和详细原因
- sweep_analysis.md   ：Markdown 摘要
- 终端打印人类可读的总结

用法示例：
python utils_server/analyze_attention_sweep.py \
  --root ./results/attention_sweep \
  --min-epochs-ok 3
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("analyze_attention_sweep")

ERROR_KEYWORDS = [
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "out of memory",
    "device-side assert",
    "Segmentation fault",
    "KeyError:",
    "ValueError:",
    "IndexError:",
    "ZeroDivisionError",
    "CUBLAS",
    "cufft",
    "failed to",
    "❌",
    "ERROR",
    "Exception",
    "Aborted",
    "killed",
    "integer modulo by zero",
    "No inf checks were recorded",
]

# 匹配 “🎯 Epoch [e/N], Train Loss: x, Valid Loss: y, Test Loss: z”
EPOCH_LINE_RE = re.compile(
    r"Epoch\s*\[(\d+)\/(\d+)\].*?Train\s*Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?Valid\s*Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?Test\s*Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

CSV_LINE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)

NAN_INF_WORD_RE = re.compile(r"\b(nan|inf)\b", re.IGNORECASE)
EARLY_STOP_PATTERNS = [
    re.compile(r"Early\s*Stopping", re.IGNORECASE),
    re.compile(r"触发\s*Early\s*Stopping"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze attention sweep results and detect abnormal attentions")
    p.add_argument("--root", type=str, default=str(Path("./results") / "attention_sweep"), help="注意力扫描根目录")
    p.add_argument("--report-json", type=str, default=None, help="JSON 报告输出路径（默认写到 root 下）")
    p.add_argument("--report-md", type=str, default=None, help="Markdown 报告输出路径（默认写到 root 下）")
    p.add_argument("--min-epochs-ok", type=int, default=3, help="早停判定为异常的最小 epoch 阈值（小于该值则判为过早早停）")
    p.add_argument("--verbose", action="store_true", help="打印更多调试信息")
    return p.parse_args()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def read_text(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except Exception as e:
        LOGGER.debug("Failed to read %s: %s", path, e)
        return []


def list_attention_dirs(root: Path) -> List[Path]:
    dirs = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if child.is_dir() and child.name != "logs":
            dirs.append(child)
    return dirs


def pick_log_file(attn_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    logs_dir = attn_dir / "logs"
    attn_name = attn_dir.name
    preferred = logs_dir / f"training_{attn_name}.log"
    if preferred.exists():
        return preferred
    # 选择 logs/ 下体积最大的 .log
    if logs_dir.exists():
        candidates += [p for p in logs_dir.glob("*.log") if p.is_file()]
    # 退化到任意 .log
    candidates += [p for p in attn_dir.glob("*.log") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def parse_metrics(lines: List[str]) -> Tuple[List[Dict], Optional[int], Optional[int]]:
    metrics: List[Dict] = []
    detected_header = False
    max_epoch_seen, total_epochs = None, None
    for ln in lines:
        m = EPOCH_LINE_RE.search(ln)
        if m:
            e_cur = int(m.group(1))
            e_tot = int(m.group(2))
            tr = float(m.group(3))
            va = float(m.group(4))
            te = float(m.group(5))
            metrics.append({"epoch": e_cur, "train": tr, "valid": va, "test": te})
            max_epoch_seen = max(e_cur, max_epoch_seen or 0)
            total_epochs = e_tot
            continue
        # CSV 形式
        if not detected_header and ("Epoch" in ln and "," in ln):
            detected_header = True
            continue
        if detected_header:
            m2 = CSV_LINE_RE.match(ln)
            if m2:
                e_cur = int(m2.group(1))
                tr = float(m2.group(2))
                va = float(m2.group(3))
                te = float(m2.group(4))
                metrics.append({"epoch": e_cur, "train": tr, "valid": va, "test": te})
                max_epoch_seen = max(e_cur, max_epoch_seen or 0)
    return metrics, max_epoch_seen, total_epochs


def detect_early_stop(lines: List[str]) -> Optional[int]:
    # 尝试从 Early Stopping 附近的行推断 epoch（若日志有 Epoch [] 则会被 parse_metrics 捕获）
    for idx, ln in enumerate(lines):
        if any(p.search(ln) for p in EARLY_STOP_PATTERNS):
            # 向上回溯找最近一次 Epoch 行
            for back in range(0, 10):
                j = idx - back
                if j < 0:
                    break
                m = EPOCH_LINE_RE.search(lines[j])
                if m:
                    return int(m.group(1))
            return 1  # 保守返回 1
    return None


def has_nan_inf(lines: List[str]) -> bool:
    return any(NAN_INF_WORD_RE.search(ln) for ln in lines)


def collect_errors(lines: List[str]) -> List[str]:
    found = []
    for ln in lines:
        low = ln.lower()
        # 精细匹配 NaN/Inf 已单独处理，这里匹配其他错误关键词
        if any(k.lower() in low for k in ERROR_KEYWORDS):
            found.append(ln.strip())
    return found[:5]  # 最多保留5条代表性错误


def has_model_file(attn_dir: Path) -> bool:
    models_dir = attn_dir / "models"
    if not models_dir.exists():
        return False
    return any(p.suffix in (".pt", ".pth") for p in models_dir.glob("*"))


def analyze_attention(attn_dir: Path, min_epochs_ok: int) -> Dict:
    attn = attn_dir.name
    result: Dict = {"attention": attn, "status": "normal", "reasons": []}

    log_path = pick_log_file(attn_dir)
    if log_path is None or not log_path.exists() or log_path.stat().st_size == 0:
        result["status"] = "abnormal"
        result["reasons"].append("missing_or_empty_log")
        if not has_model_file(attn_dir):
            result["reasons"].append("model_missing")
        return result

    lines = read_text(log_path)

    # 错误关键词
    errs = collect_errors(lines)
    if errs:
        result["status"] = "abnormal"
        result.setdefault("details", {})["errors"] = errs
        result["reasons"].append("errors_in_log")

    # NaN/Inf
    if has_nan_inf(lines):
        result["status"] = "abnormal"
        result["reasons"].append("nan_or_inf_detected")

    # 指标
    metrics, last_epoch, total_epochs = parse_metrics(lines)
    if not metrics:
        result["status"] = "abnormal"
        result["reasons"].append("no_metrics_parsed")
    else:
        last = metrics[-1]
        for key in ("valid", "train", "test"):
            v = last.get(key)
            if v is None or math.isinf(v) or math.isnan(v):
                result["status"] = "abnormal"
                result["reasons"].append(f"final_{key}_loss_nan_inf")
                break

    # 早停
    early_epoch = detect_early_stop(lines)
    if early_epoch is not None and early_epoch < max(1, min_epochs_ok):
        result["status"] = "abnormal"
        result["reasons"].append(f"early_stop_too_early@{early_epoch}")

    # 模型
    if not has_model_file(attn_dir):
        result["status"] = "abnormal"
        result["reasons"].append("model_missing")

    # 附加信息
    result.setdefault("details", {})["log_path"] = str(log_path)
    if total_epochs is not None:
        result["details"]["total_epochs"] = total_epochs
    if last_epoch is not None:
        result["details"]["last_epoch"] = last_epoch
    if metrics:
        result["details"]["last_metrics"] = metrics[-1]

    return result


def parse_sweep_summary(root: Path) -> Dict[str, List[str]]:
    summary_file = root / "sweep_summary.txt"
    summary = {"success": [], "failed": []}
    if not summary_file.exists():
        return summary
    text = "\n".join(read_text(summary_file))
    # 粗略解析 Success/Failed 列表
    m_succ = re.search(r"Success\s*\(\d+\)\s*:\s*\[(.*?)\]", text, re.DOTALL)
    m_fail = re.search(r"Failed\s*\(\d+\)\s*:\s*\[(.*?)\]", text, re.DOTALL)
    def split_list(m):
        if not m:
            return []
        raw = m.group(1).strip()
        if not raw:
            return []
        return [s.strip().strip("'\"").lower() for s in raw.split(',') if s.strip()]
    summary["success"] = split_list(m_succ)
    summary["failed"] = split_list(m_fail)
    return summary


def main():
    args = parse_args()
    setup_logging(args.verbose)

    root = Path(args.root)
    if not root.exists():
        LOGGER.error("根目录不存在: %s", root)
        sys.exit(2)

    LOGGER.info("扫描目录: %s", root)

    sweep_summary = parse_sweep_summary(root)
    if sweep_summary["success"] or sweep_summary["failed"]:
        LOGGER.info("读取到 sweep_summary.txt: success=%d, failed=%d",
                    len(sweep_summary["success"]), len(sweep_summary["failed"]))

    attn_dirs = list_attention_dirs(root)
    if not attn_dirs:
        LOGGER.warning("未在 %s 下发现注意力子目录（期望结构: root/<attn>/{logs,models}）", root)

    analysis: List[Dict] = []
    abnormal, normal = [], []
    for d in attn_dirs:
        res = analyze_attention(d, args.min_epochs_ok)
        # 若 sweep_summary 标记失败，附加原因
        attn_lower = d.name.lower()
        if attn_lower in sweep_summary.get("failed", []):
            if res["status"] != "abnormal":
                res["status"] = "abnormal"
            res.setdefault("reasons", []).append("marked_failed_in_sweep_summary")
        analysis.append(res)
        if res["status"] == "abnormal":
            abnormal.append(d.name)
        else:
            normal.append(d.name)

    analysis.sort(key=lambda x: x["attention"])  # 按名称排序
    abnormal.sort(); normal.sort()

    report = {
        "root": str(root.resolve()),
        "abnormal": abnormal,
        "normal": normal,
        "count": {
            "total": len(attn_dirs),
            "abnormal": len(abnormal),
            "normal": len(normal),
        },
        "items": analysis,
    }

    # 输出文件
    json_path = Path(args.report_json) if args.report_json else (root / "sweep_analysis.json")
    md_path = Path(args.report_md) if args.report_md else (root / "sweep_analysis.md")

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 摘要
    lines = [
        f"# Attention Sweep 分析报告",
        f"- 根目录: {report['root']}",
        f"- 总计: {report['count']['total']}，正常: {report['count']['normal']}，异常: {report['count']['abnormal']}",
        "",
        "## 异常注意力",
        ("无" if not abnormal else "\n".join(f"- {a}" for a in abnormal)),
        "",
        "## 正常注意力",
        ("无" if not normal else "\n".join(f"- {a}" for a in normal)),
        "",
        "## 详细条目（异常原因）",
    ]
    for item in analysis:
        if item.get("status") == "abnormal":
            reasons = ", ".join(item.get("reasons", [])) or "(无)"
            lines.append(f"- {item['attention']}: {reasons}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # 控制台输出
    print("\n=== Attention Sweep 分析完成 ===")
    print(f"根目录: {report['root']}")
    print(f"总计: {report['count']['total']} | 正常: {report['count']['normal']} | 异常: {report['count']['abnormal']}")
    if abnormal:
        print("\n异常注意力:")
        for a in abnormal:
            print(f"  - {a}")
    else:
        print("\n未检测到异常注意力 ✔")
    print(f"\nJSON 报告: {json_path}")
    print(f"Markdown 报告: {md_path}")


if __name__ == "__main__":
    main()