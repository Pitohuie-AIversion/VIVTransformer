#!/usr/bin/env bash
set -euo pipefail

# 并行注意力训练调度器 (Bash 版)
# - 读取 YAML 中的 attention_sweep_candidates / attention_skip_list
# - 在多张 GPU 上并行调度，每张卡最多同时运行若干进程
# - 每个子进程通过 CUDA_VISIBLE_DEVICES 绑定到单卡，避免 DataParallel 抢占
# - 训练完成后汇总绘制验证集 loss 曲线图

# ================== 可配置默认值（按你的服务器路径） ==================
# 参考 utils_server/run_all_loss.sh:8
PYTHON_BIN="/share/fandixiaLab/suguangsheng/anaconda3/bin/python"

# 训练脚本与项目根目录（固定服务器路径）
PROJECT_ROOT="/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench"
TRAINER="/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/generate_data/dynamic_resolution_trainer.py"
CONFIG_PATH="$PROJECT_ROOT/generate_data/dynamic_config_server_attention_sweep.yaml"

# 训练参数
EPOCHS=${EPOCHS:-1000}
GPUS_CSV=${GPUS_CSV:-"0,1"}       # 可通过环境变量覆盖，如 GPUS_CSV="0,1,2,3"
MAX_PER_GPU=${MAX_PER_GPU:-1}

# ================== 参数解析（可选覆盖） ==================
usage() {
  echo "用法: $0 [-c CONFIG] [-e EPOCHS] [-g GPU列表] [-m MAX_PER_GPU] [-p PROJECT_ROOT] [-t TRAINER] [-y PYTHON_BIN]"
  echo "示例: $0 -c $CONFIG_PATH -e 1000 -g 0,1 -m 2"
}

while getopts ":c:e:g:m:p:t:y:h" opt; do
  case $opt in
    c) CONFIG_PATH="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    g) GPUS_CSV="$OPTARG" ;;
    m) MAX_PER_GPU="$OPTARG" ;;
    p) PROJECT_ROOT="$OPTARG" ;;
    t) TRAINER="$OPTARG" ;;
    y) PYTHON_BIN="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "选项 -$OPTARG 需要参数"; usage; exit 1 ;;
    \?) echo "未知选项 -$OPTARG"; usage; exit 1 ;;
  esac
done

# ================== 前置检查 ==================
if [[ ! -x "$PYTHON_BIN" && ! "$PYTHON_BIN" =~ python$ ]]; then
  echo "[WARN] PYTHON_BIN 不可执行或不存在: $PYTHON_BIN，尝试使用系统 python"
  PYTHON_BIN="python"
fi

if [[ ! -f "$TRAINER" ]]; then
  echo "[ERR ] 找不到训练脚本: $TRAINER"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERR ] 找不到配置文件: $CONFIG_PATH"
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERR ] GPU 列表为空"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/results/attention_sweep"

# ================== CPU 线程上限，避免过量并行导致崩溃 ==================
# 该设置会被子进程继承。可通过环境变量自行覆盖，例如 OMP_NUM_THREADS=6 ./run_attention_sweep_parallel.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-4}"
export PYTORCH_NUM_THREADS="${PYTORCH_NUM_THREADS:-4}"

# ================== 读取 YAML 中的候选与跳过 ==================
read_yaml_lists() {
  local cfg="$1"
  "$PYTHON_BIN" - "$cfg" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
model = cfg.get('model', {}) if isinstance(cfg, dict) else {}
cands = model.get('attention_sweep_candidates', []) or []
skip  = model.get('attention_skip_list', []) or []
# 统一小写与去空白
cands = [str(x).strip().lower() for x in cands if str(x).strip()]
skip  = [str(x).strip().lower() for x in skip if str(x).strip()]
print('C:' + ' '.join(cands))
print('S:' + ' '.join(skip))
PY
}

CANDIDATES=()
SKIP_LIST=()
while IFS= read -r line; do
  if [[ "$line" == C:* ]]; then
    read -r -a CANDIDATES <<< "${line#C:}"
  elif [[ "$line" == S:* ]]; then
    read -r -a SKIP_LIST <<< "${line#S:}"
  fi
done < <(read_yaml_lists "$CONFIG_PATH")

# 结合内置已知失败集合，去重
KNOWN_FAIL=(sk vip ufo muse aft a2 shuffle)
for k in "${KNOWN_FAIL[@]}"; do SKIP_LIST+=("$k"); done
# 去重 SKIP_LIST
TMP=(); declare -A seen
for s in "${SKIP_LIST[@]}"; do
  [[ -z "$s" ]] && continue
  if [[ -z "${seen[$s]:-}" ]]; then TMP+=("$s"); seen[$s]=1; fi
done
SKIP_LIST=("${TMP[@]}")

# 过滤得到最终注意力列表
ATTENTIONS=(); unset seen; declare -A seen
for a in "${CANDIDATES[@]}"; do
  [[ -z "$a" ]] && continue
  # 检查是否在跳过列表
  skip_flag=0
  for s in "${SKIP_LIST[@]}"; do
    if [[ "$a" == "$s" ]]; then skip_flag=1; break; fi
  done
  [[ $skip_flag -eq 1 ]] && continue
  if [[ -z "${seen[$a]:-}" ]]; then ATTENTIONS+=("$a"); seen[$a]=1; fi
done

if [[ ${#ATTENTIONS[@]} -eq 0 ]]; then
  echo "[ERR ] 候选经跳过过滤后为空，请检查 YAML"
  exit 1
fi

echo "[INFO] 发现候选: ${CANDIDATES[*]}"
echo "[INFO] 跳过列表: ${SKIP_LIST[*]}"
echo "[INFO] 实际将运行: ${ATTENTIONS[*]}"

# ================== 并行调度 ==================
declare -A PIDS
for g in "${GPUS[@]}"; do PIDS[$g]=""; done

active_count() {
  local g="$1"; local cnt=0; local pid
  for pid in ${PIDS[$g]}; do
    if kill -0 "$pid" 2>/dev/null; then cnt=$((cnt+1)); fi
  done
  echo "$cnt"
}

prune_dead() {
  local g="$1"; local keep=(); local pid
  for pid in ${PIDS[$g]}; do
    if kill -0 "$pid" 2>/dev/null; then keep+=("$pid"); fi
  done
  PIDS[$g]="${keep[*]:-}"
}

start_job() {
  local g="$1"; local attn="$2"
  local attn_safe
  attn_safe=$(echo "$attn" | sed 's/[^a-z0-9_-]/_/g')
  local out_root="$PROJECT_ROOT/results/attention_sweep/$attn_safe"
  mkdir -p "$out_root/logs"
  echo "[GPU$g] 启动: $attn"
  (
    export CUDA_VISIBLE_DEVICES="$g"
    export PYTHONUNBUFFERED=1
    exec "$PYTHON_BIN" "$TRAINER" --config "$CONFIG_PATH" --attention-sweep "$attn" --epochs "$EPOCHS"
  ) >"$out_root/logs/train_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  local pid=$!
  PIDS[$g]="${PIDS[$g]} $pid"
}

# 构建队列（索引）
QUEUE=("${ATTENTIONS[@]}")
q_idx=0

while :; do
  # 退出条件：队列空且无存活进程
  total_alive=0
  for g in "${GPUS[@]}"; do
    prune_dead "$g"
    cnt=$(active_count "$g")
    total_alive=$((total_alive + cnt))
  done
  if [[ $q_idx -ge ${#QUEUE[@]} && $total_alive -eq 0 ]]; then
    break
  fi

  # 为每张 GPU 补齐进程到上限
  for g in "${GPUS[@]}"; do
    prune_dead "$g"
    while :; do
      cnt=$(active_count "$g")
      if [[ $cnt -lt $MAX_PER_GPU && $q_idx -lt ${#QUEUE[@]} ]]; then
        attn="${QUEUE[$q_idx]}"; q_idx=$((q_idx+1))
        start_job "$g" "$attn"
        sleep 0.2
      else
        break
      fi
    done
  done

  # 打印简单进度并等待
  counts=()
  for g in "${GPUS[@]}"; do counts+=("$(active_count "$g")"); done
  echo "[INFO] 运行中: [${counts[*]}]，剩余队列: $(( ${#QUEUE[@]} - q_idx ))"
  sleep 5
done

echo "[INFO] 所有任务已完成，开始汇总绘图..."

# ================== 汇总绘制所有注意力的验证损失曲线 ==================
"$PYTHON_BIN" - <<PY
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(r"$PROJECT_ROOT")
attn_dir = root / 'attention_results'
if not attn_dir.exists():
    print(f"未找到目录: {attn_dir}")
    raise SystemExit(0)
series = {}
for name in sorted(os.listdir(attn_dir)):
    p = attn_dir / name / 'loss_logs' / 'loss_log.txt'
    if not p.is_file():
        continue
    epochs, vals = [], []
    try:
        with open(p, 'r', encoding='utf-8') as f:
            header = True
            for line in f:
                if header:
                    header = False
                    continue
                parts = [x.strip() for x in line.strip().split(',')]
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                epochs.append(int(parts[0])); vals.append(float(parts[2]))
        if epochs and vals:
            series[name] = (epochs, vals)
    except Exception as e:
        print(f"跳过 {name}: {e}")
if not series:
    print("未收集到任何 loss 曲线，检查 attention_results/*/loss_logs/loss_log.txt 是否存在")
    raise SystemExit(0)
plt.figure(figsize=(16, 9))
for name in sorted(series.keys()):
    ep, vl = series[name]
    plt.plot(ep, vl, label=name)
plt.xlabel('Epoch'); plt.ylabel('Valid Loss (log10 scale)')
plt.yscale('log', base=10); plt.title('All Attention Validation Loss Curves')
plt.grid(True, which='both', ls='--', linewidth=0.5); plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
out_path = root / 'results' / 'attention_sweep' / 'all_valid_loss_curves.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight', facecolor='white')
print(f"已保存总图：{out_path}")
PY

echo "[OK ] 完成。请查看 $PROJECT_ROOT/results/attention_sweep/all_valid_loss_curves.png 与 $PROJECT_ROOT/attention_results/*/loss_logs/loss_log.txt"