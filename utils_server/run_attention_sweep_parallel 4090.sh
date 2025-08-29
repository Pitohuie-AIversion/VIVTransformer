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
GPUS_CSV=${GPUS_CSV:-"0,1"}       # GPU 列表，用于绑定子进程（例如 GPUS_CSV="0,1,2,3")
MAX_PER_GPU=${MAX_PER_GPU:-1}      # 同卡并发上限

# 新增：单注意力的 loss config 扫描模式默认值（针对 modify_multi_attention/main.py）
LOSS_SCAN_MODE=0
ATTENTION_SINGLE=""
GEN_EXTREME_3MODES=0
LOSS_LIMIT=""
TRAINER_SET=0
CONFIG_SET=0
TRAINER_LOSS_DEFAULT="$PROJECT_ROOT/modify_multi_attention/main.py"
CONFIG_LOSS_DEFAULT="$PROJECT_ROOT/modify_multi_attention/configs/config.yaml"

# ================== 参数解析（可选覆盖） ==================
usage() {
  echo "用法: $0 [-c CONFIG] [-e EPOCHS] [-g GPU列表] [-m MAX_PER_GPU] [-p PROJECT_ROOT] [-t TRAINER] [-y PYTHON_BIN] [-S] [-A ATTENTION] [-x] [-N LIMIT]"
  echo "模式一（注意力横向扫描，默认）: $0 -c $CONFIG_PATH -e 1000 -g 0,1 -m 2"
  echo "模式二（单注意力的 loss config 扫描）: $0 -S -A sge -g 0,1 -m 2 [-x 生成前三模态极端配置] [-N 只取前N个loss配置] [-t $TRAINER_LOSS_DEFAULT -c $CONFIG_LOSS_DEFAULT]"
}

while getopts ":c:e:g:m:p:t:y:A:N:Sxh" opt; do
  case $opt in
    c) CONFIG_PATH="$OPTARG"; CONFIG_SET=1 ;;
    e) EPOCHS="$OPTARG" ;;
    g) GPUS_CSV="$OPTARG" ;;
    m) MAX_PER_GPU="$OPTARG" ;;
    p) PROJECT_ROOT="$OPTARG" ;;
    t) TRAINER="$OPTARG"; TRAINER_SET=1 ;;
    y) PYTHON_BIN="$OPTARG" ;;
    A) ATTENTION_SINGLE="$OPTARG" ;;
    N) LOSS_LIMIT="$OPTARG" ;;
    S) LOSS_SCAN_MODE=1 ;;
    x) GEN_EXTREME_3MODES=1 ;;
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
  # 如果启用 loss 扫描但未显式提供训练脚本，则切换到默认的 modify_multi_attention/main.py
  if [[ "$LOSS_SCAN_MODE" -eq 1 && "$TRAINER_SET" -eq 0 ]]; then
    TRAINER="$TRAINER_LOSS_DEFAULT"
    echo "[INFO] 切换 TRAINER 到: $TRAINER"
  else
    exit 1
  fi
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[WARN] 找不到配置文件: $CONFIG_PATH"
  # 如果启用 loss 扫描但未显式提供配置文件，则切换到默认的 modify_multi_attention/configs/config.yaml
  if [[ "$LOSS_SCAN_MODE" -eq 1 && "$CONFIG_SET" -eq 0 ]]; then
    CONFIG_PATH="$CONFIG_LOSS_DEFAULT"
    echo "[INFO] 切换 CONFIG 到: $CONFIG_PATH"
  else
    echo "[ERR ] 找不到配置文件且无法自动切换"
    exit 1
  fi
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

# ================== 单注意力 loss config 扫描模式 ==================
if [[ "$LOSS_SCAN_MODE" -eq 1 ]]; then
  if [[ -z "$ATTENTION_SINGLE" ]]; then
    echo "[ERR ] 使用 loss 扫描模式必须指定 -A ATTENTION"
    exit 1
  fi

  # 若未设置 TRAINER/CONFIG，采用针对 modify_multi_attention 的默认
  if [[ "$TRAINER_SET" -eq 0 ]]; then
    TRAINER="$TRAINER_LOSS_DEFAULT"
  fi
  if [[ "$CONFIG_SET" -eq 0 ]]; then
    CONFIG_PATH="$CONFIG_LOSS_DEFAULT"
  fi

  echo "[INFO] 进入单注意力的 loss config 扫描模式"
  echo "[INFO] ATTENTION: $ATTENTION_SINGLE"
  echo "[INFO] TRAINER  : $TRAINER"
  echo "[INFO] CONFIG   : $CONFIG_PATH"

  # 可选：生成“前三模态极端分配”的临时配置文件
  if [[ "$GEN_EXTREME_3MODES" -eq 1 ]]; then
    echo "[INFO] 生成前三模态极端 loss 配置的临时文件..."
    TMP_CFG_PATH=$("$PYTHON_BIN" - "$CONFIG_PATH" "$PROJECT_ROOT" "$ATTENTION_SINGLE" <<'PY'
import sys, yaml
from pathlib import Path
cfg_path, project_root, attn = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 统一规范：专注前三个模态，topk 固定为 3；严格避免语义重复
loss_cfgs = [
    dict(base_weight=1.0, svd_weights=[0.0, 0.0, 0.0], topk=3),        # 仅 base
    dict(base_weight=0.0, svd_weights=[1.0, 1.0, 1.0], topk=3),        # 仅 SVD（等权）
    dict(base_weight=0.0, svd_weights=[1.0, 0.0, 0.0], topk=3),        # 纯第1模态
    dict(base_weight=0.0, svd_weights=[0.0, 1.0, 0.0], topk=3),        # 纯第2模态
    dict(base_weight=0.0, svd_weights=[0.0, 0.0, 1.0], topk=3),        # 纯第3模态
    dict(base_weight=0.1, svd_weights=[0.8, 0.15, 0.05], topk=3),      # 金字塔 1>2>3
    dict(base_weight=0.1, svd_weights=[0.05, 0.15, 0.8], topk=3),      # 反向金字塔 3>2>1
    dict(base_weight=0.5, svd_weights=[1.0, 1.0, 1.0], topk=3),        # base+SVD 等权
    dict(base_weight=0.2, svd_weights=[1.0, 1.0, 1.0], topk=3),        # base 较弱
    dict(base_weight=0.3, svd_weights=[0.6, 0.3, 0.1], topk=3),        # 递减 1>2>3（另一组）
]

cfg['loss_configs'] = loss_cfgs
out_dir = Path(project_root) / 'results' / 'loss_scan_configs'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f'config_loss_scan_{attn}.yaml'
with open(out_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(out_path.as_posix())
PY
)
    echo "[INFO] 使用临时配置文件: $TMP_CFG_PATH"
    CONFIG_PATH="$TMP_CFG_PATH"
  fi

  # 读取 loss_configs 数量
  read_loss_count() {
    local cfg="$1"
    "$PYTHON_BIN" - "$cfg" <<'PY'
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
loss_cfgs = cfg.get('loss_configs', []) or []
print(len(loss_cfgs))
PY
  }
  TOTAL_LOSS_CFGS=$(read_loss_count "$CONFIG_PATH")
  if [[ -z "$TOTAL_LOSS_CFGS" || "$TOTAL_LOSS_CFGS" -le 0 ]]; then
    echo "[ERR ] 配置文件中未找到 loss_configs"
    exit 1
  fi
  echo "[INFO] loss_configs 总数: $TOTAL_LOSS_CFGS"

  # 构建 loss 索引队列
  QUEUE=()
  for ((i=0; i< TOTAL_LOSS_CFGS; i++)); do QUEUE+=("$i"); done
  if [[ -n "$LOSS_LIMIT" ]]; then
    # 仅取前 N 个
    N=$LOSS_LIMIT
    if [[ "$N" -lt ${#QUEUE[@]} ]]; then
      QUEUE=("${QUEUE[@]:0:$N}")
      echo "[INFO] 已限制仅运行前 $N 个 loss 配置"
    fi
  fi

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

  start_job_loss() {
    local g="$1"; local loss_idx="$2"
    local out_root="$PROJECT_ROOT/results/loss_scan/$ATTENTION_SINGLE"
    mkdir -p "$out_root/logs"
    echo "[GPU $g] 启动: $ATTENTION_SINGLE (loss_idx=$loss_idx)"
    (
      export CUDA_VISIBLE_DEVICES="$g"
      export PYTHONUNBUFFERED=1
      exec "$PYTHON_BIN" "$TRAINER" --config "$CONFIG_PATH" --attention-type "$ATTENTION_SINGLE" --epochs "$EPOCHS" --device cuda --loss_idx "$loss_idx"
    ) >"$out_root/logs/train_loss_${loss_idx}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    local pid=$!
    PIDS[$g]="${PIDS[$g]} $pid"
  }

  q_idx=0
  while :; do
    total_alive=0
    for g in "${GPUS[@]}"; do
      prune_dead "$g"
      cnt=$(active_count "$g")
      total_alive=$((total_alive + cnt))
    done
    if [[ $q_idx -ge ${#QUEUE[@]} && $total_alive -eq 0 ]]; then
      break
    fi

    for g in "${GPUS[@]}"; do
      prune_dead "$g"
      while :; do
        cnt=$(active_count "$g")
        if [[ $cnt -lt $MAX_PER_GPU && $q_idx -lt ${#QUEUE[@]} ]]; then
          loss_idx="${QUEUE[$q_idx]}"; q_idx=$((q_idx+1))
          start_job_loss "$g" "$loss_idx"
          sleep 0.2
        else
          break
        fi
      done
    done

    counts=()
    for g in "${GPUS[@]}"; do counts+=("$(active_count "$g")"); done
    echo "[INFO] 运行中: [${counts[*]}]，剩余队列: $(( ${#QUEUE[@]} - q_idx ))"
    sleep 5
  done

  echo "[INFO] 所有任务已完成，开始汇总绘图 (按 loss 配置对齐)..."

  "$PYTHON_BIN" - <<PY
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(r"$PROJECT_ROOT")
attn = "$ATTENTION_SINGLE"
base = root / 'attention_results'
series = {}
if base.exists():
    for name in sorted(os.listdir(base)):
        if not name.startswith('loss_config_'):
            continue
        p = base / name / attn / 'loss_logs' / 'loss_log.txt'
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
else:
    print(f"未找到目录: {base}")

if not series:
    print("未收集到任何 loss 曲线，检查 attention_results/loss_config_*/"+attn+"/loss_logs/loss_log.txt 是否存在")
    raise SystemExit(0)
plt.figure(figsize=(16, 9))
for name in sorted(series.keys(), key=lambda x: int(x.split('_')[-1])):
    ep, vl = series[name]
    plt.plot(ep, vl, label=name)
plt.xlabel('Epoch'); plt.ylabel('Valid Loss (log10 scale)')
plt.yscale('log', base=10); plt.title(f'Loss-config Validation Curves ({attn})')
plt.grid(True, which='both', ls='--', linewidth=0.5); plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
out_path = root / 'results' / 'loss_scan' / f'{attn}_loss_scan.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight', facecolor='white')
print(f"已保存总图：{out_path}")
PY

  echo "[OK ] 完成。请查看 $PROJECT_ROOT/results/loss_scan/${ATTENTION_SINGLE}_loss_scan.png 与 $PROJECT_ROOT/attention_results/loss_config_*/${ATTENTION_SINGLE}/loss_logs/loss_log.txt"
  exit 0
fi

# ================== 读取 YAML 中的候选与跳过（注意力横向扫描模式） ==================
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
  echo "[GPU $g] 启动: $attn"
  (
    export CUDA_VISIBLE_DEVICES="$g"
    export PYTHONUNBUFFERED=1
    exec "$PYTHON_BIN" "$TRAINER" --config "$CONFIG_PATH" --attention-sweep "$attn" --epochs "$EPOCHS" --device cuda
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