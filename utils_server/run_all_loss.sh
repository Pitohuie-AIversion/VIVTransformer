#!/bin/bash

NUM_GPU=2         # 实际GPU数量
BATCH_SIZE=6      # 每批并行任务数
START_IDX=10      # 起始loss_config编号
END_IDX=29        # 终止loss_config编号（含）

PYTHON_BIN="/share/fandixiaLab/suguangsheng/anaconda3/bin/python"
MAIN_PY="/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer/modify_multi_attention/main.py"
PROJECT_ROOT="/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer"

export PYTHONPATH=$PROJECT_ROOT

i=$START_IDX
while [ $i -le $END_IDX ]
do
  batch_end=$((i + BATCH_SIZE - 1))
  if [ $batch_end -gt $END_IDX ]; then
    batch_end=$END_IDX
  fi

  for ((j=i; j<=batch_end; j++))
  do
    gpu_id=$((j % NUM_GPU))
    echo "启动 loss_config_${j} -> 显卡 $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_BIN $MAIN_PY --loss_idx $j > log_loss${j}.txt 2>&1 &
  done

  wait  # 等本批全部跑完再跑下一批
  echo "本批次($i~$batch_end)已完成，继续下一批..."
  i=$((batch_end + 1))
done

echo "全部任务结束，可用 nvidia-smi 查看显卡利用率！"
