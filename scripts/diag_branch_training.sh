#!/bin/bash
# Diagnostic: with AdamW and one learning rate for all parameters, does the new branch's
# internal weights train? T1 plus a B0 baseline under the same optimizer; same seed,
# common init and data order as P1, 2,000 updates. Earlier SGD runs: logs/diag-T1-lrscale*.
set -e
PY=/home/wyin/miniconda3/envs/torch2-3/bin/python
LR=${LR:-2e-4}
ARMS=${ARMS:-"T1 B0"}
for ARM in $ARMS; do
  NAME=diag-adamw-${ARM}-lr${LR}
  [ -d logs/$NAME ] && { echo "skip $NAME"; continue; }
  $PY code/train.py --cfg_path configs/multiview/${ARM}.yaml --name $NAME \
    --dst_name LUNA16 --num_views 6 --batch_size 1 --num_points 1024 --num_workers 2 \
    --epoch 400 --lr $LR --weight_decay 1e-2 --lr_decay 1e-3 \
    --max_steps 2000 --val_steps 1000 2000 --manifest_steps 20 \
    --val_case_count 6 --eval_scale 1.0 --eval_npoint 65536 --diag_every 25 \
    --common_init logs/p1-common-init/seed0.pth \
    --seed 0 --device cuda --no-wandb > reports/${NAME}.log 2>&1 \
    || echo "$NAME FAILED (see reports/${NAME}.log)" &
done
wait
echo "diagnostic runs done"
