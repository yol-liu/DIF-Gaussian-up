#!/bin/bash
# Seed-42 repeat of the geometry-validation screen for the leading arms:
# R0 (T1-nomax), A0 (C0 + anatomy supervision), A1 (A0 + guided sampling).
# Same protocol as reports/geometry_validation (AdamW lr 2e-4, 2,000 updates, 6 views,
# 6 eval cases, scale 1) except seed 42 and its own common initialization. Arms run
# serially on one GPU; an existing run directory is never overwritten.
PY=/home/wyin/miniconda3/envs/torch2-3/bin/python
SEED=${SEED:-42}
ARMS=${ARMS:-"R0 A0 A1"}
OUT=reports/seed${SEED}_screen
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for ARM in $ARMS; do
  NAME=geomv1-train-${ARM}-seed${SEED}
  [ -d logs/$NAME ] && { echo "skip $NAME (exists)"; continue; }
  echo "START $NAME $(date +%T)"
  $PY code/train.py --cfg_path configs/geometry_validation/${ARM}.yaml --name $NAME \
    --dst_name LUNA16 --num_views 6 --batch_size 1 --num_points 1024 --num_workers 2 \
    --epoch 400 --lr 2e-4 --weight_decay 1e-2 --lr_decay 1e-3 \
    --max_steps 2000 --val_steps 1000 2000 --manifest_steps 2000 \
    --val_case_count 6 --eval_scale 1.0 --eval_npoint 65536 --diag_every 25 \
    --common_init logs/p1-common-init/seed${SEED}.pth \
    --seed $SEED --device cuda --no-wandb > $OUT/${NAME}.log 2>&1 \
    || { echo "FAILED $NAME (see $OUT/${NAME}.log)"; exit 1; }
  echo "END $NAME $(date +%T)"
done
echo "seed${SEED} screen done"
