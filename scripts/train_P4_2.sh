gpus=1
IFS=',' read -r -a devices <<< "$gpus"
nproc=${#devices[@]}
cfg_path=./configs/experiment_plan/stage1/prob-dense-np.yaml

n_view=6
dst_name=LUNA16
name=DIFG-${n_view}v_$(date +%m%d_%H%M)

mkdir -p ./logs/$name

# 双 GPU DDP: 2 个进程分别绑定 GPU 0 / 1，NCCL 负责梯度同步
# wandb 日志: 首次使用前需 `wandb login <API_KEY>`，
# 或用离线模式: WANDB_MODE=offline bash scripts/train.sh
export WANDB_BASE_URL=http://localhost:8080
CUDA_VISIBLE_DEVICES=$gpus nohup python -m torch.distributed.run \
    --master_port 2027 \
    --nproc_per_node "$nproc" \
    code/train.py \
        --name $name \
        --batch_size 2 \
        --epoch 400 \
        --dst_name $dst_name \
        --num_views $n_view \
        --random_views \
        --cfg_path $cfg_path \
        --wandb \
        --dist \
        >> ./logs/$name/train.log 2>&1 &
