gpu=0

n_view=6
dst_name=/home/user/workspace/C2RV-CBCT/data/LUNA16
name=DIFG-6v_0910_1514
n_epoch=400
scale=1.0

CUDA_VISIBLE_DEVICES=$gpu python code/evaluate.py \
    --name $name \
    --epoch $n_epoch \
    --dst_name $dst_name \
    --split test \
    --num_views $n_view \
    --out_res_scale $scale \
    --save_results
