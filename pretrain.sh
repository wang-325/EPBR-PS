# Set the path to save checkpoints
#OUTPUT_DIR='./ntu1/attention_li_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600'
OUTPUT_DIR='./ntu1/attention_basevit_1600'
# Set the path to Kinetics train set.
DATA_PATH='./ntu1/train.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2  --use_env  run_mae_pretraining.py \
python  run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 8 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 1601 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 8

