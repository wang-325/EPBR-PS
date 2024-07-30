# Set the path to save checkpoints
OUTPUT_DIR='./NTU-ST/checkpoint'

# path to dataset (train.csv/val.csv/test.csv)
DATA_PATH='./NTU-ST'

# path to pretrain model
MODEL_PATH='./checkpoint/vit-b-1600/checkpoint-1600.pth'

python  run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set ABNTU \
    --nb_classes 8 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 32 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3\
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 200 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --num_workers 16   \
