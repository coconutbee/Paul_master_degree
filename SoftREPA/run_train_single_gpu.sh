#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
NODE='node-name'
NAME='train'

DATASET="deepfashion"
BATCHSIZE=2
NUM_SAMPLES=10
NUM_ITER=200
EPOCHS=5

# Resume settings
# RESUME=true will continue from checkpoint saved in RESUME_DIR.
RESUME=false
RESUME_DIR="./data/deepfashion/sd3_batch4_epoch2_iter1600" # Set this to the checkpoint directory you want to resume from.

# Weights & Biases settings
USE_WANDB=true
WANDB_PROJECT="SoftREPA"
WANDB_ENTITY=""
WANDB_RUN_NAME="sd3_single_gpu_b2"
WANDB_MODE="online"   # online|offline|disabled
WANDB_LOG_IMAGES=8

# Validation sample filename mode
# true: use prompt as filename, false: use 0000.png style
VAL_FILENAME_USE_PROMPT=false

# Define path
LOGDIR="./data"
DATADIR="/media/ee303/4TB/sam3-body/sam3_labeded_training/deepfashion"

# train_single_gpu.py expects --datadir to be the parent of DATASET folder.
# If user gives .../<dataset>, convert it to its parent automatically.
if [ "$(basename "$DATADIR")" = "$DATASET" ]; then
	DATADIR="$(dirname "$DATADIR")"
fi

if [ ! -d "$DATADIR/$DATASET/train2017" ] || [ ! -f "$DATADIR/$DATASET/annotations/captions_train2017.json" ]; then
	echo "[ERROR] Dataset structure not found under: $DATADIR/$DATASET"
	echo "        Expecting train2017/ and annotations/captions_train2017.json"
	exit 1
fi

echo "Train SD3 (single GPU) start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
RESUME_ARGS=""
if [ "$RESUME" = true ]; then
	RESUME_ARGS="--resume --resume_dir $RESUME_DIR"
	echo "[INFO] Resume enabled from: $RESUME_DIR"
fi

WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
	WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT --wandb_mode $WANDB_MODE --wandb_log_images $WANDB_LOG_IMAGES"
	if [ -n "$WANDB_ENTITY" ]; then
		WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
	fi
	if [ -n "$WANDB_RUN_NAME" ]; then
		WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
	fi
	echo "[INFO] WandB enabled: project=$WANDB_PROJECT run=$WANDB_RUN_NAME mode=$WANDB_MODE"
fi

python train_single_gpu.py --model sd3 --dataset $DATASET --batch_size $BATCHSIZE \
 --datadir $DATADIR --logdir $LOGDIR --num_iter $NUM_ITER --num_eval $NUM_SAMPLES \
 --n_dc_tokens 4 --apply_dc True False False --epochs $EPOCHS --use_dc_t --dweight 10 \
 --val_filename_use_prompt $VAL_FILENAME_USE_PROMPT $RESUME_ARGS $WANDB_ARGS

# echo "Train SDXL start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
# python train.py --model sdxl --dataset $DATASET --batch_size $BATCHSIZE --separate_gpus \
#  --datadir $DATADIR --logdir $LOGDIR --num_iter $NUM_ITER --num_eval $NUM_SAMPLES \
#  --n_dc_tokens 4 --apply_dc True True False --epochs 1 --dweight 10

# echo "Train SD3 start"
# python train_single_gpu.py --model sd3 --dataset $DATASET --batch_size $BATCHSIZE \
#  --datadir $DATADIR --logdir $LOGDIR --num_iter $NUM_ITER --num_eval $NUM_SAMPLES \
#  --n_dc_tokens 4 --n_dc_layers 5 --epochs 2 --use_dc_t --dweight 0

echo "All epochs completed!"