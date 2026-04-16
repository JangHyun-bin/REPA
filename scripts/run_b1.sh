#!/bin/bash
cd /home/famoz/projects/dl/REPA-prelim
source /home/famoz/miniconda3/bin/activate dl

accelerate launch train.py \
    --exp-name b1_s42 \
    --model SiT-L/2 \
    --data-dir data/ffhq256-prelim \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --mixed-precision bf16 \
    --max-train-steps 400000 \
    --checkpointing-steps 50000 \
    --sampling-steps 9999999 \
    --proj-coeff 0.5 \
    --num-classes 1000 \
    --cfg-prob 0.1 \
    --encoder-depth 8 \
    --enc-type dinov2-vit-b \
    --path-type linear \
    --prediction v \
    --weighting uniform \
    --seed 42 \
    --output-dir exps \
    --report-to wandb \
    --allow-tf32
