#!/bin/bash

EXP_NAME="lstm"

python -u trainer.py \
 --model_name lstm \
 --batch_size 8 \
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --dropout_p 0.2 \
 --model_path ./models/plms/bert-base-uncased \
 --checkpoints_dir ./checkpoints_$EXP_NAME \
 2>&1 | tee -a trainer_$EXP_NAME.log
