#!/bin/bash

EXP_NAME="lstm"

python -u trainer.py \
 --model_name lstm \
 --batch_size 8 \
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --dropout_p 0.5 \
 --model_path ../plms/bert-large-uncased \
 --checkpoints_dir ./checkpoints \
 2>&1 | tee -a trainer_$EXP_NAME.log