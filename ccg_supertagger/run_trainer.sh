#!/bin/bash

#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -j
#PJM -o job.out

source activate py310

EXP_NAME="lstm"

python -u trainer.py \
 --model_name lstm \
 --batch_size 8 \
 --embed_dim 1024 \
 --num_lstm_layers 1 \
 --dropout_p 0.5 \
 --model_path ../plms/bert-large-uncased \
 --checkpoints_dir ./checkpoints_$EXP_NAME \
 --mode train_on \
 --checkpoint_epoch 19 \
 2>&1 | tee -a trainer_$EXP_NAME.log