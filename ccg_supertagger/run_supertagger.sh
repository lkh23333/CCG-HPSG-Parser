#!/bin/bash

#PJM -g gk77
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -j
#PJM -o job.out

source activate py310

EXP_NAME="lstm"
MODEL_NAME="large"
TOPK="10"
BETA="0.001"

python -u supertagger.py \
 --model_path ../plms/bert-large-uncased \
 --checkpoint_dir ./checkpoints/lstm_bert-large-uncased_drop0.5_epoch_19.pt \
 --model_name lstm \
 --device cuda \
 --batch_size 8 \
 --embed_dim 1024 \
 --num_lstm_layers 1 \
 --top_k 10 \
 --beta 0.001 \
 --mode sanity_check \
 2>&1 | tee -a supertagger_${EXP_NAME}_${MODEL_NAME}_${TOPK}_${BETA}.log