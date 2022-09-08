#!/bin/bash

#PJM -g gk77
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -j
#PJM -o job.out

source activate py310

EXP_NAME="lstm"

python -u supertagger.py \
 --model_path ../plms/bert-base-uncased \
 --checkpoint_dir ./checkpoints/lstm_bert-base-uncased_drop0.5_epoch_14.pt \
 --model_name lstm \
 --device cuda \
 --batch_size 8 \
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --top_k 10 \
 --beta 0.0001 \
 --mode sanity_check \
 2>&1 | tee -a supertagger_$EXP_NAME.log