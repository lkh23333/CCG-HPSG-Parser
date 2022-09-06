#!/bin/bash

#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -j
#PJM -o job.out

source activate py310

python -u supertagger.py \
 --model_path ../plms/bert-base-uncased \
 --checkpoint_dir ./checkpoints/epoch_14.pt \
 --model_name lstm \
 --batch_size 8 \
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --top_k 10 \
 --beta 0.0005 \
 --mode predict_batch