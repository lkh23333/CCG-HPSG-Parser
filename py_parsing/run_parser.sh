#!/bin/bash

#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -j
#PJM -o job.out

source activate py310

EXP_NAME="lstm"

python -u parser.py \
 --supertagging_model_path ../plms/bert-large-uncased \
 --supertagging_model_checkpoint_dir ../ccg_supertagger/checkpoints/lstm_bert-base-uncased_drop0.5_epoch_14.pt \
 --predicted_auto_files_dir ./evaluation \
 --supertagging_model_name lstm \
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --decoder a_star \
 --beta 0.0001 \
 --top_k_supertags 10 \
 --batch_size 10 \
 --mode predict_batch \
 2>&1 | tee -a parser_$EXP_NAME.log