#!/bin/bash

#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -j
#PJM -o job.out

source activate py310

EXP_NAME="fc"

python -u trainer.py \
 --model_name fc \
 --batch_size 8 \
 --embed_dim 768 \
 --dropout_p 0.5 \
 --model_path ../plms/bert-base-uncased \
 --checkpoints_dir ./checkpoints \
 2>&1 | tee -a trainer_$EXP_NAME.log