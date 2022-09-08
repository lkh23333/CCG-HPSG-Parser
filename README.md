# Instructions on How to Use the Current Parser

This is a pure python CCG parser still under development. Follow instructions below to use the current parser.

## Supertagging

### Train a Supertagger
```
cd ccg_supertagger
pjsub run_trainer.sh
```

Specify different parameters in `run_trainer.sh` so as to use different functions.  
`--model_path`: the relative path to the pretrained model folder, default to `../plms/bert-large-uncased`  
`--checkpoints_dir`: the directory to the folder storing checkpoints (.pt files), default to `./checkpoints`  
`--model_name`: [`fc`, `lstm`, `lstm-crf`], default to `lstm`  
`--n_epochs`: the preset number of epochs, default to `20`  
`--device`: specify the device to use, default to `'cuda'`  
`--batch_size`: the batch size in training, default to `8`  
`--lr`: the learning rate, default to `1e-5`  
`--embed_dim`: the dimension of the last hidden vectors in used BERT, 768 for bert-base-uncased and 1024 for bert-large-uncased, default to `1024`  
`--num_lstm_layers`: number of BiLSTM layers if `--model_name` is set to `lstm` or `lstm-crf`, default to `1`  
`--dropout_p`: the dropout probability, default to `0.5`  
`--mode`: the mode to use the trainer, choices include `train`, `train_on` and `test`, default to `train`. `train` is for training from scratch. `train_on` is for training from a specific checkpoint, and `--checkpoint_epoch` should be specified. `test` is for testing on one dataset using the model from a specific checkpoint, so `--checkpoint_epoch` and `--test_mode` should be specified.  
`--test_mode`: only for `test` mode, choices include `train_eval`, `dev_eval` and `test_eval`, default to `dev_eval`  
`--checkpoint_epoch`: only for `train_on` and `test` mode, the specific epoch of checkpoint to use, default to `14`

Some example scripts
- train a supertagger with BiLSTM + bert-base-uncased
```
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
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --dropout_p 0.5 \
 --model_path ../plms/bert-base-uncased \
 --checkpoints_dir ./checkpoints \
 --mode train \
 2>&1 | tee -a trainer_$EXP_NAME.log
```

- continue to train the supertagger from checkpoint of epoch 14
```
EXP_NAME="lstm"

python -u trainer.py \
 --model_name lstm \
 --batch_size 8 \
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --dropout_p 0.5 \
 --model_path ../plms/bert-base-uncased \
 --checkpoints_dir ./checkpoints \
 --mode train_on \
 --checkpoint_epoch 14 \
 2>&1 | tee -a trainer_$EXP_NAME.log
```

- test the supertagger using the checkpoint from epoch 14, on dev data
```
EXP_NAME="lstm"

python -u trainer.py \
 --model_name lstm \
 --batch_size 8 \
 --embed_dim 768 \
 --num_lstm_layers 1 \
 --dropout_p 0.5 \
 --model_path ../plms/bert-base-uncased \
 --checkpoints_dir ./checkpoints \
 --mode test \
 --test_mode dev_eval \
 --checkpoint_epoch 14 \
 2>&1 | tee -a trainer_$EXP_NAME.log
```

### Use the Supertagger
```
cd ccg_supertagger
pjsub run_supertagger.sh
```

Specify different parameters in `run_supertagger.sh` so as to use different functions.  
`--model_path`: the relative path to the pretrained model folder, default to `../plms/bert-large-uncased`  
`--checkpoint_dir`: the relative path to the checkpoint .pt file, default to `./checkpoints/lstm_bert-large-uncased_drop0.5_epoch_19.pt`  
`--model_name`: [`fc`, `lstm`], currently does not support CRF, default to `lstm`  
`--embed_dim`: the dimension of the last hidden vectors in used BERT, 768 for bert-base-uncased and 1024 for bert-large-uncased, default to `1024`  
`--num_lstm_layers`: number of BiLSTM layers if `--model_name` contains `lstm`, default to `1`  
`--device`: specify the device to use, default to `'cuda'`  
`--batch_size`: the batch size in batch inference, default to `8`  
`--top_k`: the maximum number of supertags allowed for one word, default to `10`  
`--beta`: the coefficient used to prune predicted categories, default to `0.0005`
`--mode`: mode of the supertagger, choices include `predict` and `sanity_check`. If `predict`, you can specify the .json file directory where you put a list of pretokenized sentences using `--pretokenized_sents_dir`, and you should also specify the output file directory using `--batch_predicted_dir`. If `sanity_check`, the supertagger will just run on dev data and print the (multi)tagging accuracy and average number of categories per word. Default to `sanity_check`.  
`--pretokenized_sents_dir`: used for `predict_batch`, default to `'../data/pretokenized_sents.json'`  
`--batch_predicted_dir`: used for `predict_batch`, default to `'./batch_predicted_supertags.json'`

 - An example scripts to use the supertagger with the checkpoint file `./checkpoints/lstm_bert-base-uncased_drop0.5_epoch_14.pt` in the mode `sanity_check`
```
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
```
## Parsing

### Use the Parser
```
cd py_parsing
pjsub run_parser.sh
```

Specify different parameters in `run_parser.sh` so as to use different functions.  
`--supertagging_model_path`: the path to the supertagging model, default to `../plms/bert-base-uncased`  
`--supertagging_model_checkpoint_dir`: the path to the supertagging model checkpoint .pt file, default to `../ccg_supertagger/checkpoints/lstm_bert-base-uncased_drop0.5_epoch_14.pt`  
`--predicted_auto_files_dir`: the directory to save the predicted auto file, default to `./evaluation`  
`--supertagging_model_name`: the model name to use, choices include `fc` and `lstm`, default to `lstm`  
`--embed_dim`: the dimension of the last hidden vectors in used BERT, 768 for bert-base-uncased and 1024 for bert-large-uncased, default to `768`  
`--num_lstm_layers`: number of BiLSTM layers if `--model_name` contains `lstm`, default to `1`  
`--decoder`: the decoder to use, choices include `base` and `a_star`, default to `a_star`  
`--apply_cat_filtering`: to control whether to apply category filtering, default to `True`  
`--apply_supertagging_pruning`: to control whether to apply the supertagging pruning method, if True, please specify the `--beta` parameter, default to `True`  
`--beta`: the coefficient to prune predicted categories whose probabilities lie within $\beta$ of that of the best category, default to `0.0005`  
`--top_k_supertags`: the maximum number of supertags allowed for one word, default to `10`  
`--beam_width`: used for `base` decoder, default to `4`  
`--batch_size`: the batch size set for supertagging, default to `10`  
`--decoder_timeout`: the preset maximum time for decoding one sentence, if exceeded the parser returns a null parse, default to `16.0`  
`--possible_roots`: categories allowable at the root of one parse, default to `S[dcl]|NP|S[wq]|S[q]|S[qem]|S[b]\\NP`  
`--device`: the device to use during supertagging, default to `cuda`  
`--mode`: the mode of the parser, choices include `sanity_check`, `predict_sent`, `batch_sanity_check` and `predict_batch`. If `sanity_check`, the parser reads the sample data in `sample.auto` and returns the parsing result with its golden supertags. If `predict_sent`, the parser reads sample data in `sample.auto` and returns the parsing result using its own supertagging results. If `batch_sanity_check`, the parser reads in dev data and returns the predicted .auto file using their golden supertags. If `predict_batch`, the parser reads in dev data and returns the predicted .auto file using its own supertagging results. Default to `sanity_check`.  

- An example script to use the parser with a BiLSTM+bert-base-uncased supertagging model and A* decoding in `predict_batch` mode
```
EXP_NAME="lstm"

python -u parser.py \
 --supertagging_model_path ../plms/bert-base-uncased \
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
```
### Evaluation of the Parsing Results
```
cd py_parsing/evaluation
export CANDC=candc
python -m depccg.tools.evaluate PATH/TO/wsj_00.parg PATH/TO/PREDICTED.auto
```
