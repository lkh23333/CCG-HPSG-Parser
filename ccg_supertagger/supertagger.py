from typing import List, Dict, Any, TypeVar
import os, sys, numpy, re
import torch
import torch.nn as nn

from utils import pre_tokenize_sent
from models.simple_model import CCGSupertaggerModel

sys.path.append('..')
from data_loader import load_auto_file

CategoryStr = TypeVar('CategoryStr')
SupertaggerOutput = List[List[CategoryStr]]


class CCGSupertagger:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        idx2category: Dict[int, str],
        top_k: int = 1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.idx2category = idx2category
        self.top_k = top_k

    def predict_sent(self, sent: str) -> SupertaggerOutput:
        # predict 1-top category for each token
        pretokenized_sent = pre_tokenize_sent(sent)
        word_piece_tracked = [len(item) for item in self.tokenizer(pretokenized_sent, add_special_tokens = False).input_ids]
        inputs = self.tokenizer(
            pretokenized_sent,
            add_special_tokens = False,
            is_split_into_words = True
        )
        outputs = self.model(
            encoded_batch = torch.LongTensor([inputs.input_ids]),
            mask = torch.FloatTensor([inputs.attention_mask]),
            word_piece_tracked = [word_piece_tracked]
        )
        return self._convert_model_outputs(outputs, len(word_piece_tracked))

    def _convert_model_outputs(self, outputs: torch.Tensor, sent_len: int) -> SupertaggerOutput:
        # outputs: B*L*C, B = 1
        # sent_len: length of the pretokenized sentence
        outputs = torch.squeeze(outputs) # L*C
        predicted = list()
        for i in range(sent_len):
            topk_ids = torch.topk(outputs[i], self.top_k)[1]
            predicted.append([self.idx2category[int(idx.item())] for idx in topk_ids])
        return predicted

    def _load_model_checkpoint(self, checkpoints_dir: str, checkpoint_epoch: int):
        checkpoint = torch.load(os.path.join(checkpoints_dir, f'epoch_{checkpoint_epoch}.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

if __name__ == '__main__':
    dev_data_dir = '../data/ccgbank-wsj_00.auto'
    _, categories = load_auto_file(dev_data_dir)
    categories = sorted(categories)
    category2idx = {categories[idx]: idx for idx in range(len(categories))}
    UNK_CATEGORY = 'UNK_CATEGORY'
    category2idx[UNK_CATEGORY] = len(category2idx)
    idx2category = {idx: category for category, idx in category2idx.items()}

    from transformers import BertTokenizer
    model_path = './models/plms/bert-base-uncased'
    supertagger = CCGSupertagger(
        model = CCGSupertaggerModel(model_path, len(category2idx)),
        tokenizer = BertTokenizer.from_pretrained(model_path),
        idx2category = idx2category
    )
    checkpoints_dir = './checkpoints'
    checkpoint_epoch = 5
    supertagger._load_model_checkpoint(checkpoints_dir, checkpoint_epoch)

    sent = 'Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group'
    predicted = supertagger.predict_sent(sent)
    print(predicted)