from typing import List, Dict, Any, TypeVar
import numpy, re
import torch
import torch.nn as nn

from utils import pre_tokenize_sent


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
        return _convert_model_outputs(outputs, len(word_piece_tracked))

    def _convert_model_outputs(self, outputs: torch.Tensor, sent_len: int) -> SupertaggerOutput:
        # outputs: B*L*C, B = 1
        # sent_len: length of the pretokenized sentence
        outputs = torch.squeeze(outputs) # L*C
        predicted = list()
        for i in range(sent_len):
            topk_ids = torch.topk(outputs[i], self.top_k)[1]
            predicted.append([self.idx2category[int(idx.item())] for idx in topk_ids])
        return predicted

if __name__ == '__main__':
    from transformers import BertTokenizer
    from transformers import BertModel
    tokenizer = BertTokenizer.from_pretrained('./models/plms/bert-base-uncased')
    model = BertModel.from_pretrained('./models/plms/bert-base-uncased')
    
    text_0 = 'I really want to do something.'
    text_1 = 'I haven\'t done that before.'
    pretokenized_sent = pre_tokenize_sent(text_1)
    
    print(tokenizer([text_0, text_1], add_special_tokens = False, padding = True))

    print(pretokenized_sent)
    
    word_piece_tracked = [len(item) for item in tokenizer(pretokenized_sent, add_special_tokens = False).input_ids]
    
    print(word_piece_tracked)
    
    inputs = tokenizer(
        pretokenized_sent,
        add_special_tokens = False,
        is_split_into_words = True
    )
    
    f2 = model(
            input_ids = torch.LongTensor([inputs.input_ids]),
            attention_mask = torch.Tensor([inputs.attention_mask]),
            output_hidden_states = True
        ).last_hidden_state # B*L*H
    
    print(f2.shape)
    
    f3 = f2.clone()
    word_piece_tracked = [word_piece_tracked]
    for i in range(f3.shape[0]):
        k = 0
        for j in range(len(word_piece_tracked[i])):
            n_piece = word_piece_tracked[i][j]
            f3[i, j] = torch.sum(f3[i, k:k+n_piece], dim = 0)
            k += n_piece
    
    