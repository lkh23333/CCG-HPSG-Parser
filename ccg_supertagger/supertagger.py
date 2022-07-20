from typing import List, Dict, Any, TypeVar
import numpy, re
import torch
import torch.nn as nn

from utils import pre_tokenize_sent


CategoryStr = TypeVar('CategoryStr')
SupertaggerOutput = List[List[CategoryStr]]


class CCGSupertagger:
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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
        return _convert_model_outputs(outputs)

    def _convert_model_outputs(self, output_dicts: List[Dict[str, numpy.ndarray]]) -> SupertaggerOutput:
        pass

if __name__ == '__main__':
    from transformers import BertTokenizer
    from transformers import BertModel
    tokenizer = BertTokenizer.from_pretrained('./models/plms/bert-base-uncased')
    model = BertModel.from_pretrained('./models/plms/bert-base-uncased')
    
    text_0 = 'I really want to do something.'
    text_1 = 'I haven\'t done that before.'
    pretokenized_sent = pre_tokenize_sent(text_1)
    
    print(tokenizer([text_0, text_1], add_special_tokens = False, padding = True))
    assert None

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
    
    