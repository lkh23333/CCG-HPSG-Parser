import sys, os
import re
import torch
from typing import List, Tuple, Dict

sys.path.append('..')
from data_loader import DataItem

def pre_tokenize_sent(sent: str) -> List[str]:
    splited = sent.split(' ')
    returned = list()
    for token in splited:
        if re.match('[a-zA-Z]', token[-1]):
            returned.append(token)
        else:
            returned.extend([token[0: -1], token[-1]])
    return returned

def get_cat_ids(categories: List[str], category2idx: Dict[str, int]) -> List[int]:
    # take one list of category strings and return a list of corresponding one-hot vectors
    return [category2idx[category] for category in categories]

def prepare_data(data_items: List[DataItem], tokenizer, category2idx: Dict[str, int]):
    # return wrapped data needed to input into the model.
    # !!! The result hasn't been padded !!! 
    data = list() # a list containing a list of input_ids for each sentence
    mask = list() # a list containing the attention mask list for each sentence
    word_piece_tracked = list() # a list containing the list of word_piece_tracked for each sentence
    target = list() # a list containing the list of one-hot vectors for each sentence

    for data_item in data_items:
        pretokenized_sent = [token.contents for token in data_item.tokens]
        categories = [str(token.tag) for token in data_item.tokens]

        word_piece_tracked.append(
            [len(item) for item in tokenizer(pretokenized_sent, add_special_tokens = False).input_ids]
        )

        inputs = tokenizer(
            pretokenized_sent,
            add_special_tokens = False,
            is_split_into_words = True
        )
        data.append(inputs.input_ids)
        mask.append(inputs.attention_mask)
        target.append(get_cat_ids(categories, category2idx))

    max_length = max(
        max([len(input_ids) for input_ids in data]),
        max([len(tgt) for tgt in target])
    )
    for i in range(len(data)):
        assert len(data[i]) == len(mask[i])
        data[i] = data[i] + [0] * (max_length - len(data[i])) # padding
        mask[i] = mask[i] + [0] * (max_length - len(mask[i])) # padding
    for i in range(len(target)):
        target[i] = target[i] + [-100] * (max_length - len(target[i])) # padding

    return {
        'input_ids': data,
        'mask': mask,
        'word_piece_tracked': word_piece_tracked,
        'target': target
    }

def get_cross_entropy_loss(outputs: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, int]:
    # outputs: B*L*C, B - batch, L - max_length, C - number of categories
    # target: B*L, padding value -100
    loss = 0.0
    cnt = 0
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            if target[i, j] >= 0:
                loss += -torch.log(outputs[i, j, target[i, j]])
                cnt += 1
    return loss/cnt, cnt



if __name__ == '__main__':
    text = 'I haven\'t done that before.'
    pre_tokenize_sent(text)