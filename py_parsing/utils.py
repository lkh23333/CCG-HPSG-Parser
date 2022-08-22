import sys
from typing import *

sys.path.append('..')
from data_loader import DataItem

def get_pretokenized_sents(data_items: List[DataItem]):
    pretokenized_sents = list()
    for data_item in dev_data_items:
        pretokenized_sents.append([token.contents for token in data_item.tokens])
    return pretokenized_sents