from typing import *
import sys, time
import torch
import torch.nn as nn

from ccg_parsing_models import BaseParsingModel, SpanParsingModel
from decoders import Chart, Decoder, CCGBaseDecoder, CCGSpanDecoder

sys.path.append('..')
from base import Token, Category, ConstituentNode


class Parser:
    
    def __init__(
        self,
        parsing_model: nn.Module,
        decoder: Decoder,
        idx2tag: Dict[int, Any]
    ):
        self.parsing_model = parsing_model
        self.decoder = decoder
        self.idx2tag = idx2tag

    def batch_parse(
        self,
        pretokenized_sents: List[List[str]]
    ) -> List[Chart]:

        batch_representations = self.parsing_model(pretokenized_sents)
        for representation in batch_representations[0]:
            topk_ps, topk_ids = torch.topk(representation, k=8)
            print([self.idx2tag[int(idx.item())] for idx in topk_ids])
        charts = self.decoder.batch_decode(pretokenized_sents, batch_representations)
        
        return charts

    def parse(self, pretokenized_sent: List[str]) -> Chart:
        return self.batch_parse([pretokenized_sent])[0]



if __name__ == '__main__':
    # sample use
    pretokenized_sent = ['I', 'like', 'apples']

    import json
    with open('../data/lexical_category2idx_from_train_data.json', 'r', encoding = 'utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: cat for cat, idx in category2idx.items()}
    beam_width = 8

    decoder = CCGBaseDecoder(beam_width, idx2category)
    with open('../data/instantiated_unary_rules_with_X.json', 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open('../data/instantiated_binary_rules_from_train_data.json', 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    decoder._get_instantiated_unary_rules(instantiated_unary_rules)
    decoder._get_instantiated_binary_rules(instantiated_binary_rules)

    parsing_model = BaseParsingModel(
        model_path = '../plms/bert-base-uncased',
        supertagging_n_classes = len(idx2category),
        checkpoints_dir = '../ccg_supertagger/checkpoints',
        checkpoint_epoch = 20
    )
    parser = Parser(
        parsing_model = parsing_model,
        decoder = decoder,
        idx2tag = idx2category
    )


    chart = parser.parse(pretokenized_sent).chart
    print(chart[0][-1])