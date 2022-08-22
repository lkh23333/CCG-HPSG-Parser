from typing import *
import sys, time, json
import torch
import torch.nn as nn

from ccg_parsing_models import BaseParsingModel, SpanParsingModel
from decoders.decoder import Chart, Decoder
from decoders.ccg_base_decoder import CCGBaseDecoder

sys.path.append('..')
from ccg_supertagger.utils import pre_tokenize_sent
from base import Token, Category, ConstituentNode
from data_loader import load_auto_file
from tools import to_auto


class Parser:
    
    def __init__(
        self,
        parsing_model: nn.Module,
        decoder: Decoder
    ):
        self.parsing_model = parsing_model
        self.decoder = decoder

    def batch_parse(
        self,
        pretokenized_sents: List[List[str]]
    ) -> List[Chart]:

        batch_representations = self.parsing_model(pretokenized_sents)
        charts = self.decoder.batch_decode(pretokenized_sents, batch_representations)
        
        return charts

    def parse(self, pretokenized_sent: List[str]) -> Chart:
        return self.batch_parse([pretokenized_sent])[0]

    def batch_sanity_check(
        self,
        pretokenized_sents: List[List[str]],
        golden_supertags: List[List[str]],
        print_cell_items: bool = False
    ) -> List[Chart]:
        
        charts = list()
        for i in range(len(pretokenized_sents)):
            charts.append(self.decoder.sanity_check(pretokenized_sents[i], golden_supertags[i], print_cell_items))
        return charts

    def sanity_check(self, pretokenized_sent: List[str], golden_supertags: List[str], print_cell_items: bool = False) -> Chart:
        return self.batch_sanity_check([pretokenized_sent], [golden_supertags], print_cell_items)[0]


if __name__ == '__main__':
    # sample use
    data_items, _ = load_auto_file('sample.auto')
    pretokenized_sent = [token.contents for token in data_items[0].tokens]
    golden_supertags = [str(token.tag) for token in data_items[0].tokens]

    with open('../data/lexical_category2idx_cutoff.json', 'r', encoding = 'utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: cat for cat, idx in category2idx.items()}
    beam_width = 3
    with open('../data/cat_dict.json', 'r', encoding = 'utf8') as f:
        cat_dict = json.load(f)

    decoder = CCGBaseDecoder(
        top_k = 50,
        beam_width = beam_width,
        idx2tag = idx2category,
        cat_dict = cat_dict
    )
    with open('../data/instantiated_unary_rules_with_X.json', 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open('../data/instantiated_seen_binary_rules.json', 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    decoder._get_instantiated_unary_rules(instantiated_unary_rules)
    decoder._get_instantiated_binary_rules(instantiated_binary_rules)

    parsing_model = BaseParsingModel(
        model_path = '../plms/bert-base-uncased',
        supertagging_n_classes = len(idx2category),
        checkpoints_dir = '../ccg_supertagger/checkpoints',
        checkpoint_epoch = 2
    )
    parser = Parser(
        parsing_model = parsing_model,
        decoder = decoder
    )


    # chart = parser.sanity_check(pretokenized_sent, golden_supertags, print_cell_items=True)
    chart = parser.parse(pretokenized_sent)
    
    for cell_item in chart.chart[0][-1].cell_items:
        print(to_auto(cell_item.constituent))