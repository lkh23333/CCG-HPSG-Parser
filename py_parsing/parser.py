from typing import *
import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn

from ccg_parsing_models import BaseParsingModel, LSTMParsingModel
from decoders.decoder import Chart, Decoder
from decoders.ccg_base_decoder import CCGBaseDecoder
from decoders.ccg_a_star_decoder import CCGAStarDecoder

sys.path.append('..')
from ccg_supertagger.utils import pre_tokenize_sent
from base import Atom, Token, Category, ConstituentNode
from data_loader import DataItem, load_auto_file
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
        charts = self.decoder.batch_decode(
            pretokenized_sents, batch_representations
        )
        
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
            charts.append(
                self.decoder.sanity_check(
                    pretokenized_sents[i],
                    golden_supertags[i],
                    print_cell_items
                )
            )
        return charts

    def sanity_check(
        self,
        pretokenized_sent: List[str],
        golden_supertags: List[str],
        print_cell_items: bool = False
    ) -> Chart:
        return self.batch_sanity_check(
            [pretokenized_sent],
            [golden_supertags],
            print_cell_items
        )[0]


def get_batch_data(data_items: List[DataItem]) -> Dict[str, Any]:
    pretokenized_sents = list()
    golden_supertags = list()
    data_ids = list()
    for data_item in data_items:
        pretokenized_sents.append(
            [
                token.contents
                for token in data_item.tokens
            ]
        )
        golden_supertags.append(
            [
                str(token.tag)
                for token in data_item.tokens
            ]
        )
        data_ids.append(data_item.id)

    return {
        'pretokenized_sents': pretokenized_sents,
        'golden_supertags': golden_supertags,
        'data_ids': data_ids
    }


def run(
    batch_data: Dict[str, Any],
    parser: Parser,
    saving_dir: str,
    batch_size: int = 10,
    possible_roots: str = 'S[dcl]|NP|S[wq]|S[q]|S[qem]|S[b]\\NP',
    mode: str = 'predict_batch'
) -> None:
    """
    Input:
        batch_data -  a dictionary storing pretokenized sentences,
                      corresponding golden supertags and data ids
        parser - the parser
        saving_dir - the directory to save the predicted .auto file
        batch_size - the batch size set for supertagging
        possible_roots - a list of allowable categories at the root of each parse
        mode - the mode specified for batch parsing,
               choices in ['batch_sanity_check', 'predict_batch']
    """
    pretokenized_sents = batch_data['pretokenized_sents']
    golden_supertags = batch_data['golden_supertags']
    data_ids = batch_data['data_ids']
    possible_roots = possible_roots.split('|')

    accumulated_time = 0
    n_null_parses = 0
    buffer = []
    for i in range(0, len(pretokenized_sents), batch_size):
        print(f'======== {i} / {len(pretokenized_sents)} ========')

        t0 = time.time()

        if mode == 'predict_batch':
            charts = parser.batch_parse(
                pretokenized_sents[i: i + batch_size]
            )
        elif mode == 'batch_sanity_check':
            charts = parser.batch_sanity_check(
                pretokenized_sents[i: i + batch_size],
                golden_supertags[i: i + batch_size],
                print_cell_items = False
            )
        else:
            raise RuntimeError('Please check the batch running mode!!!')

        time_cost = time.time() - t0
        print(f'time cost for this batch: {time_cost}s')
        accumulated_time += time_cost

        tmp_data_ids = data_ids[i: i + batch_size]
        for j in range(len(charts)):

            buffer.append(tmp_data_ids[j] + '\n')

            cell_item = None
            if charts[j] is None:
                cell_item = None
            elif charts[j].chart[0][-1].cell_items is None:
                cell_item = None
            elif len(charts[j].chart[0][-1].cell_items) == 0:
                cell_item = None
            else:
                for item in charts[j].chart[0][-1].cell_items:
                    if isinstance(item.constituent.tag, Atom):
                        if str(item.constituent.tag) in possible_roots:
                            print(str(item.constituent))
                            cell_item = item
                            break

            if cell_item:
                buffer.append(to_auto(cell_item.constituent) + '\n')
            else:
                buffer.append('(<L S None None None S>)\n')
                n_null_parses += 1

    print(
        f'averaged parsing time of each sentence: {accumulated_time / len(pretokenized_sents)}'
    )

    print(
        f'null parses: {n_null_parses} / {len(pretokenized_sents)} = {n_null_parses / len(pretokenized_sents): .2f}'
    )

    with open(saving_dir, 'w', encoding='utf8') as f:
        f.writelines(buffer)


def apply_parser(args):

    with open(args.lexical_category2idx_dir, 'r', encoding = 'utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: cat for cat, idx in category2idx.items()}
    with open(args.cat_dict_dir, 'r', encoding = 'utf8') as f:
        cat_dict = json.load(f)

    if args.decoder == 'base':
        print(
            f'======== decoder{args.decoder}_beamwidth{args.beam_width}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout} ========'
        )
        decoder = CCGBaseDecoder(
            beam_width=args.beam_width,
            idx2tag=idx2category,
            cat_dict=cat_dict,
            top_k=args.top_k_supertags,
            apply_supertagging_pruning=args.apply_supertagging_pruning,
            beta=args.beta,
            timeout=args.decoder_timeout,
            apply_cat_filtering=args.apply_cat_filtering
        )
    elif args.decoder == 'a_star':
        print(
            f'======== decoder{args.decoder}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout} ========'
        )
        decoder = CCGAStarDecoder(
            idx2tag=idx2category,
            cat_dict=cat_dict,
            top_k=args.top_k_supertags,
            apply_supertagging_pruning=args.apply_supertagging_pruning,
            beta=args.beta,
            timeout=args.decoder_timeout,
            apply_cat_filtering=args.apply_cat_filtering
        )
    else:
        raise RuntimeError('Please check the setting of the decoder!!!')
    
    with open(args.instantiated_unary_rules_dir, 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open(args.instantiated_binary_rules_dir, 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    decoder._get_instantiated_unary_rules(instantiated_unary_rules)
    decoder._get_instantiated_binary_rules(instantiated_binary_rules)

    if args.supertagging_model_name == 'fc':
        parsing_model = BaseParsingModel(
            model_path=args.supertagging_model_path,
            supertagging_n_classes=len(idx2category),
            embed_dim=args.embed_dim,
            checkpoint_dir=args.supertagging_model_checkpoint_dir,
            device=torch.device(args.device)
        )
    elif args.supertagging_model_name == 'lstm':
        parsing_model = LSTMParsingModel(
            model_path=args.supertagging_model_path,
            supertagging_n_classes=len(idx2category),
            embed_dim=args.embed_dim,
            num_lstm_layers=args.num_lstm_layers,
            checkpoint_dir=args.supertagging_model_checkpoint_dir,
            device=torch.device(args.device)
        )
    else:
        raise RuntimeError('Please check the supertagging model name!!!')

    parser = Parser(
        parsing_model = parsing_model,
        decoder = decoder
    )

    if args.mode == 'sanity_check':
        data_items, _ = load_auto_file(args.sample_data_dir)
        pretokenized_sent = [token.contents for token in data_items[0].tokens]
        golden_supertags = [str(token.tag) for token in data_items[0].tokens]

        chart = parser.sanity_check(pretokenized_sent, golden_supertags, print_cell_items=True)
        
        # print out all successful parses
        for cell_item in chart.chart[0][-1].cell_items:
            print(to_auto(cell_item.constituent))

    elif args.mode == 'predict_sent':
        data_items, _ = load_auto_file(args.sample_data_dir)
        pretokenized_sent = [token.contents for token in data_items[0].tokens]
        chart = parser.parse(pretokenized_sent)

        # print out all successful parses
        for cell_item in chart.chart[0][-1].cell_items:
            print(to_auto(cell_item.constituent))

    elif args.mode == 'batch_sanity_check':
        data_items, _ = load_auto_file(args.dev_data_dir)
        batch_data = get_batch_data(data_items)
        plm_name = args.supertagging_model_path.split('/')[-1]
        saving_dir = os.path.join(
            args.predicted_auto_files_dir,
            f'DECODER{args.decoder}MODEL{args.supertagging_model_name}PLM{plm_name}_beamwidth{args.beam_width}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout}_GOLD.auto'
        )
        
        run(
            batch_data=batch_data,
            parser=parser,
            saving_dir=saving_dir,
            batch_size=args.batch_size,
            possible_roots=args.possible_roots,
            mode=args.mode
        )

    elif args.mode == 'predict_batch':
        data_items, _ = load_auto_file(args.dev_data_dir)
        batch_data = get_batch_data(data_items)
        plm_name = args.supertagging_model_path.split('/')[-1]
        saving_dir = os.path.join(
            args.predicted_auto_files_dir,
            f'DECODER{args.decoder}MODEL{args.supertagging_model_name}PLM{plm_name}_beamwidth{args.beam_width}_topk{args.top_k_supertags}_beta{args.beta}_timeout{args.decoder_timeout}.auto'
        )
        
        run(
            batch_data=batch_data,
            parser=parser,
            saving_dir=saving_dir,
            batch_size=args.batch_size,
            possible_roots=args.possible_roots,
            mode=args.mode
        )
    
    else:
        raise RuntimeError('Please check the mode of the parser!!!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='apply parsing')

    parser.add_argument('--sample_data_dir', type=str, default='./sample.auto')
    parser.add_argument('--train_data_dir', type=str,
                        default='../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--dev_data_dir', type=str,
                        default='../data/ccgbank-wsj_00.auto')
    parser.add_argument('--test_data_dir', type=str,
                        default='../data/ccgbank-wsj_23.auto')
    parser.add_argument('--lexical_category2idx_dir', type=str,
                        default='../data/lexical_category2idx_cutoff.json')
    parser.add_argument('--instantiated_unary_rules_dir', type=str,
                        default='../data/instantiated_unary_rules_with_X.json')
    parser.add_argument('--instantiated_binary_rules_dir', type=str,
                        default='../data/instantiated_seen_binary_rules.json')
    parser.add_argument('--cat_dict_dir', type=str,
                        default='../data/cat_dict.json')
                        
    parser.add_argument('--supertagging_model_path', type=str,
                        default='../plms/bert-base-uncased')
    parser.add_argument('--supertagging_model_checkpoint_dir',
                        type=str, default='../ccg_supertagger/checkpoints/lstm_bert-base-uncased_drop0.5_epoch_14.pt')
    parser.add_argument('--predicted_auto_files_dir',
                        type=str, default='./evaluation')

    parser.add_argument('--supertagging_model_name', type=str, default='lstm',
                        choices=['fc', 'lstm'])
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_lstm_layers', type=int, default=1)
    parser.add_argument('--decoder', type=str, default='a_star',
                        choices=['base', 'a_star'])
    parser.add_argument('--apply_cat_filtering', type=bool, default=True)
    parser.add_argument('--apply_supertagging_pruning',
                        type=bool, default=True)
    parser.add_argument('--beta', type=float, default=0.0005)
    parser.add_argument('--top_k_supertags', type=int, default=10)
    parser.add_argument('--beam_width', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--decoder_timeout', help='time out value for decoding one sentence',
                        type=float, default=16.0)
    parser.add_argument('--possible_roots', help='possible categories at the roots of parses',
                        type=str, default='S[dcl]|NP|S[wq]|S[q]|S[qem]|S[b]\\NP')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--mode', type=str, default='sanity_check',
                        choices=['sanity_check', 'predict_sent', 'batch_sanity_check', 'predict_batch'])

    args = parser.parse_args()

    apply_parser(args)