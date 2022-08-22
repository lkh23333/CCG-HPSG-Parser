import time, os, sys, random, argparse
import torch

from decoders import CCGBaseDecoder
from ccg_parsing_models import BaseParsingModel
from parser import Parser

sys.path.append('..')
from base import Atom
from data_loader import load_auto_file
from tools import to_auto

def main(args):
    data_items, _ = load_auto_file(args.dev_data_dir)
    pretokenized_sents = list()
    golden_supertags = list()
    data_ids = list()
    for data_item in data_items:
        pretokenized_sents.append([token.contents for token in data_item.tokens])
        golden_supertags.append([str(token.tag) for token in data_item.tokens])
        data_ids.append(data_item.id)

    import json
    with open(args.lexical_category2idx_dir, 'r', encoding = 'utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: cat for cat, idx in category2idx.items()}

    decoder = CCGBaseDecoder(
        beam_width = args.beam_width,
        top_k = args.top_k_supertags,
        idx2tag = idx2category,
        timeout = args.decoder_timeout
    )
    with open(args.instantiated_unary_rules_dir, 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open(args.instantiated_binary_rules_dir, 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    decoder._get_instantiated_unary_rules(instantiated_unary_rules)
    decoder._get_instantiated_binary_rules(instantiated_binary_rules)

    parsing_model = BaseParsingModel(
        model_path = args.supertagging_model_path,
        supertagging_n_classes = len(idx2category),
        checkpoints_dir = args.supertagging_model_checkpoints_dir,
        checkpoint_epoch = args.supertagging_model_checkpoint_epoch,
        device = args.device
    )

    parser = Parser(
        parsing_model = parsing_model,
        decoder = decoder
    )

    buffer = []
    for i in range(0, len(pretokenized_sents), args.batch_size):
        print(f'======== {i} / {len(pretokenized_sents)} ========')

        t0 = time.time()

        # charts = parser.batch_parse(
        #     pretokenized_sents[i: i + args.batch_size]
        # )
        charts = parser.batch_sanity_check(
            pretokenized_sents[i: i + args.batch_size],
            golden_supertags[i: i + args.batch_size],
            print_cell_items = False
        )
        print(f'time: {time.time() - t0}s')
        
        # for chart in charts:
        #     print(chart.chart[0][-1])

        tmp_data_ids = data_ids[i: i + args.batch_size]
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
                        if str(item.constituent.tag) in ['S[dcl]', 'NP', 'S[wq]', 'S[b]\\NP']:
                            print(str(item.constituent))
                            cell_item = item
                            break
            
            if cell_item:
                buffer.append(to_auto(cell_item.constituent) + '\n')
            else:
                buffer.append('(<L S None None None S>)\n')
    
    predicted_auto_file_output_dir = os.path.join(
        args.predicted_auto_files_dir,
        '_'.join([
            'unary_X_binary_seen',
            'topk' + str(args.top_k_supertags),
            'beam_width' + str(args.beam_width),
            'timeout' + str(args.decoder_timeout)
        ]) + '_GOLD.auto'
    )
    with open(predicted_auto_file_output_dir, 'w', encoding='utf8') as f:
        f.writelines(buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'parsing')
    parser.add_argument('--train_data_dir', type = str, default = '../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--dev_data_dir', type = str, default = '../data/ccgbank-wsj_00.auto')
    parser.add_argument('--test_data_dir', type = str, default = '../data/ccgbank-wsj_23.auto')
    parser.add_argument('--sanity_check_data_dir', type = str, default = '../data/train_data_sample1000.auto')
    parser.add_argument('--lexical_category2idx_dir', type = str, default = '../data/lexical_category2idx_cutoff.json')
    parser.add_argument('--instantiated_unary_rules_dir', type = str, default = '../data/instantiated_unary_rules_with_X.json')
    parser.add_argument('--instantiated_binary_rules_dir', type = str, default = '../data/instantiated_seen_binary_rules.json')
    parser.add_argument('--supertagging_model_path', type = str, default = '../plms/bert-base-uncased')
    parser.add_argument('--supertagging_model_checkpoints_dir', type = str, default = '../ccg_supertagger/checkpoints')
    parser.add_argument('--supertagging_model_checkpoint_epoch', type = str, default = 2)
    parser.add_argument('--device', type = torch.device, default = torch.device('cuda:5'))
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--top_k_supertags', type = int, default = 1)
    parser.add_argument('--beam_width', type = int, default = 30)
    parser.add_argument('--decoder_timeout', help = 'time out value for decoding one sentence', type = float, default = 16.0)
    parser.add_argument('--predicted_auto_files_dir', type = str, default = './evaluation')
    args = parser.parse_args()

    print(f'======== unary_X_binary_seen_topk{args.top_k_supertags}_beam_width{args.beam_width}_timeout{args.decoder_timeout} ========')
    main(args)