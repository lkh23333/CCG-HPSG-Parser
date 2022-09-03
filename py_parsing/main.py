import time, os, sys, random, argparse
import torch

from decoders.ccg_base_decoder import CCGBaseDecoder
from decoders.ccg_a_star_decoder import CCGAStarDecoder
from ccg_parsing_models import BaseParsingModel, LSTMParsingModel
from parser import Parser

sys.path.append('..')
from base import Atom
from data_loader import load_auto_file
from tools import to_auto

def main(args):
    # data_items, _ = load_auto_file(args.sanity_check_data_dir)
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
    with open(args.cat_dict_dir, 'r', encoding = 'utf8') as f:
        cat_dict = json.load(f)

    # decoder = CCGBaseDecoder(
    #     beam_width = args.beam_width,
    #     idx2tag = idx2category,
    #     cat_dict = cat_dict,
    #     top_k = args.top_k_supertags,
    #     timeout = args.decoder_timeout,
    #     apply_cat_filtering = args.apply_cat_filtering
    # )
    decoder = CCGAStarDecoder(
        idx2tag = idx2category,
        cat_dict = cat_dict,
        top_k = args.top_k_supertags,
        apply_supertagging_pruning = args.apply_supertagging_pruning,
        beta = args.beta,
        timeout = args.decoder_timeout,
        apply_cat_filtering = args.apply_cat_filtering
    )
    with open(args.instantiated_unary_rules_dir, 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open(args.instantiated_binary_rules_dir, 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    decoder._get_instantiated_unary_rules(instantiated_unary_rules)
    decoder._get_instantiated_binary_rules(instantiated_binary_rules)

    parsing_model = LSTMParsingModel(
        model_path = args.supertagging_model_path,
        supertagging_n_classes = len(idx2category),
        embed_dim = args.embed_dim,
        checkpoints_dir = args.supertagging_model_checkpoints_dir,
        checkpoint_epoch = args.supertagging_model_checkpoint_epoch,
        device = args.device
    )

    parser = Parser(
        parsing_model = parsing_model,
        decoder = decoder
    )

    accumulated_time = 0
    buffer = []
    for i in range(0, len(pretokenized_sents), args.batch_size):
        print(f'======== {i} / {len(pretokenized_sents)} ========')

        t0 = time.time()

        charts = parser.batch_parse(
            pretokenized_sents[i: i + args.batch_size]
        )
        # charts = parser.batch_sanity_check(
        #     pretokenized_sents[i: i + args.batch_size],
        #     golden_supertags[i: i + args.batch_size],
        #     print_cell_items = False
        # )

        time_cost = time.time() - t0
        print(f'time: {time_cost}s')
        accumulated_time += time_cost
        
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
    
    print(f'averaged parsing time of each sentence: {accumulated_time / len(pretokenized_sents)}')

    predicted_auto_file_output_dir = os.path.join(
        args.predicted_auto_files_dir,
        '_'.join([
            'a_star_unary_X_binary_seen',
            'cat_dict' + str(args.apply_cat_filtering),
            'topk' + str(args.top_k_supertags),
            'beam_width' + str(args.beam_width),
            'timeout' + str(args.decoder_timeout)
        ]) + '.auto'
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
    parser.add_argument('--cat_dict_dir', type = str, default = '../data/cat_dict.json')
    parser.add_argument('--supertagging_model_path', type = str, default = '../plms/bert-large-uncased')
    parser.add_argument('--supertagging_model_checkpoints_dir', type = str, default = '../ccg_supertagger/checkpoints')
    parser.add_argument('--supertagging_model_checkpoint_epoch', type = str, default = 19)
    parser.add_argument('--predicted_auto_files_dir', type = str, default = './evaluation')

    parser.add_argument('--embed_dim', type = int, default = 1024)
    parser.add_argument('--apply_cat_filtering', type = bool, default = True)
    parser.add_argument('--apply_supertagging_pruning', type = bool, default = True)
    parser.add_argument('--beta', type = float, default = 0.00005)
    parser.add_argument('--device', type = torch.device, default = torch.device('cpu'))
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--top_k_supertags', type = int, default = 10)
    parser.add_argument('--beam_width', type = int, default = 4)
    parser.add_argument('--decoder_timeout', help = 'time out value for decoding one sentence', type = float, default = 16.0)

    args = parser.parse_args()

    # print(f'======== base unary_X_binary_seen_topk{args.top_k_supertags}_beam_width{args.beam_width}_timeout{args.decoder_timeout} ========')
    print(f'======== A* unary_X_binary_seen_topk{args.top_k_supertags}_cat_dict{args.apply_cat_filtering}_timeout{args.decoder_timeout} ========')
    main(args)