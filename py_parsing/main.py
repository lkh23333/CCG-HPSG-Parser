import time, sys, random
import torch
from parsing import CCGParser

sys.path.append('../ccg_supertagger')
from supertagger import CCGSupertagger
from models.simple_model import CCGSupertaggerModel
sys.path.append('..')
from data_loader import load_auto_file
from utils import to_auto


if __name__ == '__main__':

    dev_data_dir = '../data/ccgbank-wsj_00.auto'
    dev_data_items, categories = load_auto_file(dev_data_dir)
    categories = sorted(categories)
    category2idx = {categories[idx]: idx for idx in range(len(categories))}
    UNK_CATEGORY = 'UNK_CATEGORY'
    category2idx[UNK_CATEGORY] = len(category2idx)
    idx2category = {idx: category for category, idx in category2idx.items()}

    pretokenized_sents = list()
    for data_item in dev_data_items:
        pretokenized_sents.append([token.contents for token in data_item.tokens])

    from transformers import BertTokenizer
    model_path = '../ccg_supertagger/models/plms/bert-base-uncased'
    supertagger = CCGSupertagger(
        model = CCGSupertaggerModel(model_path, len(category2idx)),
        tokenizer = BertTokenizer.from_pretrained(model_path),
        idx2category = idx2category
    )
    checkpoints_dir = '../ccg_supertagger/checkpoints'
    checkpoint_epoch = 5
    supertagger._load_model_checkpoint(checkpoints_dir, checkpoint_epoch)

    pyparser = CCGParser(
        idx2category = idx2category,
        beam_width = 3
    )

    import json
    with open('../data/instantiated_unary_rules_from_train_data.json', 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open('../data/instantiated_binary_rules_from_train_data.json', 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    pyparser._get_instantiated_unary_rules(instantiated_unary_rules)
    pyparser._get_instantiated_binary_rules(instantiated_binary_rules)

    batch_size = 10
    buffer = []
    for i in range(0, len(pretokenized_sents), batch_size):
        print(f'======== {i} / {len(pretokenized_sents)} ========')
        outputs = supertagger.get_model_outputs_for_batch(pretokenized_sents[i: i+batch_size])
        data_ids = [data_item.id for data_item in dev_data_items[i: i+batch_size]]

        t0 = time.time()
        charts = pyparser.batch_parse(
            pretokenized_sents = pretokenized_sents[i: i+batch_size],
            tags_distributions = outputs
        )
        print(f'time: {time.time() - t0}s')
        
        # for chart in charts:
        #     print(chart.chart[0][-1])

        for j in range(len(charts)):

            if len(charts[j].chart[0][-1]) == 0:
                chart_item = None
            else:
                chart_item = charts[j].chart[0][-1][random.randint(0, max(0, len(charts[j].chart[0][-1])-1))]
            buffer.append(data_ids[j] + '\n')

            if chart_item:
                buffer.append(to_auto(chart_item.constituent) + '\n')
            else:
                buffer.append('()\n')
    
    with open('./evaluation/wsj_00.predicted.auto', 'w', encoding='utf8') as f:
        f.writelines(buffer)