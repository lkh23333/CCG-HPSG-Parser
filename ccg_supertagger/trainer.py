from typing import Union, Optional, Dict, Any
import sys, os, argparse, logging
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from models.simple_model import CCGSupertaggerModel
from utils import prepare_data

sys.path.append('..')
from data_loader import AutoParser

UNK_CATEGORY = 'UNK_CATEGORY'


class CCGSupertaggingDataset(Dataset):
    def __init__(self, data, mask, word_piece_tracked, target):
        self.data = torch.LongTensor(data)
        self.mask = torch.FloatTensor(mask)
        self.word_piece_tracked = word_piece_tracked
        self.target = torch.LongTensor(target)
    
    def __getitem__(self, idx):
        return (self.data[idx], self.mask[idx], self.word_piece_tracked[idx], self.target[idx])

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    batch = list(zip(*batch))
    
    data = torch.stack(batch[0])
    mask = torch.stack(batch[1])
    word_piece_tracked = batch[2]
    target = torch.stack(batch[3])
    
    del batch
    return data, mask, word_piece_tracked, target

class CCGSupertaggingTrainer:
    def __init__(
        self,
        n_epochs: int,
        device: torch.device,
        model: nn.Module,
        batch_size: int,
        optimizer: torch.optim = AdamW,
        lr = 0.001
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataset: Dataset, dev_dataset: Dataset):
        train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_fn
        )

        self.model = self.model.to(self.device)
        self.model.train()

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.optimizer(params, lr = self.lr)

        for epoch in range(1, self.n_epochs + 1):
            i = 0
            for data, mask, word_piece_tracked, target in train_dataloader:
                i += 1

                data = data.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                outputs = self.model(data, mask, word_piece_tracked)

                outputs_ = outputs.view(-1, outputs.size(-1))
                target_ = target.view(-1)
                loss = self.criterion(outputs_, target_)
                print(f'[epoch {epoch}/{self.n_epochs}] averaged training loss of batch {i}/{len(train_dataloader)} = {loss.item()}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'\n======== [epoch {epoch}/{self.n_epochs}] train data evaluation ========\n')
            self.test(train_dataset, mode = 'train_eval')
            print(f'\n======== [epoch {epoch}/{self.n_epochs}] dev data evaluation ========\n')
            self.test(dev_dataset, mode = 'dev_eval')

    def test(self, dataset: Dataset, mode: str): # mode: ['train_eval', 'dev_eval', 'test_eval']
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = collate_fn
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        loss_sum = 0.
        correct_cnt = 0
        total_cnt = 0

        i = 0
        for data, mask, word_piece_tracked, target in dataloader:
            i += 1
            if i % 50 == 0:
                print(f'{mode} progress: {i}/{len(dataloader)}')

            data = data.to(self.device)
            mask = mask.to(self.device)
            target = target.to(self.device)

            outputs = self.model(data, mask, word_piece_tracked)

            outputs_ = outputs.view(-1, outputs.size(-1))
            target_ = target.view(-1)
            loss = self.criterion(outputs_, target_)
            loss_sum += loss.item()

            total_cnt += sum(
                [
                    sum([1 if cat_id >= 0 else 0 for cat_id in tgt])
                    for tgt in target
                ]
            )
            correct_cnt += (torch.argmax(outputs, dim = 2) == target).sum()

        loss_sum /= len(dataloader)
        print(f'averaged {mode} loss = {loss_sum}')
        print(f'{mode} acc = {(correct_cnt / total_cnt) * 100: .3f}')

def main(args):
    auto_parser = AutoParser()

    print('================= parsing data =================\n')
    train_data_items, categories = auto_parser.load_auto_file(args.train_data_dir)
    dev_data_items, _ = auto_parser.load_auto_file(args.dev_data_dir)
    # test_data_items, _ = auto_parser.load_auto_file(args.test_data_dir)

    category2idx = {categories[idx]: idx for idx in range(len(categories))}
    category2idx[UNK_CATEGORY] = len(category2idx)
    idx2category = {idx: category for category, idx in category2idx.items()}

    print('================= preparing data =================\n')
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_data = prepare_data(train_data_items, tokenizer, category2idx)
    dev_data = prepare_data(dev_data_items, tokenizer, category2idx)
    # test_data = prepare_data(test_data_items, tokenizer, category2idx)

    train_dataset = CCGSupertaggingDataset(
        data = train_data['input_ids'],
        mask = train_data['mask'],
        word_piece_tracked = train_data['word_piece_tracked'],
        target = train_data['target']
    )
    dev_dataset = CCGSupertaggingDataset(
        data = dev_data['input_ids'],
        mask = dev_data['mask'],
        word_piece_tracked = dev_data['word_piece_tracked'],
        target = dev_data['target']
    )

    trainer = CCGSupertaggingTrainer(
        n_epochs = args.n_epochs,
        device = args.device,
        model = CCGSupertaggerModel(
            model_path = args.model_path,
            n_classes = len(category2idx),
            dropout_p = args.dropout_p
        ),
        batch_size = args.batch_size,
        lr = args.lr
    )

    print('================= supertagging training =================\n')
    trainer.train(train_dataset = train_dataset, dev_dataset = dev_dataset)
    # trainer.test(dataset = test_dataset)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'supertagging')
    parser.add_argument('--n_epochs', type = int, default = 20)
    parser.add_argument('--device', type = torch.device, default = torch.device('cuda:0'))
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--dropout_p', type = float, default = 0.2)
    parser.add_argument('--sample_data_dir', type = str, default = '../data/ccg-sample.auto')
    parser.add_argument('--train_data_dir', type = str, default = '../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--dev_data_dir', type = str, default = '../data/ccgbank-wsj_00.auto')
    parser.add_argument('--test_data_dir', type = str, default = '../data/ccgbank-wsj_23.auto')
    parser.add_argument('--model_path', type = str, default = './models/plms/bert-base-uncased')
    args = parser.parse_args()

    main(args)