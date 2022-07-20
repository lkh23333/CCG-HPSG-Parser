from typing import Union, Optional, Dict, Any
import sys, os, argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from models.simple_model import CCGSupertaggerModel
from utils import prepare_data, get_cross_entropy_loss

sys.path.append('..')
from data_loader import AutoParser


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

    def train(self, dataset: Dataset):
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_fn
        )

        self.model.to(self.device)
        self.model.train()

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.optimizer(params, lr = self.lr)

        for epoch in range(1, self.n_epochs + 1):
            print(f'=============== {epoch}/{self.n_epochs} ===============\n')
            loss_sum = 0.
            correct_cnt = 0
            total_cnt = 0

            for data, mask, word_piece_tracked, target in dataloader:
                data = data.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                outputs = self.model(data, mask, word_piece_tracked)
                calculate = get_cross_entropy_loss(outputs, target)
                loss = calculate[0]
                loss_sum += loss.item()
                total_cnt += calculate[1]
                print(f'training loss = {loss.item()}\n')
                correct_cnt += (torch.argmax(outputs, dim = 2) == target).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_sum /= len(dataloader)
            print(f'======== acc: {(correct_cnt / total_cnt) * 100: .3f} ========\n')


    def test(self, dataset: Dataset):
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_fn
        )

        self.model.to(self.device)
        self.model.eval()

        loss_sum = 0.
        for data, mask, word_piece_tracked, target in dataloader:
            data = data.to(self.device)
            mask = mask.to(self.device)
            target = target.to(self.device)

            outputs = model(data, mask, word_piece_tracked)
            loss = F.cross_entropy(outputs.permute(0, 2, 1), target.permute(0, 1))
            loss_sum += loss.item()
            print(f'testing loss = {loss.item()}\n')

        loss_sum /= len(dataloader)

def main(args):
    auto_parser = AutoParser()

    # train_data_items, categories = auto_parser.load_auto_file(args.train_data_dir)
    # dev_data_items, categories = auto_parser.load_auto_file(args.dev_data_dir)
    # test_data_items, _ = auto_parser.load_auto_file(args.test_data_dir)
    data_items, categories = auto_parser.load_auto_file(args.sample_data_dir)

    category2idx = {categories[idx]: idx for idx in range(len(categories))}
    idx2category = {idx: category for category, idx in category2idx.items()}

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    # train_data = prepare_data(train_data_items, tokenizer, category2idx)
    # dev_data = prepare_data(dev_data_items, tokenizer, category2idx)
    # test_data = prepare_data(test_data_items, tokenizer, category2idx)
    data = prepare_data(data_items, tokenizer, category2idx)

    dataset = CCGSupertaggingDataset(
        data = data['input_ids'],
        mask = data['mask'],
        word_piece_tracked = data['word_piece_tracked'],
        target = data['target']
    )

    trainer = CCGSupertaggingTrainer(
        n_epochs = args.n_epochs,
        device = args.device,
        model = CCGSupertaggerModel(
            model_path = args.model_path,
            n_classes = len(category2idx),
            dropout_p = args.dropout_p
        ),
        batch_size = args.batch_size
    )

    trainer.train(dataset)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'supertagging')
    parser.add_argument('--n_epochs', type = int, default = 20)
    parser.add_argument('--device', type = torch.device, default = torch.device('cuda:0'))
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--dropout_p', type = float, default = 0.2)
    parser.add_argument('--sample_data_dir', type = str, default = '../data/ccg-sample.auto')
    parser.add_argument('--train_data_dir', type = str, default = '../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--dev_data_dir', type = str, default = '../data/ccgbank-wsj_00.auto')
    parser.add_argument('--test_data_dir', type = str, default = '../data/ccgbank-wsj_23.auto')
    parser.add_argument('--model_path', type = str, default = './models/plms/bert-base-uncased')
    args = parser.parse_args()

    main(args)