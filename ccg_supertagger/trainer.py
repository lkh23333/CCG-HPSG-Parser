from typing import Union, Optional, Dict, Callable, Any
import sys, os, argparse, random, json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from models import BaseSupertaggingModel, LSTMSupertaggingModel, LSTMCRFSupertaggingModel
from utils import prepare_data

sys.path.append('..')
from data_loader import load_auto_file


# to set the random seeds
def _setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


_setup_seed(0)


# to set the random seed of the dataloader
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class CCGSupertaggingDataset(Dataset):
    def __init__(self, data, mask, word_piece_tracked, target):
        self.data = data
        self.mask = mask
        self.word_piece_tracked = word_piece_tracked
        self.target = target

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx], self.word_piece_tracked[idx], self.target[idx]

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
            checkpoints_dir: str,
            train_dataset: Dataset,
            dev_dataset: Dataset,
            test_dataset: Dataset = None,
            optimizer: torch.optim = AdamW,
            lr=0.00001,
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.checkpoints_dir = checkpoints_dir
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=2)

    def train(self, checkpoint_epoch: int = 0, print_every: int = 50):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
            num_workers=0,
            generator=g
        )

        self.model.to(self.device)

        if isinstance(self.optimizer, Callable):
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = self.optimizer(params, lr=self.lr)

        best_epoch = -1
        best_dev_acc = -1

        for epoch in range(checkpoint_epoch + 1, self.n_epochs + 1):
            self.model.train()

            i = 0
            for data, mask, word_piece_tracked, target in train_dataloader:
                i += 1

                data = data.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                if self.model.__class__.__name__ == 'LSTMCRFSupertaggingModel':
                    outputs = self.model(data, target, mask, word_piece_tracked)
                    loss = outputs
                else:
                    outputs = self.model(data, mask, word_piece_tracked)

                    outputs_ = outputs.view(-1, outputs.size(-1))
                    target_ = target.view(-1)
                    loss = self.criterion(outputs_, target_)

                if i % print_every == 0:
                    print(
                        f'[epoch {epoch}/{self.n_epochs}] averaged training loss of batch {i}/{len(train_dataloader)} = {loss.item()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'\n======== [epoch {epoch}/{self.n_epochs}] saving the checkpoint ========\n')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                f=os.path.join(self.checkpoints_dir, f'epoch_{epoch}.pt')
            )

            with torch.no_grad():
                print(f'\n======== [epoch {epoch}/{self.n_epochs}] train data evaluation ========\n')
                _ = self.test(self.train_dataset, mode='train_eval')
                print(f'\n======== [epoch {epoch}/{self.n_epochs}] dev data evaluation ========\n')
                dev_acc = self.test(self.dev_dataset, mode='dev_eval')

                if dev_acc > best_dev_acc:
                    best_epoch = epoch
                    best_dev_acc = dev_acc

        print('\n#Params: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print(f'\nBest epoch = {best_epoch} with dev_eval acc = {best_dev_acc}\n')

    def test(self, dataset: Dataset, mode: str):  # modes: ['train_eval', 'dev_eval', 'test_eval']

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            worker_init_fn=_seed_worker,
            num_workers=0,
            generator=g
        )

        self.model.to(self.device)
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

            if self.model.__class__.__name__ == 'LSTMCRFSupertaggingModel':
                outputs = self.model(data, target, mask, word_piece_tracked)
                loss = outputs

                predicted_tags = self.model.predict(data, mask, word_piece_tracked)

                predicted = torch.empty_like(target).fill_(-1)
                for j in range(predicted.shape[0]):
                    predicted[j, 0:len(predicted_tags[j])] = torch.tensor(predicted_tags[j])

                correct_cnt += (predicted == target).sum()
            else:
                outputs = self.model(data, mask, word_piece_tracked)

                outputs_ = outputs.view(-1, outputs.size(-1))
                target_ = target.view(-1)
                loss = self.criterion(outputs_, target_)

                outputs = self.softmax(outputs)
                correct_cnt += (torch.argmax(outputs, dim=2) == target).sum()

            loss_sum += loss.item()

            total_cnt += sum(
                [
                    len(word_pieces) for word_pieces in word_piece_tracked
                ]
            )

        loss_sum /= len(dataloader)
        acc = (correct_cnt / total_cnt) * 100
        print(f'averaged {mode} loss = {loss_sum}')
        print(f'{mode} acc = {acc: .3f}')

        return acc

    def load_checkpoint_and_train(self, checkpoint_epoch: int):  # set the epoch from which to restart training
        checkpoint = torch.load(
            os.path.join(self.checkpoints_dir, f'epoch_{checkpoint_epoch}.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.optimizer(params, lr=self.lr)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']

        self.train(checkpoint_epoch=epoch)

    def load_checkpoint_and_test(self, checkpoint_epoch: int,
                                 mode: str):  # modes: ['train_eval', 'dev_eval', 'test_eval']
        checkpoint = torch.load(
            os.path.join(self.checkpoints_dir, f'epoch_{checkpoint_epoch}.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            if mode == 'train_eval':
                self.test(self.train_dataset, mode='train_eval')
            elif mode == 'dev_eval':
                self.test(self.dev_dataset, mode='dev_eval')
            elif mode == 'test_eval':
                self.test(self.test_dataset, mode='test_eval')
            else:
                raise ValueError('the mode should be one of train_eval, dev_eval and test_eval')


def main(args):
    print('================= parsing data =================\n')
    # train_data_items, _ = load_auto_file(args.train_data_dir)
    dev_data_items, _ = load_auto_file(args.dev_data_dir)
    # test_data_items, _ = load_auto_file(args.test_data_dir)

    with open(args.lexical_category2idx_dir, 'r', encoding='utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: category for category, idx in category2idx.items()}

    # # for testing codes
    # categories = sorted(categories) # !!! to ensure the same order for reproducibility !!!
    # category2idx = {categories[idx]: idx for idx in range(len(categories))}
    # idx2category = {idx: category for category, idx in category2idx.items()}

    print('================= preparing data =================\n')
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    # train_data = prepare_data(train_data_items, tokenizer, category2idx)
    dev_data = prepare_data(dev_data_items, tokenizer, category2idx)
    # test_data = prepare_data(test_data_items, tokenizer, category2idx)

    # train_dataset = CCGSupertaggingDataset(
    #     data=train_data['input_ids'],
    #     mask=train_data['mask'],
    #     word_piece_tracked=train_data['word_piece_tracked'],
    #     target=train_data['target']
    # )
    dev_dataset = CCGSupertaggingDataset(
        data=dev_data['input_ids'],
        mask=dev_data['mask'],
        word_piece_tracked=dev_data['word_piece_tracked'],
        target=dev_data['target']
    )

    model = BaseSupertaggingModel
    if args.model_name == 'lstm':
        model = LSTMSupertaggingModel
    elif args.model_name == 'lstm-crf':
        model = LSTMCRFSupertaggingModel

    trainer = CCGSupertaggingTrainer(
        n_epochs=args.n_epochs,
        device=args.device,
        model=model(
            model_path=args.model_path,
            n_classes=len(category2idx),
            embed_dim=args.embed_dim,
            num_lstm_layers=args.num_lstm_layers,
            dropout_p=args.dropout_p
        ),
        batch_size=args.batch_size,
        checkpoints_dir=args.checkpoints_dir,
        train_dataset=dev_dataset,
        dev_dataset=dev_dataset,
        lr=args.lr
    )

    print('================= supertagging training =================\n')
    # trainer.train()  # default training from the beginning
    # trainer.load_checkpoint_and_train(checkpoint_epoch=2) # train from (checkpoint_epoch + 1)
    trainer.load_checkpoint_and_test(checkpoint_epoch=19, mode='dev_eval')
    # trainer.test(dataset = self.test_dataset, mode = 'test_eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='supertagging')
    parser.add_argument('--model_name', type=str, default='lstm', choices=['fc', 'lstm', 'lstm-crf'])
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'))
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--num_lstm_layers', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--sample_data_dir', type=str, default='../data/ccg-sample.auto')
    parser.add_argument('--train_data_dir', type=str, default='../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--dev_data_dir', type=str, default='../data/ccgbank-wsj_00.auto')
    parser.add_argument('--test_data_dir', type=str, default='../data/ccgbank-wsj_23.auto')
    parser.add_argument('--lexical_category2idx_dir', type=str, default='../data/lexical_category2idx_cutoff.json')
    parser.add_argument('--model_path', type=str, default='../plms/bert-large-uncased')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    main(args)