"""
!!!Unfinished!!!
"""

import sys, json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from ccg_parsing_models import SpanParsingModel
from decoders import ChartItem, Decoder, CCGSpanDecoder
from utils import get_pretokenized_sents

sys.path.append('..')
from base import ConstituentNode
from data_loader import DataItem, load_auto_file


class CCGParsingDataset(Dataset):
    def __init__(self, pretokenized_sents: List[List[str]], data_items: List[DataItem]):
        self.pretokenized_sents = pretokenized_sents
        self.data_items = data_items
    
    def __getitem__(self, idx):
        return (self.pretokenized_sents[idx], self.golden_trees[idx])

    def __len__(self):
        return len(self.pretokenized_sents)

class CCGParsingTrainer:

    def __init__(
        self,
        parsing_model: nn.Module,
        decoder: Decoder,
        n_epochs: int,
        device: torch.device,
        batch_size: int,
        checkpoints_dir: str,
        train_dataset: Dataset,
        dev_dataset: Dataset,
        test_dataset: Dataset = None,
        optimizer: torch.optim = AdamW,
        lr: float = 1e-4,
    ):
        self.parsing_model = parsing_model
        self.decoder = decoder
        self.n_epochs = n_epochs
        self.device = device
        self.batch_size = batch_size
        self.checkpoints_dir = checkpoints_dir
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.lr = lr

    def train(self, checkpoint_epoch: int = 0):
        
        train_dataloader = DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )

        self.parsing_model.to(self.device)
        self.parsing_model.train()
        self.decoder.mode = 'train'

        if isinstance(self.optimizer, Callable):
            params = filter(lambda p: p.requires_grad, self.parsing_model.parameters())
            self.optimizer = self.optimizer(params, lr = self.lr)

        for epoch in range(checkpoint_epoch + 1, self.n_epochs + 1):
            i = 0
            for pretokenized_sents, data_items in train_dataloader:
                i += 1

                batch_representations = self.parsing_model(pretokenized_sents)
                golden_charts = [
                    self.decoder.get_golden_chart(data_items[i], batch_representations[i])
                    for i in range(len(batch_representations))
                ]
                predicted_charts = self.decoder.batch_decode(
                    pretokenized_sents = pretokenized_sents,
                    batch_representations = batch_representations,
                    golden_charts = golden_charts
                )

                loss = self._get_loss(predicted_charts, golden_charts)
                print(f'[epoch {epoch}/{self.n_epochs}] averaged training loss of batch {i}/{len(train_dataloader)} = {loss.item()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'\n======== [epoch {epoch}/{self.n_epochs}] saving the checkpoint ========\n')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.parsing_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                f = os.path.join(self.checkpoints_dir, f'epoch_{epoch}.pt')
            )

            print(f'\n======== [epoch {epoch}/{self.n_epochs}] train data evaluation ========\n')
            self.test(self.train_dataset, mode = 'train_eval')
            print(f'\n======== [epoch {epoch}/{self.n_epochs}] dev data evaluation ========\n')
            self.test(self.dev_dataset, mode = 'dev_eval')

    def test(self, dataset: Dataset, mode: str): # modes: ['train_eval', 'dev_eval', 'test_eval']
        
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = False
        )

        self.parsing_model.to(self.device)
        self.parsing_model.eval()
        self.decoder.mode = 'test'

        loss_sum = 0.

        i = 0
        for pretokenized_sents, data_items in dataloader:
            i += 1
            if i % 50 == 0:
                print(f'{mode} progress: {i}/{len(dataloader)}')

            batch_representations = self.parsing_model(pretokenized_sents)
            golden_charts = [
                self.decoder.get_golden_chart(data_items[i], batch_representations[i])
                for i in range(len(batch_representations))
            ]
            predicted_charts = self.decoder.batch_decode(
                pretokenized_sents = pretokenized_sents,
                batch_representations = batch_representations,
                golden_charts = golden_charts
            )

            loss = self._get_loss(predicted_charts, golden_charts)
            loss_sum += loss.item()

        loss_sum /= len(dataloader)
        print(f'averaged {mode} loss = {loss_sum}')

    @staticmethod
    def _get_loss(predicted_charts: List[Chart], golden_charts: List[Chart]):
        loss = 0.
        for predicted_chart, golden_chart in zip(predicted_charts, golden_charts):
            predicted_score = predicted_chart.chart[0][-1].cell_items[0].score
            golden_score = golden_chart.chart[0][-1].cell_items[0].score
            loss += max(
                0,
                predicted_score - golden_score
            )
        return loss

    def load_checkpoint_and_train(self, checkpoint_epoch: int): # set the epoch from which to restart training
        checkpoint = torch.load(
            os.path.join(self.checkpoints_dir, f'epoch_{checkpoint_epoch}.pt'),
            map_location = self.device
        )
        self.parsing_model.load_state_dict(checkpoint['model_state_dict'])
        self.parsing_model.to(self.device)

        params = filter(lambda p: p.requires_grad, self.parsing_model.parameters())
        self.optimizer = self.optimizer(params, lr = self.lr)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']

        self.train(checkpoint_epoch = epoch)

    def load_checkpoint_and_test(self, checkpoint_epoch: int, mode: str): # modes: ['train_eval', 'dev_eval', 'test_eval']
        checkpoint = torch.load(
            os.path.join(self.checkpoints_dir, f'epoch_{checkpoint_epoch}.pt'),
            map_location = self.device
        )
        self.parsing_model.load_state_dict(checkpoint['model_state_dict'])

        if mode == 'train_eval':
            self.test(self.train_dataset, mode = 'train_eval')
        elif mode == 'dev_eval':
            self.test(self.dev_dataset, mode = 'dev_eval')
        elif mode == 'test_eval':
            self.test(self.test_dataset, mode = 'test_eval')
        else:
            raise ValueError('the mode should be one of train_eval, dev_eval and test_eval')

def main(args):

    print('================= parsing data =================\n')
    # train_data_items, _ = load_auto_file(args.train_data_dir)
    dev_data_items, _ = load_auto_file(args.dev_data_dir)
    # test_data_items, _ = load_auto_file(args.test_data_dir)

    print('================= getting pretokenized sents =================\n')
    # train_pretokenized_sents = get_pretokenized_sents(train_data_items)
    dev_pretokenized_sents = get_pretokenized_sents(dev_data_items)
    # test_pretokenized_sents = get_pretokenized_sents(test_data_items)

    with open(args.lexical_category2idx_dir, 'r', encoding = 'utf8') as f:
        lexical_category2idx = json.load(f)

    with open(args.parsing_category2idx_dir, 'r', encoding = 'utf8') as f:
        parsing_category2idx = json.load(f)
    parsing_idx2category = {idx: category for category, idx in category2idx.items()}

    # train_dataset = CCGParsingDataset(
    #     pretokenized_sents = train_pretokenized_sents,
    #     data_items = train_data_items
    # )
    dev_dataset = CCGParsingDataset(
        pretokenized_sents = dev_pretokenized_sents,
        data_items = dev_data_items
    )

    trainer = CCGParsingTrainer(
        parsing_model = SpanParsingModel(
            model_path = args.supertagging_model_path,
            supertagging_n_classes = len(lexical_category2idx),
            parsing_n_classes = len(parsing_category2idx),
            checkpoints_dir = args.supertagging_model_checkpoints_dir,
            checkpoint_epoch = args.supertagging_model_checkpoint_epoch
        ),
        decoder = CCGSpanDecoder(
            beam_width = args.beam_width,
            idx2tag = parsing_idx2category
        ),
        n_epochs = args.n_epochs,
        device = args.device,
        batch_size = args.batch_size,
        checkpoints_dir = args.parsing_model_checkpoints_dir,
        train_dataset = train_dataset,
        dev_dataset = dev_dataset,
        lr = args.lr
    )

    print('================= parsing training =================\n')
    trainer.train() # default training from the beginning
    # trainer.load_checkpoint_and_train(checkpoint_epoch=1) # train from (checkpoint_epoch + 1)
    # trainer.load_checkpoint_and_test(checkpoint_epoch=1, mode='train_eval')
    # trainer.test(dataset = self.test_dataset, mode = 'test_eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'parsing')
    parser.add_argument('--train_data_dir', type = str, default = '../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--dev_data_dir', type = str, default = '../data/ccgbank-wsj_00.auto')
    parser.add_argument('--test_data_dir', type = str, default = '../data/ccgbank-wsj_23.auto')
    parser.add_argument('--lexical_category2idx_dir', type = str, default = '../data/lexical_category2idx_cutoff.json')
    parser.add_argument('--parsing_category2idx_dir', type = str, default = '../data/parsing_categoey2idx.json')
    parser.add_argument('--instantiated_unary_rules_dir', type = str, default = '../data/instantiated_unary_rules_with_X.json')
    parser.add_argument('--instantiated_binary_rules_dir', type = str, default = '../data/instantiated_seen_binary_rules.json')
    parser.add_argument('--supertagging_model_path', type = str, default = '../plms/bert-base-uncased')
    parser.add_argument('--supertagging_model_checkpoints_dir', type = str, default = '../ccg_supertagger/checkpoints')
    parser.add_argument('--supertagging_model_checkpoint_epoch', type = str, default = 2)
    parser.add_argument('--device', type = torch.device, default = torch.device('cuda:2'))
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--beam_width', type = int, default = 5)
    parser.add_argument('--parsing_model_checkpoints_dir', type = str, default = './checkpoints')
    parser.add_argument('--parsing_model_checkpoint_epoch', type = str, default = 2)
    parser.add_argument('--decoder_timeout', help = 'time out value for decoding one sentence', type = float, default = 16.0)
    parser.add_argument('--predicted_auto_files_dir', type = str, default = './evaluation')
    args = parser.parse_args()

    main(args)