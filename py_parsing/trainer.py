import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from decoders import Decoder, ChartItem

sys.path.append('..')
from base import ConstituentNode


class CCGParsingDataset(Dataset):
    def __init__(self, pretokenized_sents, golden_trees):
        self.pretokenized_sents = pretokenized_sents
        self.golden_trees = golden_trees
    
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
            for pretokenized_sents, golden_trees in train_dataloader:
                i += 1

                predicted_charts = self.decoder.batch_decode(
                    pretokenized_sents = pretokenized_sents,
                    batch_representations = self.parsing_model(pretokenized_sents)
                )
                predicted_trees = [chart[0][-1][0] for chart in predicted_charts]

                loss = self._get_hinge_loss(predicted_trees, golden_trees)
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
        for pretokenized_sents, golden_trees in dataloader:
            i += 1
            if i % 50 == 0:
                print(f'{mode} progress: {i}/{len(dataloader)}')

            predicted_charts = self.decoder.batch_decode(
                    pretokenized_sents = pretokenized_sents,
                    batch_representations = self.parsing_model(pretokenized_sents)
                )
            predicted_trees = [chart[0][-1][0] for chart in predicted_charts]

            loss = self._get_hinge_loss(predicted_trees, golden_trees)
            loss_sum += loss.item()

        loss_sum /= len(dataloader)
        print(f'averaged {mode} loss = {loss_sum}')

    @staticmethod
    def _get_hinge_loss(predicted_trees: List[ChartItem], golden_trees: List[ChartItem]):
        loss = 0.
        for predicted_tree, golden_tree in zip(predicted_trees, golden_trees):
            loss += max(
                0,
                predicted_tree.score + _get_hamming_loss(predicted_tree, golden_tree) - golden_tree.score
            )
        return loss

    @staticmethod
    def _get_hamming_loss(predicted_tree: ChartItem, golden_tree: ChartItem):
        pass

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