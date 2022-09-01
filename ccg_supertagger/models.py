from typing import List
import os, random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


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

class BaseSupertaggingModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        n_classes: int,
        dropout_p: float = 0.2
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.w1 = nn.Linear(768, 1024)
        self.w2 = nn.Linear(1024, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_p)
        self.softmax = nn.Softmax(dim = 2)
    
    def forward(self, encoded_batch, mask, word_piece_tracked: List[List[int]]):
        f0 = self.bert(
            input_ids = encoded_batch,
            attention_mask = mask
        ).last_hidden_state # B*L*H
        
        for i in range(f0.shape[0]):
            k = 0
            for j in range(len(word_piece_tracked[i])):
                n_piece = word_piece_tracked[i][j]
                f0[i, j] = torch.sum(f0[i, k:k+n_piece], dim = 0) / n_piece # to take the average of word pieces
                k += n_piece
        
        f1 = self.dropout(
            self.relu(
                self.w1(f0)
            )
        )
        f2 = self.dropout(
            self.relu(
                self.w2(f1)
            )
        )

        return f2 # B*L*C (C the number of classes)


class LSTMSupertaggingModel(nn.Module):
    def __init__(
            self,
            model_path: str,
            n_classes: int,
            lstm_dim: int = 384,
            dropout_p: float = 0.2
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_dim, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=dropout_p)
        self.linear = nn.Linear(in_features=lstm_dim*2, out_features=n_classes)

    def forward(self, encoded_batch, mask, word_piece_tracked: List[List[int]]):
        f0 = self.bert(
            input_ids=encoded_batch,
            attention_mask=mask
        ).last_hidden_state  # B*L*H

        for i in range(f0.shape[0]):
            k = 0
            for j in range(len(word_piece_tracked[i])):
                n_piece = word_piece_tracked[i][j]
                f0[i, j] = torch.sum(f0[i, k:k + n_piece], dim=0) / n_piece  # to take the average of word pieces
                k += n_piece

        f1, _ = self.lstm(f0)

        f2 = self.dropout(
                self.linear(f1)
            )

        return f2  # B*L*C (C the number of classes)


class LSTMCRFSupertaggingModel(nn.Module):
    def __init__(
            self,
            model_path: str,
            n_classes: int,
            lstm_dim: int = 384,
            dropout_p: float = 0.2
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_dim, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=dropout_p)
        self.linear = nn.Linear(in_features=lstm_dim*2, out_features=n_classes)
        self.crf = CRF(num_tags=n_classes, batch_first=True)

    def _get_lstm_features(self, encoded_batch, mask, word_piece_tracked: List[List[int]]):
        f0 = self.bert(
            input_ids=encoded_batch,
            attention_mask=mask
        ).last_hidden_state  # B*L*H

        crf_mask = torch.zeros_like(mask)

        for i in range(f0.shape[0]):
            k = 0
            for j in range(len(word_piece_tracked[i])):
                n_piece = word_piece_tracked[i][j]
                f0[i, j] = torch.sum(f0[i, k:k + n_piece], dim=0) / n_piece  # to take the average of word pieces
                crf_mask[i, j] = 1
                k += n_piece

        f1, _ = self.lstm(f0)

        f2 = self.dropout(
                self.linear(f1)
            )

        return f2, crf_mask  # B*L*C (C the number of classes)

    def forward(self, encoded_batch, tags, mask, word_piece_tracked: List[List[int]]):
        emissions, crf_mask = self._get_lstm_features(encoded_batch, mask, word_piece_tracked)
        loss = -1 * self.crf(emissions, tags, mask=crf_mask.byte(), reduction='mean')

        return loss


    def predict(self, encoded_batch, mask, word_piece_tracked: List[List[int]]):
        emissions, crf_mask = self._get_lstm_features(encoded_batch, mask, word_piece_tracked)

        return self.crf.decode(emissions, mask=crf_mask.byte())