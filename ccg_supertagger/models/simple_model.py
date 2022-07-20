from typing import List
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel


class CCGSupertaggerModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        n_classes: int,
        dropout_p: float = 0.2
    ):
        super(CCGSupertaggerModel, self).__init__()
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
        f1 = self.w1(
            self.dropout(
                self.relu(f0)
            )
        )
        f2 = self.w2(
            self.dropout(
                self.relu(f1)
            )
        )
        for i in range(f2.shape[0]):
            k = 0
            for j in range(len(word_piece_tracked[i])):
                n_piece = word_piece_tracked[i][j]
                f2[i, j] = torch.sum(f2[i, k:k+n_piece], dim = 0) / n_piece # to take the average of word pieces
                k += n_piece

        return self.softmax(f2) # B*L*C (C the number of classes)