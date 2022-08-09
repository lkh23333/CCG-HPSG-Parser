import sys
from typing import *
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

sys.path.append('..')
from ccg_supertagger.supertagger import CCGSupertagger
from ccg_supertagger.models import BaseSupertaggingModel

DATA_MASK_PADDING = 0

SupertaggingRepresentations = torch.Tensor # l_sent*supertagging_n_classes
SpanRepresentations = torch.Tensor # l_sent*(l_sent+1)*parsing_n_classes

class BaseParsingModel(nn.Module):

    def __init__(
        self,
        model_path: str,
        supertagging_n_classes: int,
        checkpoints_dir: str,
        checkpoint_epoch: int
    ):
        super().__init__()
        self.supertagger = CCGSupertagger(
            model = BaseSupertaggingModel(model_path, supertagging_n_classes),
            tokenizer = BertTokenizer.from_pretrained(model_path)
        )
        self.supertagger._load_model_checkpoint(checkpoints_dir, checkpoint_epoch)

    def forward(self, pretokenized_sents: List[List[str]]) -> List[SupertaggingRepresentations]:
        # return the embedding of each word in every sentence from the base supertagging model
        return self.supertagger.get_model_outputs_for_batch(pretokenized_sents)

class SpanParsingModel(BaseParsingModel):

    def __init(
        self,
        model_path: str,
        supertagging_n_classes: int,
        parsing_n_classes: int,
        checkpoints_dir: str,
        checkpoint_epoch: int
    ):
        super().__init__(
            model_path = model_path,
            supertagging_n_classes = supertagging_n_classes,
            checkpoints_dir = checkpoints_dir,
            checkpoint_epoch = checkpoint_epoch
        )
        self.bert = self.supertagger.model.bert
        self.tokenizer = self.supertagger.tokenizer
        self.w1 = nn.Linear(768, 2048)
        self.w2 = nn.Linear(2048, parsing_n_classes)
        self.parsing_n_classes = parsing_n_classes
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(2048)

    def _prepare_batch_data(self, pretokenized_sents: List[List[str]]):
        # return wrapped data needed to input into the model.
        data = list() # a list containing a list of input_ids for each sentence
        mask = list() # a list containing the attention mask list for each sentence
        word_piece_tracked = list() # a list containing the list of word_piece_tracked for each sentence

        for pretokenized_sent in pretokenized_sents:
            word_piece_tracked.append(
                [len(item) for item in self.tokenizer(pretokenized_sent).input_ids]
            )

            inputs = self.tokenizer(
                pretokenized_sent,
                is_split_into_words = True
            )
            data.append(inputs.input_ids)
            mask.append(inputs.attention_mask)

        max_length = max([len(input_ids) for input_ids in data])
        for i in range(len(data)):
            assert len(data[i]) == len(mask[i])
            data[i] = data[i] + [DATA_MASK_PADDING] * (max_length - len(data[i])) # padding
            mask[i] = mask[i] + [DATA_MASK_PADDING] * (max_length - len(mask[i])) # padding

        return {
            'input_ids': torch.LongTensor(data),
            'mask': torch.FloatTensor(mask),
            'word_piece_tracked': word_piece_tracked
        }

    def forward(self, pretokenized_sents: List[List[str]]) -> Tuple[List[SupertaggingRepresentations], List[SpanRepresentations]]:
        # return the supertagging outputs for the sentences
        # and the embeddings of each word, [SEP] (start) and [CLS] (end) token in every sentence from the BERT in the supertagger
        batch_supertagging_outputs = self.supertagger.get_model_outputs_for_batch(pretokenized_sents)
        
        batch_data = self._prepare_batch_data(pretokenized_sents)
        batch_input_ids = data['input_ids'] # B * (1([CLS]) + n_word_pieces + 1([SEP]) + n_paddings)
        batch_mask = data['mask'] # B * (1([CLS]) + n_word_pieces + 1([SEP]) + n_paddings)
        batch_word_piece_tracked = data['word_piece_tracked'] # B lists, each of the length l_sent

        batch_span_representations = list()
        for l in range(len(batch_word_piece_tracked)):

            input_ids = batch_input_ids[l]
            mask = batch_mask[l]
            word_piece_tracked = batch_word_piece_tracked[l]

            span_representations = torch.zeros(
                (
                    len(word_piece_tracked),
                    len(word_piece_tracked) + 1,
                    self.parsing_n_classes
                )
            )

            f0 = torch.squeeze(
                self.bert(
                    input_ids = input_ids,
                    attention_mask = mask
                ).last_hidden_state
            ) # L*H, note that L = 1([CLS]) + n_word_pieces + 1([SEP])

            f1 = torch.zeros((2 + len(word_piece_tracked), f0.shape[1])) # (l_sent+2)*H
            f1[0] = f0[0]
            f1[-1] = f0[-1] # assign [CLS] and [SEP]
            k = 1
            for j in range(1, len(word_piece_tracked) + 1):
                n_piece = word_piece_tracked[j - 1]
                f1[j] = torch.sum(f0[k:k+n_piece], dim = 0) / n_piece # to take the average of word pieces
                k += n_piece

            word_representation_odd = f1[:, ::2]
            word_representation_even = f1[:, 1::2]

            for i in range(len(word_piece_tracked)):
                for j in range(i + 1, len(word_piece_tracked) + 1):
                    span_representations[i, j] = torch.cat(
                        [
                            word_representation_even[j] - word_representation_even[i],
                            word_representation_odd[j + 1] - word_representation_odd[i + 1]
                        ]
                    )

            span_representations = self.w2(
                self.relu(
                    self.layer_norm(
                        self.w1(
                            span_representations
                        )
                    )
                )
            )
            batch_span_representations.append(span_representations)

        return batch_supertagging_outputs, batch_span_representations