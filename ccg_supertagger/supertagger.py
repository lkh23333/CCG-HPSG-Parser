from typing import List, Dict, Union, Any, TypeVar
import os, sys, numpy, re
import torch
import torch.nn as nn

from ccg_supertagger.utils import pre_tokenize_sent
from ccg_supertagger.models import BaseSupertaggingModel

sys.path.append('..')
from data_loader import load_auto_file

CategoryStr = TypeVar('CategoryStr')
SupertaggerOutput = List[List[CategoryStr]]

DATA_MASK_PADDING = 0


class CCGSupertagger:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        idx2category: Dict[int, str],
        top_k: int = 1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.idx2category = idx2category
        self.top_k = top_k

    def _prepare_batch_data(self, batch: List[List[str]]) -> Dict[str, Any]:
        # batch: a list of pretokenized sentences (a list of strings)
        data = list() # a list containing a list of input_ids for each sentence
        mask = list() # a list containing the attention mask list for each sentence
        word_piece_tracked = list() # a list containing the list of word_piece_tracked for each sentence

        for pretokenized_sent in batch:
            word_piece_tracked.append(
                [len(item) for item in self.tokenizer(pretokenized_sent, add_special_tokens = False).input_ids]
            )

            inputs = self.tokenizer(
                pretokenized_sent,
                add_special_tokens = False,
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

    def _convert_model_outputs(self, outputs: List[torch.Tensor]) -> List[SupertaggerOutput]:
        # outputs: a list of tensors, each of the shape of the length of one sentence * C
        batch_predicted = list()
        for output in outputs:
            predicted = list()
            for i in range(output.shape[0]):
                topk_ids = torch.topk(output[i], self.top_k)[1]
                predicted.append([self.idx2category[int(idx.item())] for idx in topk_ids])
            batch_predicted.append(predicted)
        return batch_predicted

    def _load_model_checkpoint(self, checkpoints_dir: str, checkpoint_epoch: int):
        checkpoint = torch.load(os.path.join(checkpoints_dir, f'epoch_{checkpoint_epoch}.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_model_outputs_for_batch(self, batch: List[Union[str, List[str]]]) -> List[torch.Tensor]:
        for i in range(len(batch)):
            if isinstance(batch[i], str):
                batch[i] = pre_tokenize_sent(batch[i])
        
        batch_data = self._prepare_batch_data(batch)
        outputs = self.model(
            encoded_batch = batch_data['input_ids'],
            mask = batch_data['mask'],
            word_piece_tracked = batch_data['word_piece_tracked']
        ) # B*L*C
        sents_lengths = [len(word_piece_tracked) for word_piece_tracked in batch_data['word_piece_tracked']]
        
        softmax = nn.Softmax(dim = 1)
        return [
            softmax(outputs[i, :sents_lengths[i], :])
            for i in range(len(batch))
        ] # a list, each of the shape l_sent * C

    def get_model_outputs_for_sent(self, sent: Union[str, List[str]]) -> torch.Tensor:
        return self.get_model_outputs_for_batch([sent])[0] # L*C -> length of this sentence *C

    def predict_batch(self, batch: List[Union[str, List[str]]]) -> List[SupertaggerOutput]:
        outputs = self.get_model_outputs_for_batch(batch)
        return self._convert_model_outputs(outputs)

    def predict_sent(self, sent: Union[str, List[str]]) -> SupertaggerOutput:
        return self.predict_batch([sent])[0]


if __name__ == '__main__':
    # sample use
    import json
    lexical_category2idx_dir = '../data/lexical_category2idx_from_train_data.json'
    with open(lexical_category2idx_dir, 'r', encoding = 'utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: category for category, idx in category2idx.items()}

    from transformers import BertTokenizer
    model_path = './models/plms/bert-base-uncased'
    supertagger = CCGSupertagger(
        model = BaseSupertaggingModel(model_path, len(category2idx)),
        tokenizer = BertTokenizer.from_pretrained(model_path),
        idx2category = idx2category
    )
    checkpoints_dir = './checkpoints'
    checkpoint_epoch = 5
    supertagger._load_model_checkpoint(checkpoints_dir, checkpoint_epoch)

    sent = 'Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group'
    predicted = supertagger.predict_batch([sent, sent])
    print(predicted)