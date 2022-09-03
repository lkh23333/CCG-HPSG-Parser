from typing import List, Dict, Union, Any, TypeVar
import os, sys, numpy, re
import torch
import torch.nn as nn

sys.path.append('..')
from ccg_supertagger.utils import pre_tokenize_sent
from ccg_supertagger.models import BaseSupertaggingModel, LSTMSupertaggingModel
from base import Category
from data_loader import load_auto_file

CategoryStr = TypeVar('CategoryStr')
SupertaggerOutput = List[List[CategoryStr]]

DATA_MASK_PADDING = 0


class CCGSupertagger:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        idx2category: Dict[int, str] = None,
        top_k: int = 1,
        beta: float = 1e-5, # pruning parameter for supertagging
        device: torch.device = torch.device('cuda')
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.idx2category = idx2category
        if idx2category is not None:
            self.category2idx = {idx: cat for idx, cat in idx2category.items()}
        self.top_k = top_k
        self.beta = beta
        self.device = device
        self.softmax = nn.Softmax(dim = 2)

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
        if self.idx2category == None:
            raise RuntimeError('Please specify idx2category in the supertagger!!!')
            
        outputs = self._prune(outputs)

        batch_predicted = list()
        for output in outputs:
            predicted = list()
            for i in range(output.shape[0]):
                # topk_ids = torch.topk(output[i], self.top_k)[1]
                topk_ps, topk_ids = torch.topk(output[i], self.top_k)
                ids = topk_ids[topk_ps > 0]
                # predicted.append([str(Category.parse(self.idx2category[idx.item()])) for idx in topk_ids])
                predicted.append([str(Category.parse(self.idx2category[idx.item()])) for idx in ids])
            batch_predicted.append(predicted)
        return batch_predicted

    def _prune(self, outputs) -> torch.Tensor:
        # assign all probabilities beta times less than the best one to 0
        for output in outputs:
            for i in range(output.shape[0]):
                top_p = torch.topk(output[i], 1)[0]
                binarized = (output[i] > self.beta * top_p)
                output[i] = output[i] * binarized

        return outputs

    def _load_model_checkpoint(self, checkpoints_dir: str, checkpoint_epoch: int):
        checkpoint = torch.load(
            os.path.join(checkpoints_dir, f'epoch_{checkpoint_epoch}.pt'),
            map_location = self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_model_outputs_for_batch(self, batch: List[Union[str, List[str]]]) -> List[torch.Tensor]:
        
        self.model.to(self.device)
        self.model.eval()
        
        for i in range(len(batch)):
            if isinstance(batch[i], str):
                batch[i] = pre_tokenize_sent(batch[i])
        
        batch_data = self._prepare_batch_data(batch)
        batch_data['input_ids'] = batch_data['input_ids'].to(self.device)
        batch_data['mask'] = batch_data['mask'].to(self.device)
        outputs = self.model(
            encoded_batch = batch_data['input_ids'],
            mask = batch_data['mask'],
            word_piece_tracked = batch_data['word_piece_tracked']
        ) # B*L*C
        outputs = self.softmax(outputs)

        sents_lengths = [len(word_piece_tracked) for word_piece_tracked in batch_data['word_piece_tracked']]

        return [
            outputs[i, :sents_lengths[i], :]
            for i in range(len(batch))
        ] # a list, each of the shape l_sent * C

    def get_model_outputs_for_sent(self, sent: Union[str, List[str]]) -> torch.Tensor:
        return self.get_model_outputs_for_batch([sent])[0] # L*C -> length of this sentence *C

    def predict_batch(self, batch: List[Union[str, List[str]]]) -> List[SupertaggerOutput]:
        outputs = self.get_model_outputs_for_batch(batch)
        return self._convert_model_outputs(outputs)

    def predict_sent(self, sent: Union[str, List[str]]) -> SupertaggerOutput:
        return self.predict_batch([sent])[0]

    def sanity_check(
        self,
        pretokenized_sents: List[List[str]],
        golden_supertags: List[List[str]],
        batch_size = 10
    ) -> None:
        # check the supertagger through re-calculation of the acc
        # can also used for multitagging acc checking
        correct_cnt = 0
        total_cnt = 0
        n_categories = 0

        for i in range(0, len(pretokenized_sents), batch_size):
            if i % 50 == 0:
                print(f'progress: {i} / {len(pretokenized_sents)}')
            sents = pretokenized_sents[i : i + batch_size]
            supertags = golden_supertags[i : i + batch_size]

            predicted = supertagger.predict_batch(sents)

            total_cnt += sum([len(golden) for golden in supertags])
            for j in range(len(supertags)):
                for k in range(len(supertags[j])):
                    n_categories += len(predicted[j][k])
                    if supertags[j][k] in predicted[j][k]:
                        correct_cnt += 1

        print(f'per-word acc of the supertagger = {(correct_cnt / total_cnt) * 100: .3f} (correct if the golden tag is in the top k predicted ones)')
        print(f'averaged number of categories per word = {(n_categories / total_cnt): .2f}')


if __name__ == '__main__':
    # sample use
    import json
    lexical_category2idx_dir = '../data/lexical_category2idx_cutoff.json'
    with open(lexical_category2idx_dir, 'r', encoding = 'utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: category for category, idx in category2idx.items()}

    from transformers import BertTokenizer
    model_path = '../plms/bert-large-uncased'
    supertagger = CCGSupertagger(
        # model = BaseSupertaggingModel(model_path, len(category2idx)),
        model = LSTMSupertaggingModel(model_path, len(category2idx), embed_dim = 1024),
        tokenizer = BertTokenizer.from_pretrained(model_path),
        idx2category = idx2category,
        top_k = 10,
        beta = 0.00005,
    )
    checkpoints_dir = './checkpoints'
    checkpoint_epoch = 19
    supertagger._load_model_checkpoint(checkpoints_dir, checkpoint_epoch)

    data_items, _ = load_auto_file('../data/ccgbank-wsj_00.auto')
    pretokenized_sents = [[token.contents for token in item.tokens] for item in data_items]
    golden_supertags = [[str(token.tag) for token in item.tokens] for item in data_items]
    
    supertagger.sanity_check(pretokenized_sents, golden_supertags)