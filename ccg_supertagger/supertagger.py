from typing import List, Dict, Union, Any, TypeVar
import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer

sys.path.append('..')
from data_loader import load_auto_file
from base import Category
from ccg_supertagger.models import BaseSupertaggingModel, LSTMSupertaggingModel
from ccg_supertagger.utils import pre_tokenize_sent


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
        beta: float = 1e-5,  # pruning parameter for supertagging
        device: torch.device = torch.device('cuda:0')
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.idx2category = idx2category
        if idx2category is not None:
            self.category2idx = {idx: cat for idx, cat in idx2category.items()}
        self.top_k = top_k
        self.beta = beta
        self.device = device
        self.softmax = nn.Softmax(dim=2)

    def _prepare_batch_data(self, batch: List[List[str]]) -> Dict[str, Any]:
        """
        Input:
            batch - a list of pretokenized sentences (a list of strings)
        Output:
            wrapped and padded batch data to input into the model
        """ 
        data = list()  # a list containing a list of input_ids for each sentence
        mask = list()  # a list containing the attention mask list for each sentence
        word_piece_tracked = list()  # a list containing the list of word_piece_tracked for each sentence

        for pretokenized_sent in batch:
            word_piece_tracked.append(
                [len(item) for item in self.tokenizer(pretokenized_sent, add_special_tokens=False).input_ids]
            )

            inputs = self.tokenizer(
                pretokenized_sent,
                add_special_tokens=False,
                is_split_into_words=True
            )
            data.append(inputs.input_ids)
            mask.append(inputs.attention_mask)

        max_length = max([len(input_ids) for input_ids in data])
        for i in range(len(data)):
            assert len(data[i]) == len(mask[i])
            data[i] = data[i] + [DATA_MASK_PADDING] * \
                (max_length - len(data[i]))  # padding
            mask[i] = mask[i] + [DATA_MASK_PADDING] * \
                (max_length - len(mask[i]))  # padding

        return {
            'input_ids': torch.LongTensor(data),
            'mask': torch.FloatTensor(mask),
            'word_piece_tracked': word_piece_tracked
        }

    def _convert_model_outputs(self, outputs: List[torch.Tensor]) -> List[SupertaggerOutput]:
        """
        Input:
            outputs - a list of tensors, each of shape (the length of one sentence * C)
        Output:
            a list of category lists,
            each of which corresponds to predicted supertags for a sentence
        """
        if self.idx2category is None:
            raise RuntimeError('Please specify idx2category in the supertagger!!!')

        outputs = self._prune(outputs)

        batch_predicted = list()
        for output in outputs:
            predicted = list()
            for i in range(output.shape[0]):
                topk_ps, topk_ids = torch.topk(output[i], self.top_k)
                ids = topk_ids[topk_ps > 0]
                predicted.append(
                    [
                        str(Category.parse(self.idx2category[idx.item()]))
                        for idx in ids
                    ]
                )
            batch_predicted.append(predicted)
        return batch_predicted

    def _prune(self, outputs) -> torch.Tensor:
        # assign all probabilities less than beta times of the best one to 0
        for output in outputs:
            for i in range(output.shape[0]):
                top_p = torch.topk(output[i], 1)[0]
                binarized = (output[i] > self.beta * top_p)
                output[i] = output[i] * binarized

        return outputs

    def _load_model_checkpoint(self, checkpoint_dir: str):
        checkpoint = torch.load(
            checkpoint_dir,
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_model_outputs_for_batch(self, batch: List[Union[str, List[str]]]) -> List[torch.Tensor]:
        """
        Input:
            batch - a list of sentences (str) or pretokenized sentences (List[str]),
                    better to be pretokenized as the pre_tokenized_sent is not very complete yet
        Output:
            a list of tensors, each of the shape l_sent * C
        """
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()

            for i in range(len(batch)):
                if isinstance(batch[i], str):
                    batch[i] = pre_tokenize_sent(batch[i])

            batch_data = self._prepare_batch_data(batch)
            batch_data['input_ids'] = batch_data['input_ids'].to(self.device)
            batch_data['mask'] = batch_data['mask'].to(self.device)
            outputs = self.model(
                encoded_batch=batch_data['input_ids'],
                mask=batch_data['mask'],
                word_piece_tracked=batch_data['word_piece_tracked']
            )  # B*L*C
            outputs = self.softmax(outputs)

            sents_lengths = [
                len(word_piece_tracked)
                for word_piece_tracked in batch_data['word_piece_tracked']
            ]

            return [
                outputs[i, :sents_lengths[i], :]
                for i in range(len(batch))
            ]

    def get_model_outputs_for_sent(self, sent: Union[str, List[str]]) -> torch.Tensor:
        """
        Input:
            sent - a sentence (str) or a pretokenzied sentence (List[str]),
                   better to be pretokenized as the pre_tokenized_sent is not very complete yet
        Output:
            a tensor of shape (length of this sentence *C)
        """
        return self.get_model_outputs_for_batch([sent])[0]

    def predict_batch(self, batch: List[Union[str, List[str]]]) -> List[SupertaggerOutput]:
        outputs = self.get_model_outputs_for_batch(batch)
        return self._convert_model_outputs(outputs)

    def predict_sent(self, sent: Union[str, List[str]]) -> SupertaggerOutput:
        return self.predict_batch([sent])[0]

    # check the supertagger through re-calculation of the acc
    # can also used for multitagging acc checking
    def sanity_check(
        self,
        pretokenized_sents: List[List[str]],
        golden_supertags: List[List[str]],
        batch_size=10
    ) -> None:
        """
        Input:
            pretokenized_sents - a list of pretokenized sentences (List[str])
            golden_supertags - a list of golden supertag lists,
                               each of which is a list of golden supertag strings
            batch_size - the batch size to be passed into the supertagging model
        """
        correct_cnt = 0
        total_cnt = 0
        n_categories = 0

        for i in range(0, len(pretokenized_sents), batch_size):
            if i % 50 == 0:
                print(f'progress: {i} / {len(pretokenized_sents)}')
            sents = pretokenized_sents[i: i + batch_size]
            supertags = golden_supertags[i: i + batch_size]

            predicted = self.predict_batch(sents)

            total_cnt += sum([len(golden) for golden in supertags])
            for j in range(len(supertags)):
                for k in range(len(supertags[j])):
                    n_categories += len(predicted[j][k])
                    if supertags[j][k] in predicted[j][k]:
                        correct_cnt += 1

        print(
            f'per-word acc of the supertagger = {(correct_cnt / total_cnt) * 100: .3f} (correct if the golden tag is in the top k predicted ones)'
        )
        print(
            f'averaged number of categories per word = {(n_categories / total_cnt): .2f}'
        )


def apply_supertagger(args):
    # sample use
    with open(args.lexical_category2idx_dir, 'r', encoding='utf8') as f:
        category2idx = json.load(f)
    idx2category = {idx: category for category, idx in category2idx.items()}

    if args.model_name == 'fc':
        model = BaseSupertaggingModel(
            model_path=args.model_path,
            n_classes=len(category2idx)
        )
    elif args.model_name == 'lstm':
        model = LSTMSupertaggingModel(
            model_path=args.model_path,
            n_classes=len(category2idx),
            embed_dim=args.embed_dim,
            num_lstm_layers=args.num_lstm_layers
        )
    else:
        raise RuntimeError('Please check the model name!!!')

    supertagger = CCGSupertagger(
        model=model,
        tokenizer=BertTokenizer.from_pretrained(args.model_path),
        idx2category=idx2category,
        top_k=args.top_k,
        beta=args.beta,
        device=args.device
    )
    supertagger._load_model_checkpoint(args.checkpoint_dir)

    if args.mode == 'sanity_check':
        # use dev data for sanity check
        data_items, _ = load_auto_file(args.dev_data_dir)
        pretokenized_sents = [
            [token.contents for token in item.tokens]
            for item in data_items
        ]
        golden_supertags = [
            [str(token.tag) for token in item.tokens]
            for item in data_items
        ]

        supertagger.sanity_check(pretokenized_sents, golden_supertags)
    elif args.mode == 'predict_sent':
        # predict supertags of only one sentence
        # and print the results
        predicted = supertagger.predict_sent(args.sent_to_predict)
        print(predicted)
    elif args.mode == 'predict_batch':
        # predict supertags of many sentences from args.pretokenized_sents_dir
        # and save the results to args.batch_predicted_dir
        with open(args.pretokenized_sents_dir, 'r', encoding='utf8') as f:
            pretokenized_sents = json.load(f)
        predicted = supertagger.predict_batch(pretokenized_sents)
        with open(args.batch_predicted_dir, 'w', encoding='utf8') as f:
            json.dump(predicted, f, indent=2, ensure_ascii=False)
    else:
        raise RuntimeError('Please check the mode of the supertagger!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='apply supertagging')
    parser.add_argument('--sample_data_dir', type=str,
                        default='../data/ccg-sample.auto')
    parser.add_argument('--train_data_dir', type=str,
                        default='../data/ccgbank-wsj_02-21.auto')
    parser.add_argument('--dev_data_dir', type=str,
                        default='../data/ccgbank-wsj_00.auto')
    parser.add_argument('--test_data_dir', type=str,
                        default='../data/ccgbank-wsj_23.auto')
    parser.add_argument('--lexical_category2idx_dir', type=str,
                        default='../data/lexical_category2idx_cutoff.json')
    parser.add_argument('--model_path', type=str,
                        default='../plms/bert-base-uncased')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/epoch_14')

    parser.add_argument('--model_name', type=str,
                        default='lstm', choices=['fc', 'lstm'])
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--num_lstm_layers', type=int, default=1)
    parser.add_argument('--device', type=torch.device,
                        default=torch.device('cuda'))
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--beta', help='the coefficient used to prune predicted categories',
                        type=float, default=0.0005)

    parser.add_argument('--mode', type=str, default='',
                        choices=['predict_sent', 'predict_batch', 'sanity_check'])
    parser.add_argument('--sent_to_predict', type=List[str],
                        default=['No', ',', 'it', 'was', 'n\'t', 'Black', 'Monday', '.'])
    parser.add_argument('--pretokenized_sents_dir', type=str,
                        default='../data/pretokenized_sents.json')
    parser.add_argument('--batch_predicted_dir', type=str,
                        default='./batch_predicted_supertags.json')
    args = parser.parse_args()

    apply_supertagger(args)
