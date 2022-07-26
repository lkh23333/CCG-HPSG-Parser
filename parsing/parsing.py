from typing import Tuple, List, Dict, TypeVar
import sys
import torch

sys.path.append('..')
import base, ccg_rules
from base import Token, Category, ConstituentNode

Node = TypeVar('Node')
P = TypeVar('P') # P - probability
Y = TypeVar('Y')
Pair = Tuple[Y, Y]


class TableItem:
    def __init__(self, constituent: ConstituentNode, probability: P):
        self.constituent = constituent
        self.probability = probability

    def __repr__(self):
        return str({'constituent': str(self.constituent), 'p': self.probability})

class Parser:

    def __init__(
        self,
        pretokenized_sent: List[str],
        categories_distribution: torch.Tensor, # L*C
        idx2category: Dict[int, str],
        beam_width: int,
        unary_rule_pairs: List[Pair[str]]
    ):
        self.table = [None] * len(pretokenized_sent)
        for i in range(len(pretokenized_sent)):
            self.table[i] = [None] * (len(pretokenized_sent) + 1)
        # [0, ..., L-1] * [0, 1, ..., L]
        self.beam_width = beam_width
        self.unary_rule_pairs = unary_rule_pairs

        topk_ps, topk_ids = torch.topk(categories_distribution, k = self.beam_width, dim = 1)
        ktop_categories = list()
        for i in range(topk_ids.shape[0]):
            topk_p = topk_ps[i]
            topk_id = topk_ids[i]
            topk = list()
            for j in range(topk_p.shape[0]):
                p = topk_p[j]
                idx = topk_id[j]
                topk.append({'category_str': idx2category[int(idx.item())], 'p': p})
            ktop_categories.append(topk)
        
        self.tokens = [
            [
                {
                    'token': Token(contents = word, tag = Category.parse(category['category_str'])),
                    'p': category['p']
                }
                for category in ktop
            ]
            for (word, ktop) in zip(pretokenized_sent, ktop_categories)
        ] # L*k tokens with correponding probabilities of the category
    
    def cky_parse(self):
        for i in range(len(self.table)):
            self._apply_unary_rules(i)
            for k in range(i - 1, -1, -1):
                self._apply_binary_rules(k, i + 1)

    def _apply_unary_rules(self, i: int): # i is the start position of the token to be processed
        if self.table[i][i + 1]:
            raise ValueError(f'Cell[{i}][{i+1}] has been taken up, please check!')
        results = [
            TableItem(
                constituent = ConstituentNode(
                    tag = self.tokens[i][j]['token'].tag,
                    children = [self.tokens[i][j]['token']]
                ),
                probability = torch.log(self.tokens[i][j]['p'])
            )
            for j in range(len(self.tokens[i]))
        ]

        results_ = list()
        for result in results:
            results_.extend(
                [
                    TableItem(
                        constituent = constituent,
                        probability = result.probability
                    )
                    for constituent in ccg_rules.apply_instantiated_unary_rules(result.constituent, self.unary_rule_pairs)
                ]
            )

        results.extend(results_)
        self.table[i][i + 1] = results

    def _apply_binary_rules(self, i: int, k: int):
        # i - start position, k - end position
        if self.table[i][k]:
            raise ValueError(f'Cell[{i}][{k}] has been taken up, please check!')
        results = list()
        for j in range(i + 1, k):
            for left in self.table[i][j]:
                for right in self.table[j][k]:
                    for binary_rule in ccg_rules.binary_rules:
                        result = binary_rule(left.constituent, right.constituent)
                        if result:
                            results.append(
                                TableItem(
                                    constituent = result,
                                    probability = left.probability + right.probability
                                )
                            )
        results = sorted(results, key = lambda x:x.probability, reverse = True)[:self.beam_width]
        self.table[i][k] = results


if __name__ == '__main__':    
    pretokenized_sent = ['I', 'like', 'apples']
    categories_distribution = torch.Tensor([[0.6,0.3,0.1],[0.4,0.5,0.1],[0.45,0.35,0.2]])
    idx2category = {0:'NP', 1:'(S\\NP)/NP', 2:'S'}
    beam_width = 1
    unary_rule_pairs = [
        ['N', 'NP'],
        ['S[pss]\\NP', 'NP\\NP'],
        ['S[ng]\\NP', 'NP\\NP'],
        ['S[adj]\\NP', 'NP\\NP'],
        ['S[to]\\NP', 'NP\\NP'],
        ['S[dcl]/NP', 'NP\\NP'],
        ['S[to]\\NP', '(S/S)'],
        ['S[pss]\\NP', '(S/S)'],
        ['S[ng]\\NP', '(S/S)'],
        ['NP', '(S[X]/(S[X]\\NP))'],
        ['NP', '((S[X]\\NP)\\((S[X]\\NP)/NP))'],
        ['PP', '((S[X]\\NP)\\((S[X]\\NP)/PP))'],
        ['NP', '(((S[X]\\NP)/NP)\\(((S[X]\\NP)/NP)/NP))'],
        ['NP', '(((S[X]\\NP)/PP)\\(((S[X]\\NP)/PP)/NP))'],
        ['(S[ng]\\NP)', 'NP']
    ]
    parser = Parser(
        pretokenized_sent = pretokenized_sent,
        categories_distribution = categories_distribution,
        idx2category = idx2category,
        beam_width = beam_width,
        unary_rule_pairs = unary_rule_pairs
    )

    parser.cky_parse()
    print(parser.table[0][-1])