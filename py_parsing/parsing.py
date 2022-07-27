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
        idx2category: Dict[int, str],
        beam_width: int,
        unary_rule_pairs: List[Pair[str]]
    ):
        self.idx2category = idx2category
        self.beam_width = beam_width
        self.unary_rule_pairs = unary_rule_pairs

    def cky_parse(
        self,
        pretokenized_sent: List[str],
        categories_distribution: torch.Tensor # L*C
    ):
        table = [None] * len(pretokenized_sent)
        for i in range(len(pretokenized_sent)):
            table[i] = [None] * (len(pretokenized_sent) + 1)
        # [0, ..., L-1] * [0, 1, ..., L]

        topk_ps, topk_ids = torch.topk(categories_distribution, k = self.beam_width, dim = 1)
        ktop_categories = list()
        for i in range(topk_ids.shape[0]):
            topk_p = topk_ps[i]
            topk_id = topk_ids[i]
            topk = list()
            for j in range(topk_p.shape[0]):
                p = topk_p[j]
                idx = topk_id[j]
                topk.append({'category_str': self.idx2category[int(idx.item())], 'p': p})
            ktop_categories.append(topk)
        
        tokens = [
            [
                {
                    'token': Token(contents = word, tag = Category.parse(category['category_str'])),
                    'p': category['p']
                }
                for category in ktop
            ]
            for (word, ktop) in zip(pretokenized_sent, ktop_categories)
        ] # L*k tokens with correponding probabilities of the category

        for i in range(len(table)):
            table = self._apply_unary_rules(table, tokens, i)
            for k in range(i - 1, -1, -1):
                table = self._apply_binary_rules(table, k, i + 1)

        return table

    def _apply_unary_rules(self, table, tokens, i: int): # i is the start position of the token to be processed
        if table[i][i + 1]:
            raise ValueError(f'Cell[{i}][{i+1}] has been taken up, please check!')
        results = [
            TableItem(
                constituent = ConstituentNode(
                    tag = tokens[i][j]['token'].tag,
                    children = [tokens[i][j]['token']]
                ),
                probability = torch.log(tokens[i][j]['p'])
            )
            for j in range(len(tokens[i]))
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
        table[i][i + 1] = results

        return table

    def _apply_binary_rules(self, table, i: int, k: int):
        # i - start position, k - end position
        if table[i][k]:
            raise ValueError(f'Cell[{i}][{k}] has been taken up, please check!')
        results = list()
        for j in range(i + 1, k):
            for left in table[i][j]:
                for right in table[j][k]:
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
        table[i][k] = results

        return table

if __name__ == '__main__':    
    pretokenized_sent = ['I', 'like', 'apples']
    categories_distribution = torch.Tensor([[0.6,0.3,0.1],[0.4,0.5,0.1],[0.45,0.35,0.2]])
    idx2category = {0:'NP', 1:'(S\\NP)/NP', 2:'S'}
    beam_width = 3
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
        idx2category = idx2category,
        beam_width = beam_width,
        unary_rule_pairs = unary_rule_pairs
    )
    table = parser.cky_parse(
        pretokenized_sent = pretokenized_sent,
        categories_distribution = categories_distribution
    )
    print(table[0][-1])