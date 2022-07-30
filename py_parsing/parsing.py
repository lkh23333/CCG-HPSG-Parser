from typing import Tuple, List, Dict, Any, TypeVar
import sys, time
import torch
import bisect

sys.path.append('..')
import base, ccg_rules
from base import Token, Category, ConstituentNode

Node = TypeVar('Node')
LogP = TypeVar('LogP') # LogP - log probability
Y = TypeVar('Y')
Pair = Tuple[Y, Y]
CategoryStr = TypeVar('CategoryStr')
RuleName = TypeVar('RuleName')
InstantiatedBinaryRule = List[Tuple[CategoryStr, CategoryStr, List[Tuple[CategoryStr, RuleName]]]]


class TableItem:
    def __init__(self, constituent: ConstituentNode, log_probability: LogP):
        self.constituent = constituent
        self.log_probability = log_probability

    def __repr__(self):
        return str({'constituent': str(self.constituent), 'log_p': self.log_probability})

class Parser:
    
    def __init__(
        self,
        beam_width: int,
        idx2tag: Dict[int, Any]
    ):
        self.beam_width = beam_width
        self.idx2tag = idx2tag

    def batch_parse(
        self,
        pretokenized_sents: List[List[str]],
        tags_distributions: List[torch.Tensor]
        # B sentences, each of which is a tensor of the shape l_sent*C 
    ):
        tables = list()
        for pretokenized_sent, tags_distribution in zip(pretokenized_sents, tags_distributions):
            tables.append(
                self.cky_parse(pretokenized_sent, tags_distribution)
            )
        return tables

    def cky_parse(
        self,
        pretokenized_sent: List[str],
        tags_distribution: torch.Tensor
        # a tensor of the shape l_sent * C
    ):
        table = [None] * len(pretokenized_sent)
        for i in range(len(pretokenized_sent)):
            table[i] = [None] * (len(pretokenized_sent) + 1)
        # [0, ..., l_sent-1] * [0, 1, ..., l_sent]

        topk_ps, topk_ids = torch.topk(tags_distribution, k = min(tags_distribution.shape[1], self.beam_width), dim = 1)
        ktop_tags = [
            [
                {'tag': self.idx2tag[int(idx.item())], 'p': p}
                for p, idx in zip(topk_p, topk_idx)
            ]
            for topk_p, topk_idx in zip(topk_ps, topk_ids)
        ]

        tokens = [
            [
                {
                    'token': Token(contents = word, tag = tag['tag']),
                    'p': tag['p']
                }
                for tag in ktop
            ]
            for (word, ktop) in zip(pretokenized_sent, ktop_tags)
        ] # l_sent*k tokens with correponding probabilities of the tag

        for i in range(len(table)):
            table = self._apply_unary_rules(table, tokens, i)
            for k in range(i - 1, -1, -1):
                table = self._apply_binary_rules(table, k, i + 1)

        return table

    def _apply_unary_rules(self, table, tokens, i: int): # i is the start position of the token to be processed
        raise NotImplementedError('Please incorporate unary rules!!!')
        return table

    def _apply_binary_rules(self, table, i: int, k: int):
        raise NotImplementedError('Please incorporate binary rules!!!')
        return table

class CCGParser(Parser):

    def __init__(
        self,
        idx2category: Dict[int, str],
        beam_width: int
    ):
        super().__init__(beam_width = beam_width, idx2tag = idx2category)

    def _get_instantiated_unary_rules(self, unary_rule_pairs: List[Pair[CategoryStr]]):
        self.apply_instantiated_unary_rules = dict()
        for unary_rule_pair in unary_rule_pairs:
            initial_cat = Category.parse(unary_rule_pair[0])
            final_cat = Category.parse(unary_rule_pair[1])
            if initial_cat not in self.apply_instantiated_unary_rules.keys():
                self.apply_instantiated_unary_rules[initial_cat] = list()
            self.apply_instantiated_unary_rules[initial_cat].append(final_cat)

    def _get_instantiated_binary_rules(self, instantiated_binary_rules: List[InstantiatedBinaryRule]):
        self.apply_instantiated_binary_rules = dict()
        for instantiated_binary_rule in instantiated_binary_rules:
            left_cat = Category.parse(instantiated_binary_rule[0])
            if left_cat not in self.apply_instantiated_binary_rules.keys():
                self.apply_instantiated_binary_rules[left_cat] = dict()
            right_cat = Category.parse(instantiated_binary_rule[1])
            if right_cat not in self.apply_instantiated_binary_rules[left_cat].keys():
                self.apply_instantiated_binary_rules[left_cat][right_cat] = list()
            for result in instantiated_binary_rule[2]:
                self.apply_instantiated_binary_rules[left_cat][right_cat].append(
                    {'result_cat': Category.parse(result[0]), 'used_rule': result[1]}
                )

    def cky_parse(
        self,
        pretokenized_sent: List[str],
        categories_distribution: List[torch.Tensor] # l_sent * C
    ):
        table = [None] * len(pretokenized_sent)
        for i in range(len(pretokenized_sent)):
            table[i] = [None] * (len(pretokenized_sent) + 1)
        # [0, ..., L-1] * [0, 1, ..., L]

        topk_ps, topk_ids = torch.topk(
            categories_distribution,
            k = min(categories_distribution.shape[1], self.beam_width),
            dim = 1
        )
        ktop_categories = [
            [
                {'category_str': self.idx2tag[int(idx.item())], 'p': p}
                for p, idx in zip(topk_p, topk_idx)
            ]
            for topk_p, topk_idx in zip(topk_ps, topk_ids)
        ]

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
                # t0 = time.time()
                table = self._apply_binary_rules(table, k, i + 1)
                # print(f'applying binary rules - span[{k}][{i+1}]: {time.time()-t0}s')

        return table

    def _apply_unary_rules(self, table, tokens, i: int): # i is the start position of the token to be processed
        if table[i][i + 1] is not None:
            raise ValueError(f'Cell[{i}][{i+1}] has been taken up, please check!')

        results = [
            TableItem(
                constituent = ConstituentNode(
                    tag = tokens[i][j]['token'].tag,
                    children = [tokens[i][j]['token']]
                ),
                log_probability = torch.log(tokens[i][j]['p'])
            )
            for j in range(len(tokens[i]))
        ]

        results_ = list()
        for result in results:
            if result.constituent.tag in self.apply_instantiated_unary_rules.keys():
                results_.extend(
                    [
                        TableItem(
                            constituent = ConstituentNode(
                                tag = tag,
                                children = [result.constituent],
                                used_rule = 'UNARY_RULE'
                            ),
                            log_probability = result.log_probability
                        )
                        for tag in self.apply_instantiated_unary_rules[result.constituent.tag]
                    ] 
                )
        results.extend(results_)
        table[i][i + 1] = results

        return table

    def _apply_binary_rules(self, table, i: int, k: int):
        # i - start position, k - end position
        if table[i][k] is not None:
            raise ValueError(f'Cell[{i}][{k}] has been taken up, please check!')
        results = list()

        for j in range(i + 1, k):
            for left in table[i][j]:
                for right in table[j][k]:

                    # # only apply binary patterns
                    # for binary_rule in ccg_rules.binary_rules:
                    #         result = binary_rule(left.constituent, right.constituent)
                    #         if result:
                    #             results.append(
                    #                 TableItem(
                    #                     constituent = result,
                    #                     log_probability = left.log_probability + right.log_probability
                    #                 )
                    #             )


                    # only apply instantiated binary rules
                    t0 = time.time()
                    if left.constituent.tag in self.apply_instantiated_binary_rules.keys():
                        if right.constituent.tag in self.apply_instantiated_binary_rules[left.constituent.tag].keys():
                            for result in self.apply_instantiated_binary_rules[left.constituent.tag][right.constituent.tag]:
                                new_item = TableItem(
                                    constituent = ConstituentNode(
                                        tag = result['result_cat'],
                                        children = [left.constituent, right.constituent],
                                        used_rule = result['used_rule']
                                    ),
                                    log_probability = left.log_probability + right.log_probability
                                )
                                bisect.insort(results, new_item, key = lambda x: x.log_probability)

                    # # apply instantiated rules first
                    # if left.constituent.tag in self.apply_instantiated_binary_rules.keys():
                    #     if right.constituent.tag in self.apply_instantiated_binary_rules[left.constituent.tag].keys():
                    #         instantiated_cnt += 1
                    #         t1 = time.time()
                    #         results.extend(
                    #             [
                    #                 TableItem(
                    #                     constituent = ConstituentNode(
                    #                         tag = result['result_cat'],
                    #                         children = [left.constituent, right.constituent],
                    #                         used_rule = result['used_rule']
                    #                     ),
                    #                     log_probability = left.log_probability + right.log_probability
                    #                 )
                    #                 for result in self.apply_instantiated_binary_rules[left.constituent.tag][right.constituent.tag]
                    #             ]
                    #         )
                    #         instantiated_t += time.time() - t1
                    #     else:
                    #         flag = False
                    #         for binary_rule in ccg_rules.binary_rules:
                    #             result = binary_rule(left.constituent, right.constituent)
                    #             if result:
                    #                 flag = True
                    #                 results.append(
                    #                     TableItem(
                    #                         constituent = result,
                    #                         log_probability = left.log_probability + right.log_probability
                    #                     )
                    #                 )
                    #         if flag:
                    #             pattern_cnt += 1
                    # else:
                    #     flag = False
                    #     for binary_rule in ccg_rules.binary_rules:
                    #         result = binary_rule(left.constituent, right.constituent)
                    #         if result:
                    #             flag = True
                    #             results.append(
                    #                 TableItem(
                    #                     constituent = result,
                    #                     log_probability = left.log_probability + right.log_probability
                    #                 )
                    #             )
                    #     if flag:
                    #         pattern_cnt += 1
              
        results = results[len(results)-1 : len(results)-1-self.beam_width : -1]
        table[i][k] = results
        return table

if __name__ == '__main__':
    # sample use
    pretokenized_sent = ['I', 'like', 'apples']
    categories_distribution = torch.Tensor([[0.6,0.3,0.1],[0.4,0.5,0.1],[0.45,0.35,0.2]])
    idx2category = {0:'NP', 1:'(S\\NP)/NP', 2:'S'}
    beam_width = 6
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

    parser = CCGParser(
        idx2category = idx2category,
        beam_width = beam_width
    )

    import json
    with open('../../data/instantiated_unary_rules_from_train_data.json', 'r', encoding = 'utf8') as f:
        instantiated_unary_rules = json.load(f)
    with open('../../data/instantiated_binary_rules_from_train_data.json', 'r', encoding = 'utf8') as f:
        instantiated_binary_rules = json.load(f)
    
    parser._get_instantiated_unary_rules(instantiated_unary_rules)
    parser._get_instantiated_binary_rules(instantiated_binary_rules)
    table = parser.cky_parse(
        pretokenized_sent = pretokenized_sent,
        categories_distribution = categories_distribution
    )
    print(table[0][-1])