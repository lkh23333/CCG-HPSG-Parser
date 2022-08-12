import sys, time
import torch, bisect
from typing import *

from ccg_parsing_models import SupertaggingRepresentations, SpanRepresentations

sys.path.append('..')
import ccg_rules
from base import Token, Category, ConstituentNode

CategoryStr = TypeVar('CategoryStr')
RuleName = TypeVar('RuleName')
InstantiatedUnaryRule = Tuple[CategoryStr, CategoryStr, RuleName]
InstantiatedBinaryRule = Tuple[CategoryStr, CategoryStr, List[Tuple[CategoryStr, RuleName]]]


class ChartItem:
    def __init__(self, constituent: ConstituentNode, score: torch.Tensor):
        self.constituent = constituent
        self.score = score

    def __repr__(self):
        return str({'constituent': str(self.constituent), 'score': self.score})

class Chart:
    def __init__(self, l_sent: int):
        self.chart = [None] * l_sent
        for i in range(l_sent):
            self.chart[i] = [None] * (l_sent + 1)
        # [0, ..., l_sent-1] * [0, 1, ..., l_sent]
        self.l = len(self.chart)

class Decoder: # for testing directly, no need to train
    
    def __init__(
        self,
        beam_width: int,
        idx2tag: Dict[int, Any]
    ):
        self.beam_width = beam_width
        self.idx2tag = idx2tag

    def batch_decode(
        self,
        pretokenized_sents: List[List[str]],
        batch_representations: List[SupertaggingRepresentations]
    ) -> List[Chart]:

        charts = list()
        for i in range(len(pretokenized_sents)):
            charts.append(self.decode(pretokenized_sents[i], batch_representations[i]))
        return charts

    def decode(
        self,
        pretokenized_sent: List[str],
        representations: SupertaggingRepresentations
    ) -> Chart:

        chart = Chart(len(pretokenized_sent))
        
        topk_ps, topk_ids = torch.topk(representations, k = min(representations.shape[1], self.beam_width), dim = 1)
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

        # CKY algorithm
        for i in range(chart.l):
            self._apply_unary_rules(chart, tokens, i)
            for k in range(i - 1, -1, -1):
                self._apply_binary_rules(chart, k, i + 1)

        return chart

    def _apply_unary_rules(self, chart: Chart, tokens, i: int): # i is the start position of the token to be processed
        raise NotImplementedError('Please incorporate unary rules!!!')

    def _apply_binary_rules(self, chart: Chart, i: int, k: int):
        raise NotImplementedError('Please incorporate binary rules!!!')


class CCGBaseDecoder(Decoder): # for testing directly, no need to train

    def __init__(
        self,
        beam_width: int,
        idx2tag: Dict[int, str],
        timeout: int = 4
    ):
        super().__init__(
            beam_width = beam_width,
            idx2tag = idx2tag
        )
        self.timeout = timeout

    def _get_instantiated_unary_rules(self, instantiated_unary_rules: List[InstantiatedUnaryRule]):
        self.apply_instantiated_unary_rules = dict()
        for instantiated_unary_rule in instantiated_unary_rules:
            initial_cat = Category.parse(instantiated_unary_rule[0])
            final_cat = Category.parse(instantiated_unary_rule[1])
            if initial_cat not in self.apply_instantiated_unary_rules.keys():
                self.apply_instantiated_unary_rules[initial_cat] = list()
            self.apply_instantiated_unary_rules[initial_cat].append(
                {'result_cat': final_cat, 'used_rule': instantiated_unary_rule[2]}
            )

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

    def decode(
        self,
        pretokenized_sent: List[str],
        representations: SupertaggingRepresentations
    ) -> Chart:

        t0 = time.time()
        chart = Chart(len(pretokenized_sent))

        topk_ps, topk_ids = torch.topk(representations, k = min(representations.shape[1], self.beam_width), dim = 1)
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

        # CKY algorithm
        for i in range(chart.l):
            self._apply_unary_rules(chart, tokens, i)

            if (time.time() - t0) >= self.timeout:
                return None

            for k in range(i - 1, -1, -1):
                # t0 = time.time()
                self._apply_binary_rules(chart, k, i + 1)
                # print(f'applying binary rules - span[{k}][{i+1}]: {time.time()-t0}s')

        return chart

    def _apply_unary_rules(self, chart: Chart, tokens, i: int): # i is the start position of the token to be processed
        if chart.chart[i][i + 1] is not None:
            raise ValueError(f'Cell[{i}][{i+1}] has been taken up, please check!')

        results = [
            ChartItem(
                constituent = ConstituentNode(
                    tag = tokens[i][j]['token'].tag,
                    children = [tokens[i][j]['token']]
                ),
                score = torch.log(tokens[i][j]['p'])
            )
            for j in range(len(tokens[i]))
        ]

        results_ = list()
        for result in results:
            if result.constituent.tag in self.apply_instantiated_unary_rules.keys():
                results_.extend(
                    [
                        ChartItem(
                            constituent = ConstituentNode(
                                tag = tag['result_cat'],
                                children = [result.constituent],
                                used_rule = tag['used_rule']
                            ),
                            score = result.score
                        )
                        for tag in self.apply_instantiated_unary_rules[result.constituent.tag]
                    ]
                )
        results.extend(results_)
        chart.chart[i][i + 1] = results

    def _apply_binary_rules(self, chart: Chart, i: int, k: int):
        # i - start position, k - end position
        if chart.chart[i][k] is not None:
            raise ValueError(f'Cell[{i}][{k}] has been taken up, please check!')
        results = list()

        for j in range(i + 1, k):
            for left in chart.chart[i][j]:
                for right in chart.chart[j][k]:

                    # apply instantiated rules first, otherwise search for binary rules if one of the two constituents contains the X feature, otherwise no results
                    if left.constituent.tag in self.apply_instantiated_binary_rules.keys():
                        if right.constituent.tag in self.apply_instantiated_binary_rules[left.constituent.tag].keys():
                            for result in self.apply_instantiated_binary_rules[left.constituent.tag][right.constituent.tag]:
                                new_item = ChartItem(
                                    constituent = ConstituentNode(
                                        tag = result['result_cat'],
                                        children = [left.constituent, right.constituent],
                                        used_rule = result['used_rule']
                                    ),
                                    score = left.score + right.score
                                )
                                bisect.insort(results, new_item, key = lambda x: x.score)
                    #     else:
                    #         if left.constituent.tag.contain_X_feature or right.constituent.tag.contain_X_feature:
                    #             for binary_rule in ccg_rules.binary_rules:
                    #                 result = binary_rule(left.constituent, right.constituent)
                    #                 if result:
                    #                     new_item = ChartItem(
                    #                         constituent = result,
                    #                         score = left.score + right.score
                    #                     )
                    #                     bisect.insort(results, new_item, key = lambda x: x.score)
                    # else:
                    #     if left.constituent.tag.contain_X_feature or right.constituent.tag.contain_X_feature:
                    #         for binary_rule in ccg_rules.binary_rules:
                    #             result = binary_rule(left.constituent, right.constituent)
                    #             if result:
                    #                 new_item = ChartItem(
                    #                     constituent = result,
                    #                     score = left.score + right.score
                    #                 )
                    #                 bisect.insort(results, new_item, key = lambda x: x.score)
              
        results = results[-1 : len(results)-1-self.beam_width : -1] if len(results)-1-self.beam_width >= 0 else results[-1::-1]
        chart.chart[i][k] = results


class CCGSpanDecoder(CCGBaseDecoder):

    def __init__(
        self,
        beam_width: int,
        idx2tag: Dict[int, str],
        mode: str = None # ['train', 'test']
    ):
        super().__init__(beam_width, idx2tag)
        self.mode = mode
        if self.mode is None:
            raise ValueError('Please specify the mode of CCGSpanDecoder!')

    def batch_decode(
        self,
        pretokenized_sents: List[List[str]],
        batch_representations: Tuple[List[SupertaggingRepresentations], List[SpanRepresentations]]
    ) -> List[Chart]:
        if self.mode is None:
            raise ValueError('Please specify the mode of CCGSpanDecoder!')
        charts = list()
        for i in range(len(pretokenized_sents)):
            charts.append(
                self.decode(
                    pretokenized_sents[i],
                    [batch_representations[0][i], batch_representations[1][i]]
                )
            )
        return charts

    def decode(
        self,
        pretokenized_sent: List[str],
        representations: Tuple[SupertaggingRepresentations, SpanRepresentations]
    ) -> Chart:
        if self.mode is None:
            raise ValueError('Please specify the mode of CCGSpanDecoder!')

        if self.mode == 'train':
            pass
