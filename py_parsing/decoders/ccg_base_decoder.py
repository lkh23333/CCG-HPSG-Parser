import sys, time
from typing import *
import torch, numpy, bisect, math

from decoders.decoder import *

sys.path.append('..')
from ccg_parsing_models import SupertaggingRepresentations

sys.path.append('../..')
import ccg_rules
from base import Token, Category, ConstituentNode

CategoryStr = TypeVar('CategoryStr')
RuleName = TypeVar('RuleName')
InstantiatedUnaryRule = Tuple[CategoryStr, CategoryStr, RuleName]
InstantiatedBinaryRule = Tuple[CategoryStr, CategoryStr, List[Tuple[CategoryStr, RuleName]]]

def _binarize(ids, length): # to assign 0 to all designated positions
    result = numpy.ones(length, dtype=numpy.bool)
    result[ids] = 0
    return result

def apply_category_filters(
    pretokenized_sents: List[List[str]],
    batch_representations: List[SupertaggingRepresentations],
    category2idx: Dict[str, int],
    category_dict: Dict[str, List[str]]
) -> List[SupertaggingRepresentations]:

    category_dict = {
        word: _binarize(
            [category2idx[cat] for cat in cats],
            batch_representations[0].shape[1]
        )
        for word, cats in category_dict.items()
    }

    for tokens, representations in zip(pretokenized_sents, batch_representations):
        for index, token in enumerate(tokens):
            if token in category_dict:
                representations[index, category_dict[token]] = 0

    return batch_representations


class CCGBaseDecoder(Decoder): # for testing directly, no need to train

    def __init__(
        self,
        beam_width: int,
        idx2tag: Dict[int, str],
        cat_dict: Dict[str, List[str]],
        top_k: int = 3,
        timeout: float = 4.0,
        apply_cat_filtering: bool = False
    ):
        super().__init__(
            top_k = top_k,
            idx2tag = idx2tag
        )
        self.beam_width = beam_width
        self.cat_dict = cat_dict # set the possible categories allowed for each word
        self.timeout = timeout
        self.apply_cat_filtering = apply_cat_filtering

    def _get_instantiated_unary_rules(self, instantiated_unary_rules: List[InstantiatedUnaryRule]):
        self.apply_instantiated_unary_rules = dict()
        for instantiated_unary_rule in instantiated_unary_rules:
            initial_cat = Category.parse(instantiated_unary_rule[0])
            final_cat = Category.parse(instantiated_unary_rule[1])
            if initial_cat not in self.apply_instantiated_unary_rules:
                self.apply_instantiated_unary_rules[initial_cat] = list()
            self.apply_instantiated_unary_rules[initial_cat].append(
                {'result_cat': final_cat, 'used_rule': instantiated_unary_rule[2]}
            )

    def _get_instantiated_binary_rules(self, instantiated_binary_rules: List[InstantiatedBinaryRule]):
        self.apply_instantiated_binary_rules = dict()
        for instantiated_binary_rule in instantiated_binary_rules:
            left_cat = Category.parse(instantiated_binary_rule[0])
            if left_cat not in self.apply_instantiated_binary_rules:
                self.apply_instantiated_binary_rules[left_cat] = dict()
            right_cat = Category.parse(instantiated_binary_rule[1])
            if right_cat not in self.apply_instantiated_binary_rules[left_cat]:
                self.apply_instantiated_binary_rules[left_cat][right_cat] = list()
            for result in instantiated_binary_rule[2]:
                self.apply_instantiated_binary_rules[left_cat][right_cat].append(
                    {'result_cat': Category.parse(result[0]), 'used_rule': result[1]}
                )

    def _get_ktop_sorted_scores_for_possible_cats(
        self,
        pretokenized_sent: List[str],
        representations: SupertaggingRepresentations
    ) -> List[List[Tuple[Category, 'log_p']]]:

        results = list()
        for i in range(len(pretokenized_sent)):

            topk_ps, topk_ids = torch.topk(representations[i], self.top_k)
            topk_ids = topk_ids[topk_ps > 0]
            topk_ps = topk_ps[topk_ps > 0]

            sorted_possible_cats_with_scores = [
                [Category.parse(self.idx2tag[idx.item()]), math.log(float(p))]
                for (p, idx) in zip(topk_ps, topk_ids)
            ]
            results.append(sorted_possible_cats_with_scores)

        return results

    def batch_decode(
        self,
        pretokenized_sents: List[List[str]],
        batch_representations: List[SupertaggingRepresentations]
    ) -> List[Chart]:

        if self.apply_cat_filtering:
            batch_representations = apply_category_filters(
                pretokenized_sents = pretokenized_sents,
                batch_representations = batch_representations,
                category2idx = self.tag2idx,
                category_dict = self.cat_dict
            ) # to assign 0 to probabilities of all impossible categories for each word

        charts = list()
        for i in range(len(pretokenized_sents)):
            charts.append(self.decode(pretokenized_sents[i], batch_representations[i]))
        return charts

    def decode(
        self,
        pretokenized_sent: List[str],
        representations: SupertaggingRepresentations
    ) -> Chart:

        t0 = time.time()
        chart = Chart(
            l_sent = len(pretokenized_sent),
            idx2tag = self.idx2tag
        )

        ktop_sorted_cats_with_scores = self._get_ktop_sorted_scores_for_possible_cats(pretokenized_sent, representations)

        tokens = [
            [
                {
                    'token': Token(contents = word, tag = cat_with_score[0]),
                    'score': cat_with_score[1]
                }
                for cat_with_score in ktop
            ]
            for (word, ktop) in zip(pretokenized_sent, ktop_sorted_cats_with_scores)
        ] # L*k tokens with correponding probabilities of the category

        # CKY algorithm
        for i in range(chart.l):
            self._apply_token_ops(chart, tokens, i)

            if (time.time() - t0) >= self.timeout:
                return None

            for k in range(i - 1, -1, -1):
                # t0 = time.time()
                self._apply_span_ops(chart, k, i + 1)
                # print(f'applying binary rules - span[{k}][{i+1}]: {time.time()-t0}s')

        return chart

    def sanity_check(
        self,
        pretokenized_sent: List[str],
        golden_supertags: List[str],
        print_cell_items: bool = False
    ):
        chart = Chart(
            l_sent = len(pretokenized_sent),
            idx2tag = self.idx2tag
        )

        tokens = [
            [{
                'token': Token(contents = token, tag = Category.parse(golden_supertag)),
                'score': 0.0
            }]
            for (token, golden_supertag) in zip(pretokenized_sent, golden_supertags)
        ]

        for i in range(chart.l):
            self._apply_token_ops(chart, tokens, i)
            if chart.chart[i][i+1].cell_items and print_cell_items:
                print(f'span[{i}][{i+1}]', [str(cell_item.constituent.tag) for cell_item in chart.chart[i][i+1].cell_items])
            for k in range(i - 1, -1, -1):
                # t0 = time.time()
                self._apply_span_ops(chart, k, i + 1)
                if chart.chart[k][i+1].cell_items and print_cell_items:
                    print(f'span[{k}][{i+1}]', [str(cell_item.constituent.tag) for cell_item in chart.chart[k][i+1].cell_items])
                # print(f'applying binary rules - span[{k}][{i+1}]: {time.time()-t0}s')

        return chart


    def _apply_token_ops(self, chart: Chart, tokens, i: int): # i is the start position of the token to be processed
        if not chart.chart[i][i + 1]._is_null:
            raise ValueError(f'Cell[{i}][{i+1}] has been taken up, please check!')

        results = [
            CellItem(
                constituent = ConstituentNode(
                    tag = tokens[i][j]['token'].tag,
                    children = [tokens[i][j]['token']]
                ),
                score = tokens[i][j]['score']
            )
            for j in range(len(tokens[i]))
        ]

        results.extend(self._apply_unary_rules(results))
        chart.chart[i][i + 1].cell_items = results

    def _apply_span_ops(self, chart: Chart, i: int, k: int):
        # i - start position, k - end position
        if not chart.chart[i][k]._is_null:
            raise ValueError(f'Cell[{i}][{k}] has been taken up, please check!')
        results = list()

        for j in range(i + 1, k):
            for left in chart.chart[i][j].cell_items:
                for right in chart.chart[j][k].cell_items:
                    for new_item in self._apply_binary_rules(left, right):
                        bisect.insort(results, new_item, key = lambda x: x.score)
            
        results = results[-1 : len(results)-1-self.beam_width : -1] if len(results)-1-self.beam_width >= 0 else results[-1::-1]
        
        results.extend(self._apply_unary_rules(results))
        chart.chart[i][k].cell_items = results

    def _apply_unary_rules(self, cell_items: List[CellItem]) -> List[CellItem]:
        results = list()
        for cell_item in cell_items:
            if cell_item.constituent.tag in self.apply_instantiated_unary_rules:
                results.extend(
                    [
                        CellItem(
                            constituent = ConstituentNode(
                                tag = tag['result_cat'],
                                children = [cell_item.constituent],
                                used_rule = tag['used_rule']
                            ),
                            score = cell_item.score
                        )
                        for tag in self.apply_instantiated_unary_rules[cell_item.constituent.tag]
                    ]
                )
        return results

    def _apply_binary_rules(self, left: CellItem, right: CellItem) -> List[CellItem]:
        results = list()
        # apply instantiated rules first, otherwise search for binary rules if one of the two constituents contains the X feature, otherwise no results
        if left.constituent.tag in self.apply_instantiated_binary_rules:
            if right.constituent.tag in self.apply_instantiated_binary_rules[left.constituent.tag]:
                for result in self.apply_instantiated_binary_rules[left.constituent.tag][right.constituent.tag]:
                    if self._check_constraints(left.constituent, right.constituent, result['used_rule']):
                        new_item = CellItem(
                            constituent = ConstituentNode(
                                tag = result['result_cat'],
                                children = [left.constituent, right.constituent],
                                used_rule = result['used_rule']
                            ),
                            score = left.score + right.score
                        )
                        results.append(new_item)
        #     else:
        #         if left.constituent.tag.contain_X_feature or right.constituent.tag.contain_X_feature:
        #             for binary_rule in ccg_rules.binary_rules:
        #                 result = binary_rule(left.constituent, right.constituent)
        #                 if result:
        #                     new_item = CellItem(
        #                         constituent = result,
        #                         score = left.score + right.score
        #                     )
        #                     results.append(new_item)
        # else:
        #     if left.constituent.tag.contain_X_feature or right.constituent.tag.contain_X_feature:
        #         for binary_rule in ccg_rules.binary_rules:
        #             result = binary_rule(left.constituent, right.constituent)
        #             if result:
        #                 new_item = CellItem(
        #                     constituent = result,
        #                     score = left.score + right.score
        #                 )
        #                 results.append(new_item)

        return results