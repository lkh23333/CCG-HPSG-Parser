import sys, time
from typing import *
import torch, bisect
import numpy as np

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


    
class AStarCellItem(CellItem):

    def __init__(
        self,
        start_end: Tuple[int, int],
        inside_score: float,
        outside_score: float,
        constituent: ConstituentNode
    ):
        self.inside_score = inside_score
        self.outside_score = outside_score
        super().__init__(
            score = inside_score + outside_score,
            start_end = start_end,
            constituent = constituent
        )

    def __repr__(self) -> str:
        return str([str(self.constituent), self.score])

    def __eq__(self, other) -> bool:
        if isinstance(other, AStarCellItem):
            return (
                self.start_end == other.start_end
                and self.constituent == other.constituent
            )
        return False

class Agenda:

    def __init__(
        self,
        tokens: List[List[Dict[str, Any]]]
    ): # l_sent * C

        self.tokens = tokens
        self.minimun_costs = np.array([ktop[0]['score'] for ktop in tokens])
        self.cell_items = list() # an edge is a partial parse for a span
        self.initialize_cell_items()

    def __repr__(self) -> str:
        return str([[str(cell_item.constituent), cell_item.score] for cell_item in self.cell_items])

    @property
    def _is_empty(self):
        return self.cell_items == []

    def initialize_cell_items(self):
        
        # initialize with topk supertags of all words
        for idx, ktop in enumerate(self.tokens):
            for token in ktop:

                outside_score = sum(np.concatenate((self.minimun_costs[:idx], self.minimun_costs[idx + 1:])))
                cell_item = AStarCellItem(
                    start_end = (idx, idx + 1),
                    inside_score = token['score'],
                    outside_score = outside_score,
                    constituent = ConstituentNode(
                        tag = token['token'].tag,
                        children = [token['token']]
                    )
                )
                self.insert(cell_item)

    def insert(self, cell_item: AStarCellItem):
        try: # replace the edge in the agenda with the same span and category but higher cost
            idx = self.cell_items.index(cell_item)
            # print('cell_items', self.cell_items[idx])
            # print('to_insert', cell_item)
            if self.cell_items[idx].score > cell_item.score:
                self.cell_items[idx] = cell_item
        except:
            bisect.insort(self.cell_items, cell_item, key = lambda x: x.score)

    def pop(self) -> AStarCellItem:
        # pop the one with longest dependency length among lowest-cost cell items
        to_pop_idx = 0
        for idx in range(len(self.cell_items)):
            if self.cell_items[idx].score > self.cell_items[to_pop_idx].score:
                break
            if self.cell_items[idx].score == self.cell_items[to_pop_idx].score:
                if self.cell_items[idx].constituent.dep_length < self.cell_items[to_pop_idx].score:
                    to_pop_idx = idx

        return self.cell_items.pop(to_pop_idx)

class AStarChart(Chart):

    def __init__(
        self,
        l_sent: int,
        idx2tag: Dict[int, Any]
    ):
        super().__init__(l_sent, idx2tag)

    @property
    def _is_completed(self):
        return not self.chart[0][self.l]._is_null

    def _print_cell_items(self):
        for i in range(self.l):
            for j in range(i + 1, self.l + 1):
                if not self.chart[i][j]._is_null:
                    print(f'span[{i}][{j}]', [str(cell_item.constituent) for cell_item in self.chart[i][j].cell_items])

    def insert(self, cell_item: AStarCellItem): # used for A* decoding
        start, end = cell_item.start_end
        if self.chart[start][end].cell_items is None:
            self.chart[start][end].cell_items = [cell_item]
        else:
            self.chart[start][end].cell_items.append(cell_item)


def _binarize(ids, length): # to assign 0 to all designated positions
    result = np.ones(length, dtype=np.bool)
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


class CCGAStarDecoder(Decoder): # for testing directly, no need to train

    def __init__(
        self,
        idx2tag: Dict[int, str],
        cat_dict: Dict[str, List[str]],
        top_k: int = 10,
        apply_supertagging_pruning: bool = True,
        beta: float = 0.00001,
        timeout: float = 4.0,
        apply_cat_filtering: bool = True
    ):
        super().__init__(
            top_k = top_k,
            idx2tag = idx2tag
        )
        self.cat_dict = cat_dict # set the possible categories allowed for each word
        self.apply_supertagging_pruning = apply_supertagging_pruning
        self.beta = beta
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

        if self.apply_supertagging_pruning:
            representations = self._prune_supertagging_results(representations)

        results = list()
        for i in range(len(pretokenized_sent)):

            topk_ps, topk_ids = torch.topk(representations[i], self.top_k)
            topk_ids = topk_ids[topk_ps > 0]
            topk_ps = topk_ps[topk_ps > 0]

            sorted_possible_cats_with_scores = [
                [Category.parse(self.idx2tag[int(idx.item())]), -np.log(float(p))]
                for (p, idx) in zip(topk_ps, topk_ids)
            ]
            results.append(sorted_possible_cats_with_scores)

        return results

    def _prune_supertagging_results(self, representations: SupertaggingRepresentations) -> SupertaggingRepresentations:
        
        for i in range(representations.shape[0]):
            top_p = torch.topk(representations[i], 1)[0]
            binarized = (representations[i] > self.beta * top_p)
            representations[i] = representations[i] * binarized

        return representations

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

        # initialization
        agenda = Agenda(tokens)
        chart = AStarChart(
            l_sent = len(pretokenized_sent),
            idx2tag = self.idx2tag
        )
        
        # A* parsing
        while True:
            # print(agenda)
            if (time.time() - t0) >= self.timeout:
                return None

            if agenda._is_empty:
                return None
            
            current = agenda.pop()
            chart.insert(current)
            if chart._is_completed:
                return chart
            
            results = list()
            results.extend(self._apply_unary_rules(current))
            results.extend(self._forward_fundamental(agenda, chart, current))
            results.extend(self._backward_fundamental(agenda, chart, current))

            for new_cell_item in results:
                agenda.insert(new_cell_item)

    def sanity_check(
        self,
        pretokenized_sent: List[str],
        golden_supertags: List[str],
        print_cell_items: bool = True
    ):

        tokens = [
            [{
                'token': Token(contents = token, tag = Category.parse(golden_supertag)),
                'score': 0.0
            }]
            for (token, golden_supertag) in zip(pretokenized_sent, golden_supertags)
        ]

        agenda = Agenda(tokens)
        chart = AStarChart(
            l_sent = len(pretokenized_sent),
            idx2tag = self.idx2tag
        )

        while True:
            if agenda._is_empty:
                return None
            
            current = agenda.pop()
            chart.insert(current)
            if print_cell_items:
                chart._print_cell_items()
            if chart._is_completed:
                return chart

            results = list()
            results.extend(self._apply_unary_rules(current))

            if print_cell_items:
                print('unary', [str(result.constituent) for result in results])

            results.extend(self._forward_fundamental(agenda, chart, current))

            if print_cell_items:
                print('forward', [str(result.constituent) for result in results])

            results.extend(self._backward_fundamental(agenda, chart, current))

            if print_cell_items:
                print('backward', [str(result.constituent) for result in results])

            for new_cell_item in results:
                agenda.insert(new_cell_item)
            
            if print_cell_items:
                print('agenda', [str(cell_item.constituent) for cell_item in agenda.cell_items], '\n')

    def _apply_unary_rules(self, current: AStarCellItem):
        results = list()
        if current.constituent.tag in self.apply_instantiated_unary_rules:
            results.extend(
                [
                    AStarCellItem(
                        start_end = current.start_end,
                        inside_score = current.inside_score,
                        outside_score = current.outside_score,
                        constituent = ConstituentNode(
                            tag = tag['result_cat'],
                            children = [current.constituent],
                            used_rule = tag['used_rule']
                        )
                    )
                    for tag in self.apply_instantiated_unary_rules[current.constituent.tag]
                ]
            )
        return results

    def _forward_fundamental(self, agenda: Agenda, chart: Chart, current: AStarCellItem):
        results = list()
        start, end = current.start_end
        for j in range(end + 1, chart.l + 1):
            if chart.chart[end][j].cell_items is not None:
                for to_combined in chart.chart[end][j].cell_items:
                    if current.constituent.tag in self.apply_instantiated_binary_rules:
                        if to_combined.constituent.tag in self.apply_instantiated_binary_rules[current.constituent.tag]:
                            for result in self.apply_instantiated_binary_rules[current.constituent.tag][to_combined.constituent.tag]:
                                if self._check_constraints(current.constituent, to_combined.constituent, result['used_rule']):
                                    inside_score, outside_score = self._get_score(current, to_combined, agenda.minimun_costs)
                                    new_item = AStarCellItem(
                                        start_end = (start, j),
                                        inside_score = inside_score,
                                        outside_score = outside_score,
                                        constituent = ConstituentNode(
                                            tag = result['result_cat'],
                                            children = [current.constituent, to_combined.constituent],
                                            used_rule = result['used_rule']
                                        )
                                    )
                                    results.append(new_item)
        return results

    def _backward_fundamental(self, agenda: Agenda, chart: Chart, current: AStarCellItem):
        results = list()
        start, end = current.start_end
        for i in range(0, start):
            if chart.chart[i][start].cell_items is not None:
                for to_combined in chart.chart[i][start].cell_items:
                    if to_combined.constituent.tag in self.apply_instantiated_binary_rules:
                        if current.constituent.tag in self.apply_instantiated_binary_rules[to_combined.constituent.tag]:
                            for result in self.apply_instantiated_binary_rules[to_combined.constituent.tag][current.constituent.tag]:
                                if self._check_constraints(to_combined.constituent, current.constituent, result['used_rule']):
                                    inside_score, outside_score = self._get_score(to_combined, current, agenda.minimun_costs)
                                    new_item = AStarCellItem(
                                        start_end = (i, end),
                                        inside_score = inside_score,
                                        outside_score = outside_score,
                                        constituent = ConstituentNode(
                                            tag = result['result_cat'],
                                            children = [to_combined.constituent, current.constituent],
                                            used_rule = result['used_rule']
                                        )
                                    )
                                    results.append(new_item)
        return results

    @staticmethod
    def _get_score(left: AStarCellItem, right: AStarCellItem, minimum_costs: np.array) -> Tuple[float, float]:
        inside_score = left.inside_score + right.inside_score
        outside_score = sum(np.concatenate((minimum_costs[:left.start_end[0]], minimum_costs[right.start_end[1]:])))
        return inside_score, outside_score