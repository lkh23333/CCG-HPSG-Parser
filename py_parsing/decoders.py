import sys, time
import torch, bisect
from typing import *

from ccg_parsing_models import SupertaggingRepresentations, SpanRepresentations

sys.path.append('..')
import ccg_rules
from base import Token, Category, ConstituentNode
from data_loader import DataItem

CategoryStr = TypeVar('CategoryStr')
RuleName = TypeVar('RuleName')
InstantiatedUnaryRule = Tuple[CategoryStr, CategoryStr, RuleName]
InstantiatedBinaryRule = Tuple[CategoryStr, CategoryStr, List[Tuple[CategoryStr, RuleName]]]


class CellItem:

    def __init__(
        self,
        score: torch.Tensor,
        constituent: Optional[ConstituentNode] = None, # for base decoding
        start_end: Optional[Tuple[int, int]] = None,
        span_tag_idx: Optional[int] = None,
        span_split_position: Optional[Tuple[int, int, int]] = None, # k, left, right
    ):
        self.constituent = constituent
        self.start_end = start_end
        self.span_tag_idx = span_tag_idx
        self.span_split_position = span_split_position
        self.score = score

    def build_tree(
        self,
        chart: 'Chart',
        idx2tag: Dict[int, Any]
    ) -> ConstituentNode:
    # to show the tree structure, note that unary rules are represented as unary chains, i.e. there are no unary branches

        start, end = self.start_end

        if self.span_split_position is not None:
            k, left, right = self.span_split_position
            return ConstituentNode(
                tag = Category.parse(idx2tag[self.span_tag_idx]),
                children = [
                    chart.chart[start][k].cell_items[left].build_tree(chart, idx2tag),
                    chart.chart[k][end].cell_items[right].build_tree(chart, idx2tag)
                ]
            )
        else:
            return ConstituentNode(
                tag = Category.parse(idx2tag[self.span_tag_idx]),
                children = None
            )
            
class Cell:

    def __init__(
        self,
        span_representation: torch.Tensor = None,
        cell_items: List[CellItem] = None
    ):
        self.span_representation = span_representation
        self.cell_items = cell_items

    @property
    def _is_null(self):
        return self.span_representation == None and self.cell_items == None

class Chart:

    def __init__(
        self,
        l_sent: int,
        idx2tag: Dict[int, Any]
    ):
        self.l = l_sent
        self.chart = [None] * l_sent
        for i in range(l_sent):
            self.chart[i] = [Cell() for _ in range(l_sent + 1)]
        # [0, ..., l_sent-1] * [0, 1, ..., l_sent]
        self.idx2tag = idx2tag
    
    def _show_tree(self, cell_item: CellItem):
        print(str(cell_item.build_tree(self, self.idx2tag)))


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
        batch_representations: List[torch.Tensor]
    ) -> List[Chart]:

        charts = list()
        for i in range(len(pretokenized_sents)):
            charts.append(self.decode(pretokenized_sents[i], batch_representations[i]))
        return charts

    def decode(
        self,
        pretokenized_sent: List[str],
        representations: torch.Tensor
    ) -> Chart:

        chart = Chart(
            l_sent = len(pretokenized_sent),
            idx2tag = self.idx2tag
        )
        
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
            self._apply_token_ops(chart, tokens, i)
            for k in range(i - 1, -1, -1):
                self._apply_span_ops(chart, k, i + 1)

        return chart

    def _apply_token_ops(self, chart: Chart, tokens, i: int): # token-level (span[i][i+1]) operations
        # i is the start position of the token to be processed
        raise NotImplementedError('Please incorporate unary rules!!!')

    def _apply_span_ops(self, chart: Chart, i: int, k: int): # span-level (span[i][j]) operations
        raise NotImplementedError('Please incorporate binary rules!!!')


class CCGBaseDecoder(Decoder): # for testing directly, no need to train

    def __init__(
        self,
        beam_width: int,
        idx2tag: Dict[int, str],
        timeout: float = 4.0
    ):
        super().__init__(
            beam_width = beam_width,
            idx2tag = idx2tag
        )
        self.tag2idx = {tag: idx for idx, tag in self.idx2tag.items()}
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
        chart = Chart(
            l_sent = len(pretokenized_sent),
            idx2tag = self.idx2tag
        )

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
            self._apply_token_ops(chart, tokens, i)

            if (time.time() - t0) >= self.timeout:
                return None

            for k in range(i - 1, -1, -1):
                # t0 = time.time()
                self._apply_span_ops(chart, k, i + 1)
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
                score = torch.log(tokens[i][j]['p'])
            )
            for j in range(len(tokens[i]))
        ]

        results_ = list()
        for result in results:
            if result.constituent.tag in self.apply_instantiated_unary_rules.keys():
                results_.extend(
                    [
                        CellItem(
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
        chart.chart[i][i + 1].cell_items = results

    def _apply_span_ops(self, chart: Chart, i: int, k: int):
        # i - start position, k - end position
        if not chart.chart[i][k]._is_null:
            raise ValueError(f'Cell[{i}][{k}] has been taken up, please check!')
        results = list()

        for j in range(i + 1, k):
            for left in chart.chart[i][j].cell_items:
                for right in chart.chart[j][k].cell_items:

                    # apply instantiated rules first, otherwise search for binary rules if one of the two constituents contains the X feature, otherwise no results
                    if left.constituent.tag in self.apply_instantiated_binary_rules.keys():
                        if right.constituent.tag in self.apply_instantiated_binary_rules[left.constituent.tag].keys():
                            for result in self.apply_instantiated_binary_rules[left.constituent.tag][right.constituent.tag]:
                                new_item = CellItem(
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
                    #                     new_item = CellItem(
                    #                         constituent = result,
                    #                         score = left.score + right.score
                    #                     )
                    #                     bisect.insort(results, new_item, key = lambda x: x.score)
                    # else:
                    #     if left.constituent.tag.contain_X_feature or right.constituent.tag.contain_X_feature:
                    #         for binary_rule in ccg_rules.binary_rules:
                    #             result = binary_rule(left.constituent, right.constituent)
                    #             if result:
                    #                 new_item = CellItem(
                    #                     constituent = result,
                    #                     score = left.score + right.score
                    #                 )
                    #                 bisect.insort(results, new_item, key = lambda x: x.score)
              
        results = results[-1 : len(results)-1-self.beam_width : -1] if len(results)-1-self.beam_width >= 0 else results[-1::-1]
        chart.chart[i][k].cell_items = results


class CCGSpanDecoder(CCGBaseDecoder):

    def __init__(
        self,
        beam_width: int,
        idx2tag: Dict[int, str],
        mode: str = None # ['train', 'test']
    ):
        super().__init__(beam_width, idx2tag)
        self.tag2idx = {tag: idx for idx, tag in self.idx2tag.items()}
        self.mode = mode

    def get_golden_chart(self, data_item: DataItem, representations: SpanRepresentations) -> Chart:
        
        tokens = data_item.tokens
        golden_tree = data_item.tree_root
        golden_chart = Chart(l_sent = len(tokens), idx2tag = self.idx2tag)
        for i in range(golden_chart.l):
            for j in range(i + 1, golden_chart.l + 1):
                golden_chart.chart[i][j].span_representation = representations[i, j]
        
        def _categories_to_strings(node: Union[Token, ConstituentNode]):
            node.tag = str(node.tag)
            if isinstance(node, ConstituentNode):
                for child in node.children:
                    _categories_to_strings(child)

        def _get_unary_chains(node: Union[Token, ConstituentNode]):
            # change all unary rules to unary chains
            if isinstance(node, ConstituentNode):
                if len(node.children) == 1:
                    node.tag = node.tag + '-' + node.children[0].tag
                    node.children = node.children[0].children
                for child in node.children:
                    _get_unary_chains(child)

        def _get_start_end(node: Union[Token, ConstituentNode], cnt: List[int]):
            if isinstance(node, Token):
                node.start_end = (cnt[0], cnt[0] + 1)
                cnt[0] += 1
            elif isinstance(node, ConstituentNode):
                _get_start_end(node.children[0], cnt = cnt)
                _get_start_end(node.children[1], cnt = cnt)
                node.start_end = (node.children[0].start_end[0], node.children[1].start_end[1])
            else:
                raise TypeError('Please check the type of the node!!!')

        def _get_golden_scores(
            node: Union[Token, ConstituentNode],
            golden_chart: Chart,
            tag2idx: Dict[str, int],
            representations: SpanRepresentations
        ):
            if isinstance(node, Token):
                span_tag_idx = tag2idx[str(node.tag)]
                golden_chart.chart[node.start_end[0]][node.start_end[1]].cell_items = [
                    CellItem(
                        start_end = node.start_end,
                        span_tag_idx = span_tag_idx,
                        score = representations[node.start_end[0], node.start_end[1]][span_tag_idx]
                    )
                ]
            elif isinstance(node, ConstituentNode):
                _get_golden_scores(node.children[0], golden_chart, tag2idx, representations)
                _get_golden_scores(node.children[1], golden_chart, tag2idx, representations)

                span_tag_idx = tag2idx[str(node.tag)]
                golden_chart.chart[node.start_end[0]][node.start_end[1]].cell_items = [
                    CellItem(
                        start_end = node.start_end,
                        span_tag_idx = span_tag_idx,
                        span_split_position = (node.children[0].start_end[1], 0, 0),
                        score = sum([
                            representations[node.start_end[0], node.start_end[1]][span_tag_idx],
                            golden_chart.chart[node.children[0].start_end[0]][node.children[0].start_end[1]].cell_items[0].score,
                            golden_chart.chart[node.children[1].start_end[0]][node.children[1].start_end[1]].cell_items[0].score
                        ])
                    )
                ]
            else:
                raise TypeError('Please check the type of the node!!!')

        _categories_to_strings(golden_tree)
        _get_unary_chains(golden_tree)
        _get_start_end(golden_tree, cnt = [0])
        _get_golden_scores(golden_tree, golden_chart, self.tag2idx, representations)

        return golden_chart


    def batch_decode(
        self,
        pretokenized_sents: List[List[str]],
        batch_representations: List[SpanRepresentations]
    ) -> List[Chart]:
        if self.mode is None:
            raise ValueError('Please specify the mode of CCGSpanDecoder!')
        charts = list()
        for i in range(len(pretokenized_sents)):
            charts.append(
                self.decode(
                    pretokenized_sents[i],
                    batch_representations[i]
                )
            )
        return charts

    def decode(
        self,
        pretokenized_sent: List[str],
        representations: SpanRepresentations,
        golden_chart: Chart = None
    ) -> Chart:
        if self.mode is None:
            raise ValueError('Please specify the mode of CCGSpanDecoder!')

        # initialize the chart
        chart = Chart(
            l_sent = len(pretokenized_sent),
            idx2tag = self.idx2tag
        )
        for i in range(chart.l):
            for j in range(i + 1, chart.l + 1):
                chart.chart[i][j].span_representation = representations[i, j]

        # CKY algorithm
        for i in range(chart.l):
            self._apply_token_ops(chart, i)
            for k in range(i - 1, -1, -1):
                self._apply_span_ops(chart, k, i + 1)

    def _apply_token_ops(self, chart, i):

        if self.mode == 'train':
            span_tag_idx = torch.argmax(chart.chart[i][i + 1].span_representation)
            cell_item = CellItem(
                start_end = (i, i + 1),
                span_tag_idx = span_tag_idx,
                score = chart.chart[i][i + 1].span_representation[span_tag_idx] + (span_tag_idx == golden_chart[i][i + 1].span_tag_idx)
            )
            chart.chart[i][i + 1].cell_items = [cell_item]

        elif self.mode == 'test':
            topk_values, topk_ids = torch.topk(chart[i][i + 1].span_representation, k = self.beam_width)
            cell_items = []
            for score, span_tag_idx in zip(topk_values, topk_ids):
                cell_items.append(
                    CellItem(
                        start_end = (i, i + 1),
                        span_tag_idx = int(span_tag_idx),
                        score = score
                    )
                )
            chart.chart[i][i + 1].cell_items = cell_items

        else:
            raise ValueError('Please specify the mode of the decoder!!!')

    def _apply_span_ops(self, chart, i, j):
        
        if self.mode == 'train':
            
            span_tag_idx = torch.argmax(chart.chart[i][j].span_representation)
            optional_span_split_positions = []
            optional_scores = []
            for k in range(i + 1, j):
                for left in range(len(chart.chart[i][k].cell_items)):
                    for right in range(len(chart.chart[k][j].cell_items)):
                        optional_span_split_positions.append((k, left, right))
                        optional_scores.append(chart.chart[i][k].cell_items[left].score + chart.chart[k][j].cell_items[right].score)
            max_score_idx = int(torch.argmax(torch.Tensor(optional_scores)).item())
            cell_item = CellItem(
                start_end= (i, j),
                span_tag_idx = span_tag_idx,
                span_split_position = optional_span_split_positions[max_score_idx],
                score = chart.chart[i][j].span_representation[span_tag_idx] + optional_scores[max_score_idx] + (span_tag_idx == golden_chart[i][j].span_tag_idx)
            )

            chart.chart[i][j].cell_items = [cell_item]
        
        elif self.mode == 'test':
            
            topk_span_scores, topk_span_tag_ids = torch.topk(chart[i][j].span_representation, k = self.beam_width)
            optional_span_split_positions = []
            optional_span_split_scores = []
            for k in range(i + 1, j):
                for left in range(len(chart.chart[i][k].cell_items)):
                    for right in range(len(chart.chart[k][j].cell_items)):
                        span_split_score = chart.chart[i][k].cell_items[left].score + chart.chart[k][j].cell_items[right].score
                        position = bisect.bisect(optional_span_split_scores, span_split_score)
                        optional_span_split_positions.insert(position, (k, left, right))
                        optional_span_split_scores.insert(position, span_split_score)

            topk_span_scores = topk_span_scores[:self.beam_width]
            topk_span_tag_ids = topk_span_tag_ids[:self.beam_width]
            optional_span_split_positions = optional_span_split_positions[:self.beam_width]
            optional_span_split_scores = optional_span_split_scores[:self.beam_width]

            optional_ids = []
            optional_scores = []
            for span_score_idx in range(len(topk_span_scores)):
                for span_split_idx in range(len(optional_span_split_scores)):
                    score = topk_span_scores[span_score_idx] + optional_span_split_scores[span_split_idx]
                    position = bisect.bisect(optional_scores, score)
                    optional_ids.insert(position, (span_score_idx, span_split_idx))
                    optional_scores.insert(position, score)

            cell_items = list()
            for idx in range(len(optional_scores)):
                cell_items.append(
                    CellItem(
                        start_end = (i, j),
                        span_tag_idx = topk_span_tag_ids[optional_ids[idx][0]],
                        span_split_position = optional_span_split_positions[optional_ids[idx][1]],
                        score = optional_scores[idx]
                    )
                )
            
            chart.chart[i][j].cell_items = cell_items

        else:
            raise ValueError('Please specify the value of the mode!!!')


if __name__ == '__main__':
    # test
    from data_loader import load_auto_file
    filename = './sample.auto'
    items, _ = load_auto_file(filename)
    data_item = items[0]

    pretokenized_sent = ['Champagne', 'and', 'dessert', 'followed', '.']
    idx2tag = {
        0: 'NP',
        1: '.',
        2: 'S[dcl]\\NP',
        3: 'S[dcl]',
        4: 'N[conj]',
        5: 'N',
        6: 'conj',
        7: 'NP-N'
    }
    tag2idx = {v: k for k, v in idx2tag.items()}
    decoder = CCGSpanDecoder(beam_width = 3, idx2tag = idx2tag)
    representations = torch.rand(5, 6, len(idx2tag))
    softmax = torch.nn.Softmax(dim = 2)
    representations = softmax(representations)

    golden_chart = decoder.get_golden_chart(data_item, representations)
    print(golden_chart.chart[0][5].cell_items[0].score, representations[0, 5][golden_chart.chart[0][5].cell_items[0].span_tag_idx])
    print(golden_chart.chart[0][4].cell_items[0].score, representations[0, 4][golden_chart.chart[0][4].cell_items[0].span_tag_idx])
    print(golden_chart.chart[4][5].cell_items[0].score, representations[4, 5][golden_chart.chart[4][5].cell_items[0].span_tag_idx])
    print(golden_chart.chart[3][4].cell_items[0].score, representations[3, 4][golden_chart.chart[3][4].cell_items[0].span_tag_idx])
    print(golden_chart.chart[0][3].cell_items[0].score, representations[0, 3][golden_chart.chart[0][3].cell_items[0].span_tag_idx])
    print(golden_chart.chart[0][1].cell_items[0].score, representations[0, 1][golden_chart.chart[0][1].cell_items[0].span_tag_idx])
    print(golden_chart.chart[1][3].cell_items[0].score, representations[1, 3][golden_chart.chart[1][3].cell_items[0].span_tag_idx])
    print(golden_chart.chart[1][2].cell_items[0].score, representations[1, 2][golden_chart.chart[1][2].cell_items[0].span_tag_idx])
    print(golden_chart.chart[2][3].cell_items[0].score, representations[2, 3][golden_chart.chart[2][3].cell_items[0].span_tag_idx])