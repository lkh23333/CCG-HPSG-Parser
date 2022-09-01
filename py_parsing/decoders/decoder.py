import sys
import torch
from typing import *

sys.path.append('..')
from base import Token, Category, ConstituentNode


class CellItem:

    def __init__(
        self,
        score,
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

    def __repr__(self) -> str:
        return str([repr(cell_item) for cell_item in self.cell_items])

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
        top_k: int,
        idx2tag: Dict[int, Any]
    ):
        self.top_k = top_k
        self.idx2tag = idx2tag
        self.tag2idx = {tag: idx for idx, tag in self.idx2tag.items()}

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
        
        topk_ps, topk_ids = torch.topk(representations, k = min(representations.shape[1], self.top_k), dim = 1)
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

    @staticmethod
    def _check_constraints(left: ConstituentNode, right: ConstituentNode, result_used_rule: str) -> bool:
        
        # Eisner constraints
        if left.used_rule == 'FC' and (result_used_rule == 'FA' or result_used_rule == 'FC'):
            return False
        elif right.used_rule == 'BX' and (result_used_rule == 'BA' or result_used_rule == 'BX'):
            return False
        
        # Julia and Bisk's Constraint 5
        elif left.used_rule == 'FT' and result_used_rule == 'FA':
            return False
        elif right.used_rule == 'BT' and result_used_rule == 'BA':
            return False
        else:
            return True

    def _apply_token_ops(self, chart: Chart, tokens, i: int): # token-level (span[i][i+1]) operations
        # i is the start position of the token to be processed
        raise NotImplementedError('Please incorporate unary rules!!!')

    def _apply_span_ops(self, chart: Chart, i: int, k: int): # span-level (span[i][j]) operations
        raise NotImplementedError('Please incorporate binary rules!!!')