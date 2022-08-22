import sys
from typing import *
import torch, bisect

sys.path.append('..')
from decoder import *
from ccg_base_decoder import CCGBaseDecoder
from ccg_parsing_models import SpanRepresentations

sys.path.append('../..')
from data_loader import DataItem

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
        batch_representations: List[SpanRepresentations],
        golden_charts: List[Chart] = None
    ) -> List[Chart]:
        if self.mode is None:
            raise ValueError('Please specify the mode of CCGSpanDecoder!')
        charts = list()

        if golden_charts is not None:
            for i in range(len(pretokenized_sents)):
                charts.append(
                    self.decode(
                        pretokenized_sents[i],
                        batch_representations[i],
                        golden_charts[i]
                    )
                )
        else:
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
    filename = '../sample.auto' # need to specify the data item corresponding to the pretokenized_sent in this sample.auto
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