"""
some functions used to collect information from designated data
"""

import sys
import json
from typing import List, Dict

sys.path.append('..')
from base import ConstituentNode
from data_loader import DataItem, load_auto_file


def collect_unary_rules(data_items: List[DataItem], saving_dir: str) -> None:
    """
    Input:
        data_items - a list of DataItems
        saving_dir - the directory used to store collected unary rules
                     sorted by number they appear in
    """
    def _iter(node, unary_rules):
        if isinstance(node, ConstituentNode):
            if len(node.children) == 1:
                child_cat = str(node.children[0].tag)
                parent_cat = str(node.tag)
                if child_cat not in unary_rules.keys():
                    unary_rules[child_cat] = dict()
                if parent_cat not in unary_rules[child_cat].keys():
                    unary_rules[child_cat][parent_cat] = 0
                unary_rules[child_cat][parent_cat] += 1
            for child in node.children:
                _iter(child, unary_rules)

    unary_rules = dict()
    for item in data_items:
        _iter(item.tree_root, unary_rules)

    for child_cat in unary_rules:
        unary_rules[child_cat] = [
            [parent_cat, cnt]
            for parent_cat, cnt in unary_rules[child_cat].items()
        ]
        unary_rules[child_cat] = sorted(unary_rules[child_cat], key=lambda x: x[1], reverse=True)

    unary_rules_new = list()
    for child_cat in unary_rules.keys():
        for item in unary_rules[child_cat]:
            unary_rules_new.append([child_cat, item[0], item[1]])
    unary_rules_new = sorted(unary_rules_new, key=lambda x: x[2], reverse=True)

    with open(saving_dir, 'w', encoding='utf8') as f:
        json.dump(unary_rules_new, f, indent=2, ensure_ascii=False)


def collect_binary_rules(data_items: List[DataItem], saving_dir: str):
    """
    Input:
        data_items - a list of DataItems
        saving_dir - the directory used to store collected binary rules
                     sorted by number they appear in
    """
    def _iter(node, binary_rules):
        if isinstance(node, ConstituentNode):
            if len(node.children) == 2:
                child_cat_0 = str(node.children[0].tag)
                child_cat_1 = str(node.children[1].tag)
                parent_cat = str(node.tag)
                if child_cat_0 not in binary_rules:
                    binary_rules[child_cat_0] = dict()
                if child_cat_1 not in binary_rules[child_cat_0]:
                    binary_rules[child_cat_0][child_cat_1] = dict()
                if parent_cat not in binary_rules[child_cat_0][child_cat_1]:
                    binary_rules[child_cat_0][child_cat_1][parent_cat] = 0
                binary_rules[child_cat_0][child_cat_1][parent_cat] += 1
            for child in node.children:
                _iter(child, binary_rules)

    binary_rules = dict()
    for data_item in data_items:
        _iter(data_item.tree_root, binary_rules)

    binary_rules_new = []
    for left in binary_rules:
        for right in binary_rules[left]:
            for result in binary_rules[left][right]:
                binary_rules_new.append(
                    [left, right, result, binary_rules[left][right][result]]
                )

    binary_rules_new = sorted(binary_rules_new, key=lambda x: x[3], reverse=True)

    with open(saving_dir, 'w', encoding='utf8') as f:
        json.dump(binary_rules_new, f, indent=2, ensure_ascii=False)


def collect_roots(data_items: List[DataItem]) -> Dict[str, int]:
    """
    Input:
        data_items - a list of DataItems
    Output:
        A dictionary storing all possible root categories
        along with their appearance numbers
    """
    root_set = dict()
    for item in data_items:
        tag = str(item.tree_root.tag)
        if tag not in root_set:
            root_set[tag] = 0
        root_set[tag] += 1
    
    return root_set


if __name__ == '__main__':
    train_data_dir = './ccgbank-wsj_02-21.auto'
    dev_data_dir = './ccgbank-wsj_00.auto'
    test_data_dir = './ccgbank-wsj_23.auto'

    # train_data_items, _ = load_auto_file(train_data_dir)
    # dev_data_items, _ = load_auto_file(dev_data_dir)
    # test_data_items, _ = load_auto_file(test_data_dir)