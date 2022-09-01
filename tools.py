from typing import *
import json

import ccg_rules
from base import ConstituentNode, Token, Category


def _preprocess_category(x: str) -> List[str]:
    results = [x]
    if Category.parse(x) == Category.parse('(S\\NP)\\(S\\NP)'):
        results.append('(S[X]\\NP)\\(S[X]\\NP)')
    elif Category.parse(x) == Category.parse('(S\\NP)/(S\\NP)'):
        results.append('(S[X]\\NP)/(S[X]\\NP)')
    elif Category.parse(x) == Category.parse('S/S'):
        results.append('S[X]/S[X]')
    elif Category.parse(x) == Category.parse('S\\S'):
        results.append('S[X]\\S[X]')
    else:
        pass
    return results

def apply_binary_rules_to_categories(categories: List[str]):
    # apply binary rules to all possible category pairs, and collect instantiated rules along with all possible results
    instantiated_rules = list()
    possible_results = set()
    
    progress = 0
    for i in range(len(categories)):
        for j in range(len(categories)):

            progress += 1
            if progress % len(categories) == 0:
                print(f'progress: {progress} / {len(categories) * len(categories)}')

            constituent_1 = ConstituentNode(tag = Category.parse(categories[i]))
            constituent_2 = ConstituentNode(tag = Category.parse(categories[j]))
            results = [categories[i], categories[j], []]
            for binary_rule in ccg_rules.binary_rules:
                result = binary_rule(constituent_1, constituent_2)
                if result:
                    results[2].append(
                        [
                            str(result.tag),
                            ccg_rules.abbreviated_rule_name[binary_rule.__name__]
                        ]
                    )
                    possible_results.add(str(result.tag))
            if results[2]:
                instantiated_rules.append(results)

    return instantiated_rules, possible_results

def collect_binary_rules(data_dir: str, saving_dir: str):

    with open(data_dir, 'r', encoding = 'utf8') as f:
        seen_binary_rules = json.load(f)

    instantiated_binary_rules = list()

    i = 0
    for binary_rule in seen_binary_rules:
        i += 1
        print(f'progress {i} / {len(seen_binary_rules)}')
        results = [binary_rule[0], binary_rule[1], []]
        tag_0 = str(Category.parse(binary_rule[0]))
        tag_1 = str(Category.parse(binary_rule[1]))
        tag_0 = to_X_features[tag_0] if tag_0 in to_X_features.keys() else tag_0
        tag_1 = to_X_features[tag_1] if tag_1 in to_X_features.keys() else tag_1
        for rule in ccg_rules.binary_rules:
            result = rule(
                ConstituentNode(tag = Category.parse(tag_0)),
                ConstituentNode(tag = Category.parse(tag_1))
            )
            if result:
                to_add = [str(result.tag), ccg_rules.abbreviated_rule_name[rule.__name__]]
                if to_add not in results[2]:
                    if Category.parse(to_add[0]) not in [Category.parse(item[0]) for item in results[2]]:
                        results[2].append(to_add)
        if results[2]:
            if results[:2] not in [rule[:2] for rule in instantiated_binary_rules]:
                instantiated_binary_rules.append(results)
            else:
                idx = [rule[:2] for rule in instantiated_binary_rules].index(results[:2])
                instantiated_binary_rules[idx][2].extend(results[2])
        # else:
        #     results[2].append([binary_rule[2], 'UNK'])
        #     if results[:2] not in [rule[:2] for rule in instantiated_binary_rules]:
        #         instantiated_binary_rules.append(results)
        #     else:
        #         idx = [rule[:2] for rule in instantiated_binary_rules].index(results[:2])
        #         instantiated_binary_rules[idx][2].extend(results[2])
    
    print(len(instantiated_binary_rules))
    with open(saving_dir, 'w', encoding = 'utf8') as f:
        json.dump(instantiated_binary_rules, f, indent=2, ensure_ascii=False)

def collect_cats_from_markedup(file_dir: str) -> List[str]:
    cats = list()
    with open(file_dir, 'r', encoding = 'utf8') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] not in ['=', '#', ' ', '\n']:
            cats.append(line.strip())
    return cats

def to_auto(node: ConstituentNode) -> str: # convert one ConstituentNode to an .auto string
    if len(node.children) == 1 and isinstance(node.children[0], Token):
        token = node.children[0]
        cat = token.tag
        word = denormalize(token.contents)
        pos = token.POS
        return f'(<L {cat} {pos} {pos} {word} {cat}>)'
    else:
        cat = node.tag
        children = ' '.join(to_auto(child) for child in node.children)
        num_children = len(node.children)
        head_is_left = 0 if node.head_is_left else 1
        return f'(<T {cat} {head_is_left} {num_children}> {children} )'


# source: https://github.com/masashi-y/depccg
def normalize(word: str) -> str:
    if word == "-LRB-":
        return "("
    elif word == "-RRB-":
        return ")"
    elif word == "-LCB-":
        return "{"
    elif word == "-RCB-":
        return "}"
    elif word == "-LSB-":
        return "["
    elif word == "-RSB-":
        return "]"
    else:
        return word


# source: https://github.com/masashi-y/depccg
def denormalize(word: str) -> str:
    if word == "(":
        return "-LRB-"
    elif word == ")":
        return "-RRB-"
    elif word == "{":
        return "-LCB-"
    elif word == "}":
        return "-RCB-"
    elif word == "[":
        return "-LSB-"
    elif word == "]":
        return "-RSB-"
    word = word.replace(">", "-RAB-")
    word = word.replace("<", "-LAB-")
    return word

to_X_features = {
    "S/(S\\NP)": "(S[X]/(S[X]\\NP))",
    "(S\\NP)\\((S\\NP)/NP)": "((S[X]\\NP)\\((S[X]\\NP)/NP))",
    "(S\\NP)\\((S\\NP)/PP)": "((S[X]\\NP)\\((S[X]\\NP)/PP))",
    "((S\\NP)/NP)\\(((S\\NP)/NP)/NP)": "(((S[X]\\NP)/NP)\\(((S[X]\\NP)/NP)/NP))",
    "((S\\NP)/PP)\\(((S\\NP)/PP)/NP)": "(((S[X]\\NP)/PP)\\(((S[X]\\NP)/PP)/NP))"
}


if __name__ == '__main__':
    collect_binary_rules(data_dir = './data/seen_binary_rules.json', saving_dir = './data/instantiated_seen_binary_rules.json')