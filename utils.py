from typing import *
import ccg_rules
from base import ConstituentNode, Token, Category


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


if __name__ == '__main__':
    import json
    with open('./data/seen_binary_rules.json', 'r', encoding = 'utf8') as f:
        seen_binary_pairs = json.load(f)

    instantiated_binary_rules = list()
    for pair in seen_binary_pairs:
        results = [pair[0], pair[1], []]
        for binary_rule in ccg_rules.binary_rules:
            result = binary_rule(
                ConstituentNode(tag = Category.parse(pair[0])),
                ConstituentNode(tag = Category.parse(pair[1]))
            )
            if result:
                results[2].append([str(result.tag), ccg_rules.abbreviated_rule_name[binary_rule.__name__]])
        if results[2]:
            instantiated_binary_rules.append(results)
    
    with open('./data/instantiated_seen_binary_rules.json', 'w', encoding = 'utf8') as f:
        json.dump(instantiated_binary_rules, f, indent=2, ensure_ascii=False)

    # with open('./data/lexical_category2idx_cutoff.json', 'r' , encoding = 'utf8') as f:
    #     category2idx = json.load(f)

    # lexical_categories_cutoff = list(category2idx.keys())
    # cats_markedup = set(collect_cats_from_markedup('./data/markedup'))
    # instantiated_rules, possible_results = apply_binary_rules_to_categories(lexical_categories_cutoff)
    
    # with open('./data/instantiated_binary_rules_cutoff.json', 'w', encoding = 'utf8') as f:
    #     json.dump(instantiated_rules, f, indent = 2, ensure_ascii = False)
    # with open('./data/possible_binary_results_cutoff.json', 'w', encoding = 'utf8') as f:
    #     json.dump(list(possible_results), f, indent = 2, ensure_ascii = False)

    # lexical_categories_cutoff = set(lexical_categories_cutoff)
    # non_lexical_cats_markedup = cats_markedup - lexical_categories_cutoff
    # print(lexical_categories_cutoff-possible_results)
    # print(len(non_lexical_cats_markedup-possible_results))
    # print(len(cats_markedup-possible_results))

    
    
    
