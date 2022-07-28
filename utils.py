from typing import *
from base import ConstituentNode, Token, Category


def to_auto(node: ConstituentNode) -> str:
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
    token_0 = Token(contents='I', lemma='I', POS='pron', tag=Category.parse('NP'))
    token_1 = Token(contents='like', lemma='like', POS='verb', tag=Category.parse('(S\\NP)/NP'))
    token_2 = Token(contents='apples', lemma='apple', POS='noun', tag=Category.parse('NP'))

    constituent_0 = ConstituentNode(tag=token_0.tag, children=[token_0], used_rule=None)
    constituent_1 = ConstituentNode(tag=token_1.tag, children=[token_1], used_rule=None)
    constituent_2 = ConstituentNode(tag=token_2.tag, children=[token_2], used_rule=None)

    constituent_12 = ConstituentNode(tag='S\\NP', children=[constituent_1, constituent_2], used_rule='FW', head_is_left=True)
    constituent_012 = ConstituentNode(tag='S', children=[constituent_0, constituent_12], used_rule='BW', head_is_left=False)

    print(to_auto(constituent_012))
