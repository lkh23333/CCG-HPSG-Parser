import time, copy
from typing import Tuple, List, Dict, Callable, Union, Optional, TypeVar
from string import ascii_letters
from base import Token, Atom, Functor, ConstituentNode, Category, UnaryRule, BinaryRule
from ccg_unification import unification


X = TypeVar('X')
FALSE = TypeVar('False')
Pair = Tuple[X, X]


def _is_punct(x: Category) -> bool:
    if isinstance(x, Functor):
        return False
    
    return (
        not x.tag[0] in ascii_letters
        or x.tag in ('LRB', 'RRB', 'LQU', 'RQU')
    )


def _is_type_raised(x: Category) -> bool:
    if isinstance(x, Atom):
        return False
    return (
        isinstance(x.right, Functor)
        and x.right.left == x.left
    )


def _is_modifier(x: Category) -> bool:
    return isinstance(x, Functor) and x.left == x.right


'''
UNARY RULES
'''


def forward_type_raising(x: ConstituentNode, T: Category) -> Union[ConstituentNode, FALSE]:
    return ConstituentNode(
        tag = Functor(
            left = copy.deepcopy(T),
            slash = '/',
            right = Functor(
                copy.deepcopy(T), '\\', copy.deepcopy(x.tag)
            )
        ),
        children = [x],
        used_rule = 'FT',
        head_is_left = True
    )


def backward_type_raising(x: ConstituentNode, T: Category) -> Union[ConstituentNode, FALSE]:
    return ConstituentNode(
        tag = Functor(
            left = copy.deepcopy(T),
            slash = '\\',
            right = Functor(
                copy.deepcopy(T), '/', copy.deepcopy(x.tag)
            )
        ),
        children = [x],
        used_rule = 'BT',
        head_is_left = True
    )


'''
BINARY RULES
'''


def forward_application(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    
    pattern = ('a/b', 'b')
    unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern)

    if unified_pair:
        result = copy.deepcopy(y.tag) if _is_modifier(x.tag) else unified_pair[0].left
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'FA',
            head_is_left = True
        )
    return False


def backward_application(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    
    if str(x.tag) == 'S[dcl]' and str(y.tag) == 'S[em]\\S[em]':
        result = copy.deepcopy(x.tag)
    else:
        pattern = ('b', 'a\\b')
        unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern)
        if unified_pair:
            result = copy.deepcopy(x.tag) if _is_modifier(y.tag) else unified_pair[1].left
        else:
            return False

    return ConstituentNode(
        tag = result,
        children = [x, y],
        used_rule = 'BA',
        head_is_left = True
    )


def forward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:

    pattern = ('a/b', 'b/c')
    unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern)

    if unified_pair:
        result = copy.deepcopy(y.tag) if _is_modifier(x.tag) else Functor(unified_pair[0].left, '/', unified_pair[1].right)
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'FC',
            head_is_left = True
        )
    return False


def backward_crossing_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:

    pattern = ('b/c', 'a\\b')
    unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern)

    if unified_pair:
        if unified_pair[0].left in [Category.parse('N'), Category.parse('NP')]:
            return False
        result = copy.deepcopy(x.tag) if _is_modifier(y.tag) else Functor(unified_pair[1].left, '/', unified_pair[0].right)
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'BX',
            head_is_left = True
        )
    return False


def generalized_forward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:

    pattern_0 = ('a/b', '(b/c)/$')
    pattern_1 = ('a/b', '(b/c)\\$')

    unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern_0)
    if unified_pair:
        result = copy.deepcopy(y.tag) if _is_modifier(x.tag) else Functor(Functor(unified_pair[0].left, '/', unified_pair[1].left.right), '/', unified_pair[1].right)
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'GFC',
            head_is_left = True
        )
    else:
        unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern_1)
        if unified_pair:
            result = copy.deepcopy(y.tag) if _is_modifier(x.tag) else Functor(Functor(unified_pair[0].left, '/', unified_pair[1].left.right), '\\', unified_pair[1].right)
            return ConstituentNode(
                tag = result,
                children = [x, y],
                used_rule = 'GFC',
                head_is_left = True
            )
    return False


def generalized_backward_crossing_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    
    pattern_0 = ('(b/c)/$', 'a\\b')
    pattern_1 = ('(b/c)\\$', 'a\\b')

    unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern_0)
    if unified_pair:
        if unified_pair[1].right in [Category.parse('N'), Category.parse('NP')]:
            return False
        result = copy.deepcopy(x.tag) if _is_modifier(y.tag) else Functor(Functor(unified_pair[1].left, '/', unified_pair[0].left.right), '/', unified_pair[0].right)
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'GBX',
            head_is_left = True
        )
    else:
        unified_pair = unification(copy.deepcopy(x.tag), copy.deepcopy(y.tag), pattern_1)
        if unified_pair:
            if unified_pair[1].right in [Category.parse('N'), Category.parse('NP')]:
                return False
            result = copy.deepcopy(x.tag) if _is_modifier(y.tag) else Functor(Functor(unified_pair[1].left, '/', unified_pair[0].left.right), '\\', unified_pair[0].right)
            return ConstituentNode(
                tag = result,
                children = [x, y],
                used_rule = 'GBX',
                head_is_left = True
            )
    return False


def conjunction(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if (
        not _is_punct(y.tag)
        and not _is_type_raised(y.tag)
        and str(x.tag) in (',', ';', 'conj')
        and not (y.tag ^ Category.parse('NP\\NP'))
    ):
        result = Functor(copy.deepcopy(y.tag), '\\', copy.deepcopy(y.tag))
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'CONJ',
            head_is_left = True
        )
    return False


def conjunction2(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if str(x.tag) == 'conj' and y.tag == Category.parse('NP\\NP'):
        result = copy.deepcopy(y.tag)
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'CONJ',
            head_is_left = True
        )
    return False


def remove_punctuation1(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if _is_punct(x.tag):
        result = copy.deepcopy(y.tag)
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'LP',
            head_is_left = True
        )
    return False


def remove_punctuation2(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if _is_punct(y.tag):
        result = copy.deepcopy(x.tag)
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'RP',
            head_is_left = True
        )
    return False


def remove_punctuation_left(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if str(x.tag) in ('LQU', 'LRB'):
        result = Functor(copy.deepcopy(y.tag), '\\', copy.deepcopy(y.tag))
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'LP',
            head_is_left = True
        )
    return False


def comma_vp_to_adv(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if (
        str(x.tag) == ','
        and str(y.tag) in ('S[ng]\\NP', 'S[pss]\\NP')
    ):
        result = Category.parse('(S\\NP)\\(S\\NP)')
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'LP',
            head_is_left = True
        )
    return False


def parenthetical_direct_speech(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if str(x.tag) == ',' and str(y.tag) == 'S[dcl]/S[dcl]':
        result = Category.parse('(S\\NP)/(S\\NP)')
        return ConstituentNode(
            tag = result,
            children = [x, y],
            used_rule = 'LP',
            head_is_left = True
        )
    return False


unary_rules= [
    forward_type_raising,
    backward_type_raising
]

binary_rules = [
    forward_application,
    backward_application,
    forward_composition,
    backward_crossing_composition,
    generalized_forward_composition,
    generalized_backward_crossing_composition,
    conjunction,
    conjunction2,
    remove_punctuation1,
    remove_punctuation2,
    remove_punctuation_left,
    comma_vp_to_adv,
    parenthetical_direct_speech
]

abbreviated_rule_name = {
    'forward_type_raising': 'FT',
    'backward_type_raising': 'BT',
    'forward_application': 'FA',
    'backward_application': 'BA',
    'forward_composition': 'FC',
    'backward_crossing_composition': 'BX',
    'generalized_forward_composition': 'GFC',
    'generalized_backward_crossing_composition': 'GBX',
    'conjunction': 'CONJ',
    'conjunction2': 'CONJ',
    'remove_punctuation1': 'LP',
    'remove_punctuation2': 'RP',
    'remove_punctuation_left': 'LP',
    'comma_vp_to_adv': 'LP',
    'parenthetical_direct_speech': 'LP'
}


if __name__ == '__main__':
    # sample use
    token_1 = Token(contents = 'like', lemma = 'like', POS = 'verb', tag = Category.parse('(X\\NP[nb])/(PP\\S)'))
    token_2 = Token(contents = 'apples', lemma = 'apple', POS = 'noun', tag = Category.parse('NP\\X'))
    token_3 = Token(contents = 'I', lemma = 'I', POS = 'pron', tag = Category.parse('PP\\S[dcl]'))
    
    
    constituent_1 = ConstituentNode(tag = token_1.tag, children = [token_1], used_rule = None)
    constituent_2 = ConstituentNode(tag = token_2.tag, children = [token_2], used_rule = None)
    constituent_3 = ConstituentNode(tag = token_3.tag, children = [token_3], used_rule = None)
    
    T = Category.parse('S/NP')
    constituent_1_ = backward_type_raising(constituent_1, T)
    # print(str(constituent_1_))

    t1 = Category.parse("(S[dcl]\\NP)/(S[b]\\NP)")
    t2 = Category.parse("S[b]\\NP")

    for binary_rule in binary_rules:
        result = binary_rule(
            ConstituentNode(tag=t1),
            ConstituentNode(tag=t2)
        )
        if result:
            print(str(result.tag), result.used_rule)