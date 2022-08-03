import time
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
            left = T,
            slash = '/',
            right = Functor(
                T, '\\', x.tag
            )
        ),
        children = [x],
        used_rule = 'FT',
        head_is_left=True
    )


def backward_type_raising(x: ConstituentNode, T: Category) -> Union[ConstituentNode, FALSE]:
    return ConstituentNode(
        tag = Functor(
            left = T,
            slash = '\\',
            right = Functor(
                T, '/', x.tag
            )
        ),
        children = [x],
        used_rule = 'BT',
        head_is_left=True
    )


'''
BINARY RULES
'''


def forward_application(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('a/b', 'b')
    unified_pair = unification(x.tag, y.tag, pattern)

    if _is_modifier(x.tag) or _is_type_raised(x.tag):
        head_is_left = False
    else:
        head_is_left = True

    if unified_pair:
        return ConstituentNode(
            tag = unified_pair[0].left,
            children = [x, y],
            used_rule = 'FA',
            head_is_left=head_is_left
        )
    return False


def backward_application(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('b', 'a\\b')
    unified_pair = unification(x.tag, y.tag, pattern)

    if _is_modifier(y.tag) or _is_type_raised(y.tag):
        head_is_left = True
    else:
        head_is_left = False

    if unified_pair:
        return ConstituentNode(
            tag = unified_pair[1].left,
            children = [x, y],
            used_rule = 'BA',
            head_is_left=head_is_left
        )
    return False


def forward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('a/b', 'b/c')
    unified_pair = unification(x.tag, y.tag, pattern)

    if _is_modifier(x.tag) or _is_type_raised(x.tag):
        head_is_left = False
    else:
        head_is_left = True

    if unified_pair:
        return ConstituentNode(
            tag = Functor(
                left = unified_pair[0].left,
                slash = '/',
                right = unified_pair[1].right
            ),
            children = [x, y],
            used_rule = 'FC',
            head_is_left=head_is_left
        )
    return False


def backward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('b\\c', 'a\\b')
    unified_pair = unification(x.tag, y.tag, pattern)

    if _is_modifier(y.tag) or _is_type_raised(y.tag):
        head_is_left = True
    else:
        head_is_left = False

    if unified_pair:
        return ConstituentNode(
            tag = Functor(
                left = unified_pair[1].left,
                slash = '\\',
                right = unified_pair[0].right
            ),
            children = [x, y],
            used_rule = 'BC',
            head_is_left=head_is_left
        )
    return False


def generalized_forward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern_0 = ('a/b', '(b/c)/$')
    pattern_1 = ('a/b', '(b/c)\\$')
    unified_pair = unification(x.tag, y.tag, pattern_0)

    if _is_modifier(x.tag) or _is_type_raised(x.tag):
        head_is_left = False
    else:
        head_is_left = True

    if unified_pair:
        return ConstituentNode(
            tag = Functor(
                left = Functor(
                    unified_pair[0].left,
                    '/',
                    unified_pair[1].left.right
                ),
                slash = '/',
                right = unified_pair[1].right
            ),
            children = [x, y],
            used_rule = 'GFC',
            head_is_left=head_is_left
        )
    else:
        unified_pair = unification(x.tag, y.tag, pattern_1)
        if unified_pair:
            return ConstituentNode(
                tag = Functor(
                    left = Functor(
                        unified_pair[0].left,
                        '/',
                        unified_pair[1].left.right
                    ),
                    slash = '\\',
                    right = unified_pair[1].right
                ),
                children = [x, y],
                used_rule = 'GFC',
                head_is_left=head_is_left
            )
    return False


def generalized_backward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern_0 = ('(b\\c)/$', 'a\\b')
    pattern_1 = ('(b\\c)\\$', 'a\\b')
    unified_pair = unification(x.tag, y.tag, pattern_0)

    if _is_modifier(y.tag) or _is_type_raised(y.tag):
        head_is_left = True
    else:
        head_is_left = False

    if unified_pair:
        return ConstituentNode(
            tag = Functor(
                left = Functor(
                    unified_pair[1].left,
                    '\\',
                    unified_pair[0].left.right
                ),
                slash = '/',
                right = unified_pair[0].right
            ),
            children = [x, y],
            used_rule = 'GBC',
            head_is_left=head_is_left
        )
    else:
        unified_pair = unification(x.tag, y.tag, pattern_1)
        if unified_pair:
            return ConstituentNode(
                tag = Functor(
                    left = Functor(
                        unified_pair[1].left,
                        '\\',
                        unified_pair[0].left.right
                    ),
                    slash = '\\',
                    right = unified_pair[0].right
                ),
                children = [x, y],
                used_rule = 'GBC',
                head_is_left=head_is_left
            )
    return False


def conjunction(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if (
        not _is_punct(y.tag)
        and not _is_type_raised(y.tag)
        and x.tag in (
            Category.parse(','),
            Category.parse(';'),
            Category.parse('conj')
        )
        and not (y.tag ^ Category.parse('NP\\NP'))
    ):
        return ConstituentNode(
            tag = Functor(y.tag, '\\', y.tag),
            children = [x, y],
            used_rule = 'CONJ',
            head_is_left=False
        )
    return False


def conjunction2(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if x.tag == Category.parse('conj') and y.tag == Category.parse('NP\\NP'):
        return ConstituentNode(
            tag = y.tag,
            children = [x, y],
            used_rule = 'CONJ',
            head_is_left=False
        )
    return False


def remove_punctuation1(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if _is_punct(x.tag):
        return ConstituentNode(
            tag = y.tag,
            children = [x, y],
            used_rule = 'LP',
            head_is_left=False
        )
    return False


def remove_punctuation2(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if _is_punct(y.tag):
        return ConstituentNode(
            tag = x.tag,
            children = [x, y],
            used_rule = 'RP',
            head_is_left=True
        )
    return False


def remove_punctuation_left(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if x.tag in (Category.parse('LQU'), Category.parse('LRB')):
        return ConstituentNode(
            tag = Functor(y.tag, '\\', y.tag),
            children = [x, y],
            used_rule = 'LP',
            head_is_left=False
        )
    return False


def comma_vp_to_adv(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if (
        x.tag == Category.parse(',')
        and y.tag in (
            Category.parse('S[ng]\\NP'),
            Category.parse('S[pss]\\NP')
        )
    ):
        return ConstituentNode(
            tag = Category.parse('(S\\NP)\\(S\\NP)'),
            children = [x, y],
            used_rule = 'LP',
            head_is_left=False
        )
    return False


def parenthetical_direct_speech(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    if x.tag == Category.parse(',') and y.tag == Category.parse('S[dcl]/S[dcl]'):
        return ConstituentNode(
            tag = Category.parse('(S\\NP)/(S\\NP)'),
            children = [x, y],
            used_rule = 'LP',
            head_is_left=False
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
    backward_composition,
    generalized_forward_composition,
    generalized_backward_composition,
    conjunction,
    conjunction2,
    remove_punctuation1,
    remove_punctuation2,
    remove_punctuation_left,
    comma_vp_to_adv,
    parenthetical_direct_speech
]


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
    print(str(constituent_1_))
    
    constituent_12 = generalized_backward_composition(constituent_1, constituent_2)
    print(str(constituent_12))
    constituent_123 = forward_application(constituent_12, constituent_3)
    print(str(constituent_123))