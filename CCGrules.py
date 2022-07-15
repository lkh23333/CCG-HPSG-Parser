from typing import Tuple, List, Dict, Callable, Union, Optional, TypeVar
from base import Token, Atom, Functor, ConstituentNode, Category, UnaryRule, BinaryRule
from CCGUnification import unification


X = TypeVar('X')
FALSE = TypeVar('False')
Pair = Tuple[X, X]


def forward_type_raising(x: ConstituentNode, T: Category) -> Union[ConstituentNode, FALSE]:
    return ConstituentNode(
        tag = Functor(
            left = T,
            slash = '/',
            right = Functor(
                T, '\\', x
            )
        ),
        children = [x],
        used_rule = 'FT'
    )

def backward_type_raising(x: ConstituentNode, T: Category) -> Union[ConstituentNode, FALSE]:
    return ConstituentNode(
        tag = Functor(
            left = T,
            slash = '\\',
            right = Functor(
                T, '/', x
            )
        ),
        children = [x],
        used_rule = 'BT'
    )

def apply_instantiated_unary_rules(x: ConstituentNode, unary_rule_pairs: List[Pair[str]]) -> List[ConstituentNode]:
    results = list()

    for unary_rule_pair in unary_rule_pairs:
        input_cat, output_cat = unary_rule_pair
        input_cat = Category.parse(input_cat)
        output_cat = Category.parse(output_cat)
        if x.tag == input_cat:
            results.append(
                ConstituentNode(
                    tag = output_cat, children = [x], used_rule = 'UNARY_INSTANCE' ### need to specify which rule is used later
                )
            )
    return results


def forward_application(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('a/b', 'b')
    unified_pair = unification(x.tag, y.tag, pattern)
    if unified_pair:
        return ConstituentNode(
            tag = unified_pair[0].left,
            children = [x, y],
            used_rule = 'FA'
        )
    return False

def backward_application(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('b', 'a\\b')
    unified_pair = unification(x.tag, y.tag, pattern)
    if unified_pair:
        return ConstituentNode(
            tag = unified_pair[1].left,
            children = [x, y],
            used_rule = 'BA'
        )
    return False

def forward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('a/b', 'b/c')
    unified_pair = unification(x.tag, y.tag, pattern)
    if unified_pair:
        return ConstituentNode(
            tag = Functor(
                left = unified_pair[0].left,
                slash = '/',
                right = unified_pair[1].right
            ),
            children = [x, y],
            used_rule = 'FC'
        )
    return False

def backward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern = ('b\\c', 'a\\b')
    unified_pair = unification(x.tag, y.tag, pattern)
    if unified_pair:
        return ConstituentNode(
            tag = Functor(
                left = unified_pair[1].left,
                slash = '\\',
                right = unified_pair[0].right
            ),
            children = [x, y],
            used_rule = 'BC'
        )
    return False

def generalized_forward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern_0 = ('a/b', '(b/c)/$')
    pattern_1 = ('a/b', '(b/c)\\$')
    unified_pair = unification(x.tag, y.tag, pattern_0)
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
            used_rule = 'GFC'
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
                used_rule = 'GFC'
            )
    return False

def generalized_backward_composition(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pattern_0 = ('(b\\c)/$', 'a\\b')
    pattern_1 = ('(b\\c)\\$', 'a\\b')
    unified_pair = unification(x.tag, y.tag, pattern_0)
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
            used_rule = 'GBC'
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
                used_rule = 'GBC'
            )
    return False

def conjunction(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pass

def conjunction2(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pass

def remove_punctuation1(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pass

def remove_punctuation2(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pass

def remove_punctuation_left(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pass

def comma_vp_to_adv(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pass

def parenthetical_direct_speech(x: ConstituentNode, y: ConstituentNode) -> Union[ConstituentNode, FALSE]:
    pass


binary_rules: Dict[str, BinaryRule] = {
    'FA': forward_application,
    'BA': backward_application,
    'FC': forward_composition,
    'BC': backward_composition,
    'GFC': generalized_forward_composition,
    'GBC': generalized_backward_composition,
    'CONJ1': conjunction,
    'CONJ2': conjunction2,
    'LP1': remove_punctuation1,
    'RP': remove_punctuation2,
    'LP2': remove_punctuation_left,
    'LP3': comma_vp_to_adv,
    'LP4': parenthetical_direct_speech,
}

def apply_unary_rules():
    pass

def apply_binary_rules():
    pass

if __name__ == '__main__':
    
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