from typing import List, Tuple, Union, Optional, Callable, TypeVar
import re

Node = TypeVar('Node')
FALSE = TypeVar('False')

class Feature:
    def __init__(self, feature_str: str = None):
        self.feature = [feature_str]

    def __repr__(self) -> str:
        return str(self.feature[0])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Feature):
            return repr(self) == repr(other)
        return False

class Tag:
    def __init__(self, tag = None):
        self.tag = tag

class FeatureStructure(Tag):
    pass

cat_split = re.compile(r'([\[\]\(\)/\\])')
punctuations = [',', '.', ';', ':', 'LRB', 'RRB', 'conj']
class Category(Tag):
    @classmethod
    def parse(cls, category_str: str) -> 'Category':
        tokens = cat_split.sub(r' \1 ', category_str)
        buffer = list(reversed([token for token in tokens.split(' ') if token != '']))
        stack = list()

        X_generic_feature = ['X']
        while len(buffer):
            item = buffer.pop()
            if item in punctuations:
                stack.append(Atom(item))
            elif item == '(':
                pass
            elif item == ')':
                y = stack.pop()
                if len(stack) == 0:
                    return y
                slash = stack.pop()
                x = stack.pop()
                stack.append(Functor(x, slash, y))
            elif item in '/\\':
                stack.append(item)
            else:
                if len(buffer) >= 3 and buffer[-1] == '[':
                    buffer.pop()
                    feature = Feature(feature_str = buffer.pop())
                    assert buffer.pop() == ']'

                    if repr(feature) == 'X':
                        feature.feature = X_generic_feature
                        # to assign a shallow copy list containing 'X' so that when one 'X' is assigned a concrete value, the other ones too
                    
                    stack.append(Atom(item, feature))
                else:
                    stack.append(Atom(item))

        if len(stack) == 1:
            return stack[0]
        try:
            x, slash, y = stack
            return Functor(x, slash, y)
        except ValueError:
            raise RuntimeError(f'failed to parse category: {category_str}')
        
class Atom(Category):
    def __init__(self, tag: str, feature: Feature = None):
        super().__init__(tag = tag)
        self.feature = feature

    def __repr__(self) -> str: # to represent the atom structure
        return str({'tag': self.tag, 'feature': self.feature})
    
    def __str__(self) -> str: # to represent the atom category string itself
        feature = repr(self.feature)
        if feature == 'None':
            return self.tag
        return f'{self.tag}[{feature}]'
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Atom):
            return (
                self.tag == other.tag
                and self.feature == other.feature
            )
        return False

    def __hash__(self):
        return hash(str(self))

    def __xor__(self, other: object) -> bool:
        if not isinstance(other, Atom):
            return False
        return self.tag == other.tag

    @property
    def contain_X_feature(self) -> bool:
        return repr(self.feature) == 'X'

class Functor(Category):
    def __init__(self, left: Category, slash: str, right: Category):
        self.left = left
        self.slash = slash
        self.right = right

    def __repr__(self) -> str: # to represent the functor structure
        return str({'left': self.left, 'slash': self.slash, 'right': self.right})

    def __str__(self) -> str: # to represent the functor category string itself
        def _str(cat):
            if isinstance(cat, Functor):
                return f'({cat})'
            return str(cat)
        return _str(self.left) + self.slash + _str(self.right)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Functor):
            return (
                self.left == other.left
                and self.slash == other.slash
                and self.right == other.right
            )
        else:
            return False

    def __hash__(self):
        return hash(str(self))

    def __xor__(self, other: object) -> bool:
        if not isinstance(other, Functor):
            return False
        return (
            self.left ^ other.left
            and self.slash == other.slash
            and self.right ^ other.right
        )

    @property
    def contain_X_feature(self) -> bool:
        return (self.left.contain_X_feature or self.right.contain_X_feature)

class Token:
    def __init__(self, contents: str = None, lemma: str = None, POS: str = None, tag: Tag = None):
        self.contents = contents
        self.lemma = lemma
        self.POS = POS
        self.tag = tag

    def __repr__(self) -> str:
        return str({'contents': self.contents, 'lemma': self.lemma, 'POS': self.POS, 'tag': self.tag})

UnaryRule = Callable[[Node], Union[Node, FALSE]]
BinaryRule = Callable[[Node, Node], Union[Node, FALSE]]

class ConstituentNode:
    def __init__(self, tag: Tag = None, children: List[Union[Token, Node]] = None,
                 used_rule: str = None, head_is_left: bool = None):
        self.tag = tag
        self.children = children
        self.used_rule = used_rule
        self.head_is_left = head_is_left

    def __repr__(self) -> str: # to represent the constituent structure
        return str({'tag': self.tag, 'children': self.children, 'used_rule': self.used_rule})
        
    def __str__(self) -> str: # to represent the constituent category string itself
        return str(self.tag)


if __name__ == '__main__':
    # sample
    token_0 = Token(contents = 'I', lemma = 'I', POS = 'pron', tag = Category.parse('NP'))
    token_1 = Token(contents = 'like', lemma = 'like', POS = 'verb', tag = Category.parse('(S\\NP)/NP'))
    token_2 = Token(contents = 'apples', lemma = 'apple', POS = 'noun', tag = Category.parse('NP'))
    
    constituent_0 = ConstituentNode(tag = token_0.tag, children = [token_0], used_rule = None)
    constituent_1 = ConstituentNode(tag = token_1.tag, children = [token_1], used_rule = None)
    constituent_2 = ConstituentNode(tag = token_2.tag, children = [token_2], used_rule = None)

    constituent_12 = ConstituentNode(tag = 'S\\NP', children = [constituent_1, constituent_2], used_rule = 'FW')
    constituent_012 = ConstituentNode(tag = 'S', children = [constituent_0, constituent_12], used_rule = 'BW')

    
    print(constituent_012)