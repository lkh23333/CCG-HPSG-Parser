# adapted from https://github.com/masashi-y/depccg/blob/master/depccg/cat.py

from typing import TypeVar, List, Union
import re

Node = TypeVar('Node')
FALSE = TypeVar('False')


class Feature:

    def __init__(self, feature_str: str = None):
        self.feature = [feature_str]

    def __repr__(self) -> str:
        return str(self.feature[0]) if self.feature[0] is not None else ''

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Feature):
            if self.is_ignorable:
                return other.is_ignorable  # NP[nb] is equal to NP
            return self.feature[0] == other.feature[0]
        return False

    @property
    def is_ignorable(self):
        # 'nb' is treated as non-existing after combination
        return self.feature[0] is None or self.feature[0] == 'nb'


class Tag:
    def __init__(self, tag=None):
        self.tag = tag


class FeatureStructure(Tag):
    pass


cat_split = re.compile(r'([\[\]\(\)/\\])')
punctuations = [',', '.', ';', ':', 'LRB', 'RRB', 'conj']


class Category(Tag):
    @classmethod
    def parse(cls, category_str: str) -> 'Category':
        tokens = cat_split.sub(r' \1 ', category_str)
        buffer = list(
            reversed([token for token in tokens.split(' ') if token != ''])
        )
        stack = list()

        X_generic_feature = ['X']
        while len(buffer):
            item = buffer.pop()
            if item in punctuations:
                stack.append(Atom(item))
            elif item in '(<':
                stack.append(item)
            elif item in ')>':
                y = stack.pop()
                assert len(stack) > 0
                if (
                    stack[-1] == '(' and item == ')'
                    or stack[-1] == '<' and item == '>'
                ):
                    assert stack.pop() in '(<'
                    stack.append(y)
                else:
                    slash = stack.pop()
                    x = stack.pop()
                    assert stack.pop() in '(<'
                    stack.append(Functor(x, slash, y))
            elif item in '/\\':
                stack.append(item)
            else:
                if len(buffer) >= 3 and buffer[-1] == '[':
                    buffer.pop()
                    feature = Feature(feature_str=buffer.pop())
                    assert buffer.pop() == ']'

                    if repr(feature) == 'X':
                        feature.feature = X_generic_feature
                        # to assign a shallow copy list containing 'X'
                        # so that when one 'X' is assigned a concrete value
                        # the other ones too

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

    def __init__(self, tag: str, feature: Feature = Feature()):
        super().__init__(tag=tag)
        self.feature = feature

    def __repr__(self) -> str:  # to represent the atom structure
        return str({'tag': self.tag, 'feature': self.feature})

    def __str__(self) -> str:  # to represent the atom category string itself
        if repr(self.feature) == '':
            return self.tag
        return f'{self.tag}[{self.feature}]'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Atom):
            return (
                self.tag == other.tag
                and self.feature == other.feature
            )
        return False

    def __hash__(self):
        if self.feature.is_ignorable:
            return hash(str(self.tag))
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

    def __repr__(self) -> str:  # to represent the functor structure
        return str(
            {
                'left': self.left,
                'slash': self.slash,
                'right': self.right
            }
        )

    def __str__(self) -> str:
        # to represent the functor category string itself
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

    def __init__(
        self,
        contents: str = None,
        lemma: str = None,
        POS: str = None,
        tag: Tag = None
    ):
        self.contents = contents
        self.lemma = lemma
        self.POS = POS
        self.tag = tag
        self.start_end = None

    def __repr__(self) -> str:
        return str(
            {
                'contents': self.contents,
                'lemma': self.lemma,
                'POS': self.POS,
                'tag': self.tag
            }
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Token):
            return (
                self.contents == other.contents
                and self.lemma == other.lemma
                and self.POS == other.POS
                and self.tag == other.tag
            )
        return False


class ConstituentNode:

    def __init__(
        self,
        tag: Tag = None,
        children: List[Union[Token, Node]] = None,
        used_rule: str = None,
        head_is_left: bool = None
    ):
        self.tag = tag
        self.children = children
        self.used_rule = used_rule
        self.head_is_left = head_is_left
        self.start_end = None

        # used for calculating head-left dependency length
        # following depccg's rule in English 
        self._get_length()
        self._get_dep_length()

    def __repr__(self) -> str:
        # to represent the constituent structure
        return str(
            {
                'tag': self.tag,
                'children': self.children,
                'used_rule': self.used_rule
            }
        )

    def __str__(self) -> str:
        # to represent the constituent category string itself
        return str(self.tag)

    def __eq__(self, other) -> bool:
        # the two constituents are considered equal once they have the same tags
        # currently only used in A* parsing to rule out equal parses with high costs
        # during A* decoding
        if isinstance(other, ConstituentNode):
            return self.tag == other.tag
        return False

    def _get_length(self) -> None:
        # get the length of the constituent
        # i.e. number of words contained in it
        if len(self.children) == 1:
            if isinstance(self.children[0], Token):
                self.length = 1
            elif isinstance(self.children[0], ConstituentNode):
                self.length = self.children[0].length

        if len(self.children) == 2:
            self.length = self.children[0].length + self.children[1].length

    def _get_dep_length(self) -> None:
        # get the head first dep length of one parse defined by depccg (for English)
        if len(self.children) == 1:
            if isinstance(self.children[0], ConstituentNode):
                self.dep_length = self.children[0].dep_length
            elif isinstance(self.children[0], Token):
                self.dep_length = 0

        if len(self.children) == 2:
            self.dep_length = self.children[0].length + \
                              self.children[0].dep_length + \
                              self.children[1].dep_length


if __name__ == '__main__':
    # sample
    # declare one token, lemma and POS can be specified too
    token_0 = Token(contents='I', tag=Category.parse('NP'))
    token_1 = Token(contents='like', tag=Category.parse('(S\\NP)/NP'))
    token_2 = Token(contents='apples', tag=Category.parse('NP'))

    # a token should be the only child of one ConstituentNode
    # before combination with other tokens
    constituent_0 = ConstituentNode(tag=token_0.tag, children=[token_0], used_rule=None)
    constituent_1 = ConstituentNode(tag=token_1.tag, children=[token_1], used_rule=None)
    constituent_2 = ConstituentNode(tag=token_2.tag, children=[token_2], used_rule=None)

    constituent_12 = ConstituentNode(
        tag='S\\NP',
        children=[constituent_1, constituent_2],
        used_rule='FA'
    )
    constituent_012 = ConstituentNode(
        tag='S',
        children=[constituent_0, constituent_12],
        used_rule='BA'
    )

    print(str(constituent_12))
    print(str(constituent_012))
    print(constituent_012.dep_length)

    a = Category.parse('S/NP')
    b = Category.parse('(S/NP)')
    c = Category.parse('((S/NP))')
    print(str(a))
    print(str(b))
    print(str(c))
