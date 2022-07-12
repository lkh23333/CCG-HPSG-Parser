from typing import List, Tuple, Union, TypeVar
import re

Node = TypeVar('Node')

class Feature:
    def __init__(self, feature: str = None):
        self.feature = feature
    def __repr__(self) -> str:
        return str(self.feature)

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
                    feature = Feature(feature = buffer.pop())
                    assert buffer.pop() == ']'
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
    def __repr__(self) -> str:
        return str({'tag': self.tag, 'feature': self.feature})

class Functor(Category):
    def __init__(self, left: Category, slash: str, right: Category):
        self.left = left
        self.slash = slash
        self.right = right
    def __repr__(self) -> str:
        return str({'left': self.left, 'slash': self.slash, 'right': self.right})

class Token:
    def __init__(self, contents: str = None, lemma: str = None, POS: str = None, tag: Tag = None):
        self.contents = contents
        self.lemma = lemma
        self.POS = POS
        self.tag = tag
    def __repr__(self) -> str:
        return str({'contents': self.contents, 'lemma': self.lemma, 'POS': self.POS, 'tag': self.tag})

class Rule:
    pass

class UnaryRule(Rule):
    pass

class ConstituentNode:
    def __init__(self, tag: Tag = None, children: List[Node] = None, used_rule: Rule = None):
        self.tag = tag
        self.children = children
        self.used_rule = used_rule
    def __repr__(self) -> str:
        return str({'tag': self.tag, 'children': self.children, 'used_rule': self.used_rule})

def apply_unary_rules():
    pass

def apply_binary_rules():
    pass

if __name__ == '__main__':
    token_0 = Token(contents = 'I', lemma = 'I', POS = 'pron', tag = Category.parse('NP'))
    token_1 = Token(contents = 'like', lemma = 'like', POS = 'verb', tag = Category.parse('(S\\NP)/NP'))
    token_2 = Token(contents = 'apples', lemma = 'apple', POS = 'noun', tag = Category.parse('NP'))
    
    constituent_0 = ConstituentNode(tag = token_0.tag, children = token_0, used_rule = None)
    constituent_1 = ConstituentNode(tag = token_1.tag, children = token_1, used_rule = None)
    constituent_2 = ConstituentNode(tag = token_2.tag, children = token_2, used_rule = None)

    constituent_12 = ConstituentNode(tag = 'S\\NP', children = [constituent_1, constituent_2], used_rule = 'FW')
    constituent_012 = ConstituentNode(tag = 'S', children = [constituent_0, constituent_12], used_rule = 'BW')

    
    print(constituent_012)