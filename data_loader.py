"""adapted from depccg's implementation @ https://github.com/masashi-y/depccg"""
from typing import List, Tuple, NamedTuple
from base import Token, ConstituentNode, Category


class DataItem(NamedTuple):
    id: str
    tokens: List[Token]
    tree_root: ConstituentNode


class _AutoLineReader(object):
    def __init__(self, line):
        self.line = line
        self.index = 0
        self.word_id = -1
        self.tokens = []
        self.cats = set()

    def next(self):
        end = self.line.find(' ', self.index)
        res = self.line[self.index:end]
        self.index = end + 1
        return res

    def check(self, text, offset=0):
        if self.line[self.index + offset] != text:
            raise RuntimeError(f'failed to parse: {self.line}')

    def peek(self):
        return self.line[self.index]

    def parse(self):
        tree = self.next_node()
        return tree, self.tokens, self.cats

    @property
    def next_node(self):
        if self.line[self.index + 2] == 'L':
            return self.parse_leaf
        elif self.line[self.index + 2] == 'T':
            return self.parse_tree
        else:
            raise RuntimeError(f'failed to parse: {self.line}')

    def parse_leaf(self):
        self.word_id += 1
        self.check('(')
        self.check('<', 1)
        self.check('L', 2)
        self.next()

        cat_str = self.next()
        cat = Category.parse(cat_str)
        self.cats.add(cat_str)

        tag1 = self.next()  # modified POS tag
        tag2 = self.next()  # original POS
        word = self.next().replace('\\', '')

        token = Token(
            contents=word,
            POS=tag1,
            tag=cat
        )
        self.tokens.append(token)

        if word == '-LRB-':
            word = "("
        elif word == '-RRB-':
            word = ')'
        self.next()

        return token

    def parse_tree(self):
        self.check('(')
        self.check('<', 1)
        self.check('T', 2)
        self.next()

        cat_str = self.next()
        cat = Category.parse(cat_str)
        self.cats.add(cat_str)

        head_is_left = self.next() == '0'
        self.next()
        children = []
        while self.peek() != ')':
            children.append(self.next_node())
        self.next()

        if len(children) > 2:
            raise RuntimeError(f'failed to parse: {self.line}')
        else:
            node = ConstituentNode(tag=cat)
            node.children = children
            return node


def load_auto_file(filename: str) -> Tuple[List[DataItem], List[str]]:
    """read traditional AUTO file used for CCGBank
    English CCGbank contains some unwanted categories such as (S\\NP)\\(S\\NP)[conj].
    This reads the treebank while taking care of those categories.

    Args:
        filename (str): file name string

    Yields:
        Iterator[ReaderResult]: iterator object containing parse results
    """

    __fix = {'((S[b]\\NP)/NP)/': '(S[b]\\NP)/NP', 'conj[conj]': 'conj'}

    def _fix(cat):
        if cat in __fix:
            return __fix[cat]
        if cat.endswith(')[conj]') or cat.endswith('][conj]'):
            return cat[:-6]
        return cat

    data_items = list()
    all_cats = set()

    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("ID"):
                id = line
            else:
                line = ' '.join(
                    _fix(token) for token in line.split(' ')
                )
                root, tokens, cats = _AutoLineReader(line).parse()

                data_items.append(DataItem(id, tokens, root))
                all_cats.update(cats)

    return data_items, list(all_cats)


if __name__ == '__main__':
    # sample usage
    filename = "data/ccg-sample.auto"

    items, cats = load_auto_file(filename)

    for item in items:
        print(item.id)

        for token in item.tokens:
            print('{}\t{}\t{}'.format(token.contents, token.POS, token.tag))

        root = item.tree_root

        def _iter(node):
            print(node.tag)
            if isinstance(node, ConstituentNode):
                for child in node.children:
                    _iter(child)
        _iter(root)

    print(cats)
