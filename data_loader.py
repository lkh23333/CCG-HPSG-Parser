from typing import List, Iterator, NamedTuple
from pyparsing import *
from base import Token, ConstituentNode, Category


class DataItem(NamedTuple):
    id: str
    tokens: List[Token]
    tree_root: ConstituentNode


def load_auto_file(filename: str) -> Iterator[DataItem]:
    """
    load CCG data from .auto file sentence by sentence
    returns an iterator
    """

    # pyparsing patterns
    integer = pyparsing_common.integer
    category = Word('SNP()\\/|[]conj.,:;?_BU' + srange('[0-9]') + srange('[a-z]'))
    pos = Word(srange('[A-Z.,]'))
    word = Word(pyparsing_unicode.printables)
    angled = oneOf('< >')
    node_type = oneOf('T L')

    # pattern for non-terminal nodes: (<T CCGcat head dtrs> (...) (...))
    non_leaf = Suppress(angled) + node_type + category + Suppress(integer) + integer + Suppress(angled)
    
    # pattern for terminal nodes: (<L CCGcat mod_POS-tag orig_POS-tag word PredArgCat>)
    leaf = Suppress(angled) + node_type + category + pos + Suppress(pos) + word + \
        SkipTo(category).suppress() + Suppress(category) + Suppress(angled)

    parser = OneOrMore(nestedExpr(content=(non_leaf | leaf)))

    for line in open(filename):
        line = line.strip()

        if len(line) == 0:
            continue
        if line.startswith("ID"):
            id = line
        else:
            tokens = list()

            try:
                parsed = parser.parseString(line).asList()
            except ParseException as e:
                print('pyparsing error: ' + repr(e))     # TODO: use a logger at some point
                # raise
                continue

            if parsed is None or len(parsed) == 0 or len(parsed[0]) == 0:
                continue
            elif parsed[0][0] == 'L':     # if leaf node
                parsed = parsed[0]
                category = parsed[1]
                pos = parsed[2]
                word = parsed[3]

                token = Token(contents=word, POS=pos, tag=Category.parse(category))
                tokens.append(token)
            else:                         # if non-leaf node   
                parsed = parsed[0]

                def _traverse(res):
                    if res is not None and len(res) > 0:
                        if res[0] == 'L':
                            category = res[1]
                            pos = res[2]
                            word = res[3]

                            token = Token(contents=word, POS=pos, tag=Category.parse(category))
                            tokens.append(token)

                            return token
                        else:
                            category = res[1]
                            num_children = res[2]

                            node = ConstituentNode(tag=Category.parse(category))

                            if num_children == 1:
                                children = [_traverse(res[3])]
                            else:
                                children = [_traverse(res[3]), _traverse(res[4])]

                            node.children = children

                            return node

                root = _traverse(parsed)

                yield DataItem(id, tokens, root)


if __name__ == '__main__':
    filename = "data/ccg-sample.auto"

    for item in load_auto_file(filename):
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
