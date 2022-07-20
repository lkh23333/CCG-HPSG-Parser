from typing import Tuple, Union, Optional, TypeVar
from base import Category, Atom, Functor


Y = TypeVar('Y')
FALSE = TypeVar('False')
Pair = Tuple[Y, Y]

def unification(x: Category, y: Category, pattern: Pair[str]) -> Union[Pair[Category], FALSE]:
    # return the unified category pair or False if failed
    meta_x, meta_y = pattern
    meta_x = Category.parse(meta_x)
    meta_y = Category.parse(meta_y)

    def _check(cat: Category, meta_cat: Category) -> bool:
        # to check if x and y are respectively consistent with the pattern strings
        if isinstance(meta_cat, Atom):
            return True
        elif isinstance(cat, Atom):
            return False
        else:
            return (
                _check(cat.left, meta_cat.left)
                and (cat.slash == meta_cat.slash)
                and _check(cat.right, meta_cat.right)
            )
    
    if (not _check(x, meta_x)) or (not _check(y, meta_y)):
        # fail in consistency with the pattern strings, 
        # e.g. S\NP is not consistent with the pattern string a/b
        return False

    def _unify_atoms(x: Atom, y: Atom) -> Union[Pair[Atom], FALSE]:
        # return the unified atom pair or False (if failed)
        if x.tag == y.tag:
            if x.feature == y.feature:
                return (x, y)
            elif x.feature == None and y.feature != None:
                return (y, y)
            elif x.feature != None and y.feature == None:
                return (x, x)
            elif x.feature != None and repr(y.feature) == 'X':
                y.feature.feature[0] = x.feature.feature[0]
                return (x, y)
            elif y.feature != None and repr(x.feature) == 'X':
                x.feature.feature[0] = y.feature.feature[0]
                return (x, y)
        return False

    def _unify_functors(x: Functor, y: Functor) -> Union[Pair[Functor], FALSE]:
        # return the unified functor pair or False (if failed)
        if isinstance(x, Atom) and isinstance(y, Atom):
            return _unify_atoms(x, y)
        elif x.slash != y.slash:
            return False
        else:
            left_unified = _unify_functors(x.left, y.left)
            right_unified = _unify_functors(x.right, y.right)
            if left_unified and right_unified:
                return (
                    Functor(left_unified[0], x.slash, right_unified[0]),
                    Functor(left_unified[1], x.slash, right_unified[1])
                )

    def _assign(cat_1: Category, meta_cat_1: Category, cat_2: Category, meta_cat_2: Category) -> Union[Pair[Category], FALSE]:
        # return the unified category pair or False (if failed)
        
        if isinstance(meta_cat_1, Atom) and isinstance(meta_cat_2, Atom):
            if meta_cat_1 == meta_cat_2:
                if type(cat_1) != type(cat_2):
                    return False
                elif isinstance(cat_1, Atom):
                    return _unify_atoms(cat_1, cat_2)
                else:
                    return _unify_functors(cat_1, cat_2)
            return False

        elif isinstance(meta_cat_1, Functor) and isinstance(meta_cat_2, Atom):
            assigned_left = _assign(cat_1.left, meta_cat_1.left, cat_2, meta_cat_2)
            if assigned_left:
                return (
                    Functor(
                        left = assigned_left[0],
                        slash = cat_1.slash,
                        right = cat_1.right
                    ),
                    assigned_left[1]
                )
            else:
                assigned_right = _assign(cat_1.right, meta_cat_1.right, cat_2, meta_cat_2)
                if assigned_right:
                    return (
                        Functor(
                            left = cat_1.left,
                            slash = cat_1.slash,
                            right = assigned_right[0]
                        ),
                        assigned_right[1]
                    )
                else:
                    return False

        elif isinstance(meta_cat_1, Atom) and isinstance(meta_cat_2, Functor):
            assigned_left = _assign(cat_1, meta_cat_1, cat_2.left, meta_cat_2.left)
            if assigned_left:
                return (
                    assigned_left[0],
                    Functor(
                        left = assigned_left[1],
                        slash = cat_2.slash,
                        right = cat_2.right
                    )
                )
            else:
                assigned_right = _assign(cat_1, meta_cat_1, cat_2.right, meta_cat_2.right)
                if assigned_right:
                    return (
                        assigned_right[0],
                        Functor(
                            left = cat_2.left,
                            slash = cat_2.slash,
                            right = assigned_right[1]
                        )
                    )
                else:
                    return False
        
        else:
            is_assigned = False
            
            assigned_left_left = _assign(cat_1.left, meta_cat_1.left, cat_2.left, meta_cat_2.left)
            cat_1, cat_2, is_assigned = (
                Functor(assigned_left_left[0], cat_1.slash, cat_1.right),
                Functor(assigned_left_left[1], cat_2.slash, cat_2.right),
                True
            ) if assigned_left_left else (cat_1, cat_2, is_assigned)
            
            assigned_left_right = _assign(cat_1.left, meta_cat_1.left, cat_2.right, meta_cat_2.right)
            cat_1, cat_2, is_assigned = (
                Functor(assigned_left_right[0], cat_1.slash, cat_1.right),
                Functor(cat_2.left, cat_2.slash, assigned_left_right[1]),
                True
            ) if assigned_left_right else (cat_1, cat_2, is_assigned)

            assigned_right_left = _assign(cat_1.right, meta_cat_1.right, cat_2.left, meta_cat_2.left)
            cat_1, cat_2, is_assigned = (
                Functor(cat_1.left, cat_1.slash, assigned_right_left[0]),
                Functor(assigned_right_left[1], cat_2.slash, cat_2.right),
                True
            ) if assigned_right_left else (cat_1, cat_2, is_assigned)

            assigned_right_right = _assign(cat_1.right, meta_cat_1.right, cat_2.right, meta_cat_2.right)
            cat_1, cat_2, is_assigned = (
                Functor(cat_1.left, cat_1.slash, assigned_right_right[0]),
                Functor(cat_2.left, cat_2.slash, assigned_right_right[1]),
                True
            ) if assigned_right_right else (cat_1, cat_2, is_assigned)
            
            if is_assigned:
                return (cat_1, cat_2)
            else:
                return False

    return _assign(x, meta_x, y, meta_y)
    

if __name__ == '__main__':
    meta_x = 'a/b'
    meta_y = 'b/c'
    x = Category.parse('S/((S[dcl]\\NP)/S)')
    y = Category.parse('((S[X]\\NP[nb])/S[X])/PP')
    c = unification(x, y, (meta_x, meta_y))
    print(str(c[0]))
    print(str(c[1]))