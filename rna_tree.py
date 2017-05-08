from general_helpers import enum, bracket_pairs, pdb_residues
from collections import defaultdict
from itertools import chain
import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Dict
from Bio.PDB.Residue import Residue


_DataSource = enum('nts', 'dbn', 'lon')

_CACHE_DICT = {}


def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class MemoDict(object):
        def __init__(self, g):
            self.g = g

        def __call__(self, arg):
            try:
                return _CACHE_DICT[arg]
            except Exception as _:
                ret = _CACHE_DICT[arg] = self.g(arg)
                return ret

    return MemoDict(f)


@memoize
def _coords_from_residue(residue: Residue) -> Tuple[float, float, float]:
    return residue['P'].coord if 'P' in residue else residue.child_list[0].coord


def _coords_from_residues(residues: List[Residue]) -> np.ndarray:
    return np.array(list(_coords_from_residue(residue) for residue in residues))


def _ndarray_from_residues(residues: List[Residue]) -> List:
    return list(_coords_from_residue(residue) for residue in residues)


def _validate_dbn_left(dbn: str, open_c: str = '(', skip_c: str = '.') -> str:
    balance = 0

    balanced_dbn = ''
    for s in dbn:
        if s == open_c:
            balance += 1
        elif s == bracket_pairs(open_c):
            balance -= 1

        if balance < 0:
            balanced_dbn += skip_c
            balance += 1
        else:
            balanced_dbn += s

    return balanced_dbn


def _dbn_swapped(dbn: str) -> bool:
    bad_brackets = ['[', ']']
    good_brackets = ['(', ')']

    met_bad_brackets = False

    for c in dbn:
        if c in good_brackets:
            return met_bad_brackets
        if c in bad_brackets:
            met_bad_brackets = True

    return False


def _validate_dbn(dbn: str, open_c: str = '(', skip_c: str = '.') -> str:
    if ('(' not in dbn and ')' not in dbn and ('[' in dbn or ']' in dbn)) or _dbn_swapped(dbn):
        dbn = dbn.replace('(', '#'
                          ).replace(')', '@'
                                    ).replace('[', '('
                                              ).replace(']', ')'
                                                        ).replace('#', '['
                                                                  ).replace('@', ']')

    dbn = _validate_dbn_left(dbn, open_c, skip_c)
    dbn = _validate_dbn_left(dbn[::-1], bracket_pairs(open_c), skip_c)[::-1]

    return dbn


def tree_from_version(tree_version: str):
    return {'v1': RnaTree, 'v2': RnaTree2}[tree_version]


class RnaNode(object):
    def __init__(self, metadata: Dict = None):
        self.parent = None

        self._children = []
        self._metadata = metadata or {'label': 'root', 'enhanced_label': [], 'pdb_residues': []}

    def add_child(self, child: 'RnaNode') -> None:
        self._children.append(child)
        child.parent = self

    def add_residue(self, residue: Residue) -> None:
        self._metadata['pdb_residues'].append(residue)

    def set_postorder_id(self, p_id: int = 0):
        curr_id = p_id
        for child in self.children:
            curr_id = child.set_postorder_id(curr_id)
        self._metadata['dist_id'] = curr_id
        return curr_id + 1

    def set_neighbours(self, kd_tree):
        coords = _coords_from_residues(self._metadata['pdb_residues'])

        # len(coords) in [0, 1, 2] - root, leaf, inner node
        distances = [10, 20, 30, 40, 50]
        for i, coord in enumerate(coords):
            for dist in distances:
                self.enhanced_label[i][2].append(len(kd_tree.query_ball_point(coord, dist)))

        for child in self.children:
            child.set_neighbours(kd_tree)

    def loop_coords(self, skip: int = 0) -> np.ndarray:
        return _coords_from_residues(self.loop_residues[skip:])

    @property
    def label(self) -> str:
        return self._metadata['label']

    @label.setter
    def label(self, new_label: str) -> None:
        self._metadata['label'] = new_label

    @property
    def enhanced_label(self) -> list:
        return self._metadata['enhanced_label']

    @enhanced_label.setter
    def enhanced_label(self, new_enhanced_label: list) -> None:
        self._metadata['enhanced_label'] = new_enhanced_label

    @property
    def children(self) -> List['RnaNode']:
        return self._children

    @property
    def child_count(self) -> int:
        return len(self._children)

    @property
    def child_residue_count(self) -> int:
        return sum(len(child.subtree_residues) for child in self._children)

    @property
    def is_leaf(self) -> bool:
        return self.child_count == 0

    @property
    def labeled_ordered_notation(self, open_c: str = '(') -> str:
        return '{}{}{}{}'.format(open_c,
                                 self.label,
                                 ''.join(child.labeled_ordered_notation for child in self._children),
                                 bracket_pairs(open_c))

    @property
    @memoize
    def subtree_residues(self) -> List[Residue]:
        return self._metadata['pdb_residues'] + list(chain(*map(lambda x: x.subtree_residues, self.children)))

    @property
    def stem_residues(self) -> List[Residue]:
        if self.child_count > 0:
            return self._metadata['pdb_residues'] + list(chain(*map(lambda x: x.stem_residues, self.children)))
        else:
            return []

    @property
    def loop_residues(self) -> List[Residue]:
        if self.child_count > 0 and self.children[0].child_count == 0:
            return list(chain(*map(lambda x: x.subtree_residues, self.children)))
        else:
            return self.children[0].loop_residues

    @property
    def subtree_coords(self) -> np.ndarray:
        return _coords_from_residues(self.subtree_residues)

    @property
    def stem_coords(self) -> np.ndarray:
        return _coords_from_residues(self.stem_residues)

    @property
    def leaf_parent(self) -> 'RnaNode':
        return self.parent if self.is_leaf else self.children[0].leaf_parent

    @property
    def stem_len(self) -> int:
        return 0 if self.is_leaf else self.children[0].stem_len + (1 if self.parent else 0)

    @property
    def loop_size(self) -> int:
        return len(self.children) if len(self.children) > 1 else self.children[0].loop_size

    @property
    def dist_id(self) -> str:
        return self._metadata['dist_id']

    @classmethod
    def get_children(cls, node: 'RnaNode') -> List['RnaNode']:
        return node.children

    @classmethod
    def get_label(cls, node: 'RnaNode') -> str:
        return node.label


class MetaRnaTree(object):
    def __init__(self, data_source, root=None):
        self._data_source = data_source
        self._root = root or RnaNode()
        self._pseudoknots = []
        self._leafs = []

    @property
    def data_source(self):
        return self._data_source.value

    @property
    def root(self):
        return self._root

    @property
    def lon(self):
        return self.root.labeled_ordered_notation

    @property
    def labeled_ordered_notation(self):
        return self.lon

    @property
    def leafs(self):
        return self._leafs

    def generate_postorder_ids(self):
        self.root.set_postorder_id()

    def add_leaf(self, leaf):
        self._leafs.append(leaf)

    def apply_subtree_predicate(self, init_predicate, result_predicate):
        open_nodes = [leaf for leaf in self.leafs if init_predicate(leaf)]
        result_nodes = []

        while open_nodes:
            temporary_nodes = []
            for node in open_nodes:
                if not node.parent:
                    result_nodes.append(node)
                    break

                if result_predicate(node.parent):
                    temporary_nodes.append(node.parent)
                elif result_predicate(node):
                    result_nodes.append(node)
            open_nodes = temporary_nodes

        return result_nodes


class RnaTree(MetaRnaTree):
    @classmethod
    def from_nts(cls, nts, pdb_chain, open_c='(', skip_c='.', label_key='nt_id'):
        additional_bp = defaultdict(list)

        rna_tree = RnaTree(data_source=_DataSource.nts)
        node_stack = [rna_tree.root]

        dbn = _validate_dbn(''.join(nt['dbn'] for nt in nts))

        for i, nt in enumerate(nts):
            c = dbn[i]

            if not c:
                continue

            if c == open_c:
                node_stack.append(RnaNode(metadata={
                    'dbn': c,
                    'label': nt[label_key],
                    'dssr_data': nt,
                    'pdb_residues': [pdb_residues(pdb_chain, nt)]
                }))
            elif c == bracket_pairs(open_c):
                last_node = node_stack.pop()
                last_node.label += '_{}'.format(nt[label_key])
                last_node.add_residue(pdb_residues(pdb_chain, nt))
                if last_node.is_leaf:
                    rna_tree.add_leaf(last_node)
                node_stack[-1].add_child(last_node)
            else:
                leaf = RnaNode(metadata={
                    'dbn': c,
                    'label': nt[label_key],
                    'dssr_data': nt,
                    'pdb_residues': [pdb_residues(pdb_chain, nt)]
                })
                rna_tree.add_leaf(leaf)
                node_stack[-1].add_child(leaf)
                if c not in [skip_c, open_c, bracket_pairs(open_c)]:
                    if bracket_pairs(c) in additional_bp:
                        rna_tree._pseudoknots.append((additional_bp[bracket_pairs(c)], leaf))
                        additional_bp.pop(bracket_pairs(c), None)
                    else:
                        additional_bp[c] = leaf

        rna_tree.generate_postorder_ids()
        return rna_tree

    @classmethod
    def from_dbn(cls, dbn, open_c='(', skip_c='.', close_c=')'):
        rna_tree = RnaTree(data_source=_DataSource.dbn)
        node_stack = [rna_tree.root]

        for c in dbn:
            if c == open_c:
                node_stack.append(RnaNode())
            elif c == skip_c:
                node_stack[-1].add_child(RnaNode())
            elif c == close_c:
                last_node = node_stack.pop()
                node_stack[-1].add_child(last_node)

        return rna_tree

    @classmethod
    def from_lon(cls, lon, open_c='(', close_c=')'):
        rna_tree = None
        node_stack = []
        root_set = False

        for c in lon:
            if c == open_c:
                node_stack.append(RnaNode())
                if not root_set:
                    rna_tree = RnaTree(data_source=_DataSource.lon, root=node_stack[0])
                    root_set = True
            elif c == close_c:
                last_node = node_stack.pop()
                if node_stack:
                    node_stack[-1].add_child(last_node)
            else:
                node_stack[-1].label += c

        return rna_tree


class RnaTree2(MetaRnaTree):
    def generate_neighbourhoods(self):
        coords = self.root.subtree_coords
        kd_tree = KDTree(coords)
        self.root.set_neighbours(kd_tree)

    @classmethod
    def get_node_label(cls, nt):
        # nucleotide_types  -> nt['nt_name'] or nt['nt_code']  .. {'A', 'C', 'G', 'T'}
        # eta, theta  -> (nt['eta'], nt['theta'])
        # neighbors  -> [] empty, will be filled after creation of
        return [nt['nt_name'], (nt['eta'], nt['theta']), []]

    @classmethod
    def from_nts(cls, nts, pdb_chain, open_c='(', skip_c='.'):
        additional_bp = defaultdict(list)

        rna_tree = RnaTree2(data_source=_DataSource.nts)
        node_stack = [rna_tree.root]

        dbn = _validate_dbn(''.join(nt['dbn'] for nt in nts))

        for i, nt in enumerate(nts):
            c = dbn[i]

            if not c:
                continue

            if c == open_c:
                node_stack.append(RnaNode(metadata={
                    'dbn': c,
                    'label': nt['nt_id'],
                    'enhanced_label': [RnaTree2.get_node_label(nt)],
                    'dssr_data': nt,
                    'pdb_residues': [pdb_residues(pdb_chain, nt)]
                }))
            elif c == bracket_pairs(open_c):
                last_node = node_stack.pop()
                last_node.label += '_{}'.format(nt['nt_id'])
                last_node.enhanced_label.append(RnaTree2.get_node_label(nt))
                last_node.add_residue(pdb_residues(pdb_chain, nt))
                if last_node.is_leaf:
                    rna_tree.add_leaf(last_node)
                node_stack[-1].add_child(last_node)
            else:
                leaf = RnaNode(metadata={
                    'dbn': c,
                    'label': nt['nt_id'],
                    'enhanced_label': [RnaTree2.get_node_label(nt)],
                    'dssr_data': nt,
                    'pdb_residues': [pdb_residues(pdb_chain, nt)]
                })
                rna_tree.add_leaf(leaf)
                node_stack[-1].add_child(leaf)
                if c not in [skip_c, open_c, bracket_pairs(open_c)]:
                    if bracket_pairs(c) in additional_bp:
                        rna_tree._pseudoknots.append((additional_bp[bracket_pairs(c)], leaf))
                        additional_bp.pop(bracket_pairs(c), None)
                    else:
                        additional_bp[c] = leaf

        rna_tree.generate_postorder_ids()
        rna_tree.generate_neighbourhoods()
        return rna_tree
