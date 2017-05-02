from predicates import herpin_init_predicate, herpin_result_predicate
from biopython_wrapper import save_structure, calculate_rotran, calculate_rms_with_rotran
from itertools import product, permutations
from rna_tree import MetaRnaTree, RnaNode, _coords_from_residues, _ndarray_from_residues
import numpy as np
from scipy.spatial import distance
from typing import List
from collections import defaultdict
import random


def run_method(method: str,
               structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):

    return {
        'A': method_a, 'B': method_b, 'C': method_c, 'D': method_d, 'E': method_e, 'F': method_f, 'G': method_g,
        'H': method_h, 'I': method_i, 'J': method_j
    }[method.upper()](
        structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=save_args
    )


def _inner_nodes(rna_tree: MetaRnaTree) -> List[RnaNode]:
    if not rna_tree.root.children:
        return []

    open_nodes = [rna_tree.root]
    inner_nodes = []

    while open_nodes:
        new_nodes = []
        for open_node in open_nodes:
            if open_node != rna_tree.root:
                inner_nodes.append(open_node)
            new_nodes += [child for child in open_node.children if not child.is_leaf]
        open_nodes = new_nodes

    return inner_nodes


def _max_paths(paths: List[List[RnaNode]]) -> List[List[RnaNode]]:
    paths_len = [len(path) for path in paths]
    max_path_len = max(paths_len)

    paths_with_len = [(paths_len[i], path) for i, path in enumerate(paths)]
    return list(map(lambda path_len: path_len[1], filter(lambda path_len: path_len[0] == max_path_len, paths_with_len)))


def _longest_node_paths(rna_node: RnaNode) -> List[List[RnaNode]]:
    if not rna_node.children:
        return [[]]

    child_paths = [[rna_node] + path for child in rna_node.children for path in _longest_node_paths(child) if child.children]
    if not child_paths:
        child_paths = [[rna_node]]
    return _max_paths(child_paths)


def _longest_path(rna_tree: MetaRnaTree) -> List[List[RnaNode]]:
    if not rna_tree.root.children:
        return []

    return _max_paths([path for node in rna_tree.root.children for path in _longest_node_paths(node)])


def _max_node_list_distance(dist_table, list1, list2):
    min_width_distance, min_list1, min_list2 = np.sum(dist_table) + 1, None, None

    d = _node_list_distance(dist_table, list1, list2)
    if d < min_width_distance:
        min_width_distance, min_list1, min_list2 = d, list1, list2

    d = _node_list_distance(dist_table, list1[::-1], list2)
    if d < min_width_distance:
        min_width_distance, min_list1, min_list2 = d, list1[::-1], list2

    d = _node_list_distance(dist_table, list1[::-1], list2[::-1])
    if d < min_width_distance:
        min_width_distance, min_list1, min_list2 = d, list1[::-1], list2[::-1]

    d = _node_list_distance(dist_table, list1, list2[::-1])
    if d < min_width_distance:
        min_width_distance, min_list1, min_list2 = d, list1, list2[::-1]

    return min_width_distance / len(min_list1), min_list1, min_list2


def _node_list_distance(dist_table, list1, list2):
    dist = 0
    for node1, node2 in zip(list1, list2):
        dist += dist_table[node1.dist_id, node2.dist_id]
    return dist


def _save_structure(total_rotran, smoothing_rotran, save_args: dict):
    pdb_structure_1 = save_args['structure1']
    chain_1 = save_args['chain1']
    filename_1 = save_args['filename1']

    pdb_structure_2 = save_args['structure2']
    chain_2 = save_args['chain2']
    filename_2 = save_args['filename2']

    save_structure(pdb_structure_1, chain_1, filename_1)

    pdb_structure_2[0][chain_2].transform(total_rotran[0], total_rotran[1])
    pdb_structure_2[0][chain_2].transform(smoothing_rotran[0], smoothing_rotran[1])
    save_structure(pdb_structure_2, chain_2, filename_2)


def method_a(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    herpins_1 = rna_tree_1.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)
    herpins_2 = rna_tree_2.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)

    closest_h1, closest_h2 = None, None
    min_dist = dist.max() + 1
    for h1, h2 in product(herpins_1, herpins_2):
        if dist[h1.dist_id, h2.dist_id] < min_dist:
            closest_h1, closest_h2 = h1, h2
            min_dist = dist[h1.dist_id, h2.dist_id]

    if min_dist > 0:

        h1_stem_len = closest_h1.stem_len
        h2_stem_len = closest_h2.stem_len

        if h2_stem_len > h1_stem_len:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_stem_len, h2_stem_len = h2_stem_len, h1_stem_len

        h1_leaf_parent = closest_h1.leaf_parent
        h2_leaf_parent = closest_h2.leaf_parent

        while h1_stem_len != h2_stem_len:
            closest_h1 = closest_h1.children[0]
            h1_stem_len = closest_h1.stem_len

        if h1_leaf_parent.child_residue_count < h2_leaf_parent.child_residue_count:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_leaf_parent, h2_leaf_parent = h2_leaf_parent, h1_leaf_parent

        h1_stem_coords = closest_h1.stem_coords
        h1_loop_coords = closest_h1.loop_coords(
            skip=h1_leaf_parent.child_residue_count - h2_leaf_parent.child_residue_count
        )

        h1_atoms = np.concatenate((h1_stem_coords, h1_loop_coords), axis=0)
        h2_atoms = closest_h2.subtree_coords
    else:
        h1_atoms = closest_h1.subtree_coords
        h2_atoms = closest_h2.subtree_coords

    rotran = calculate_rotran(h1_atoms, h2_atoms)
    rms, psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, rotran)

    if save_args:
        _save_structure(rotran, smoothing_rotran, save_args)

    return rms, psi, rotran[0], rotran[1]


def method_b(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None, sampling_count=50):
    inner_nodes_1 = _inner_nodes(rna_tree_1)
    inner_nodes_2 = _inner_nodes(rna_tree_2)

    roulette_pieces = []

    for inner_node_1, inner_node_2 in product(inner_nodes_1, inner_nodes_2):
        avg_size = len(inner_node_1.subtree_residues) + len(inner_node_2.subtree_residues) / 2.0
        roulette_pieces.append((
            (inner_node_1, inner_node_2),
            dist[inner_node_1.dist_id, inner_node_2.dist_id].astype(float) / avg_size)
        )

    total_piece_value = sum(map(lambda piece: piece[1], roulette_pieces))
    roulette_pieces = [(piece[0], total_piece_value - piece[1]) for piece in roulette_pieces]
    total_piece_value = sum(map(lambda piece: piece[1], roulette_pieces))

    original_roulette_pieces = roulette_pieces[:]

    total_rms, total_psi, total_rotran, total_smoothing_rotran = None, None, None, None

    for sampling_id in range(sampling_count):
        piece_count = random.randrange(3, 6)

        selected_pairs = []
        for _ in range(piece_count):
            pick = random.uniform(0, total_piece_value)
            current = 0
            piece = None
            for pair, value in roulette_pieces:
                current += value
                if current > pick:
                    piece = pair, value
                    selected_pairs.append(pair)
                    break

            if piece in roulette_pieces:
                roulette_pieces = list(filter(lambda p: all(pair_i not in piece[0] for pair_i in p[0]), roulette_pieces))

        roulette_pieces = original_roulette_pieces

        nodes_a, nodes_b = list(zip(*selected_pairs))

        coords_a = _coords_from_residues(
            [residue for residues in map(lambda node: node._metadata['pdb_residues'], nodes_a) for residue in residues]
        )
        coords_b = _coords_from_residues(
            [residue for residues in map(lambda node: node._metadata['pdb_residues'], nodes_b) for residue in residues]
        )

        rotran = calculate_rotran(coords_a, coords_b)
        new_total_rms, new_total_psi, new_smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, rotran)

        if total_rms and total_rms < new_total_rms:
            continue

        total_rms, total_psi, total_rotran, total_smoothing_rotran = new_total_rms, new_total_psi, rotran, new_smoothing_rotran

    if save_args:
        _save_structure(total_rotran, total_smoothing_rotran, save_args)

    return total_rms, total_psi, total_rotran[0], total_rotran[1]


def method_c(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None, sampling_count=50):
    herpins_1 = rna_tree_1.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)
    herpins_2 = rna_tree_2.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)

    closest_h1, closest_h2 = None, None
    min_dist = dist.max() + 1
    for h1, h2 in product(herpins_1, herpins_2):
        if dist[h1.dist_id, h2.dist_id] < min_dist:
            closest_h1, closest_h2 = h1, h2
            min_dist = dist[h1.dist_id, h2.dist_id]

    if min_dist > 0:

        h1_stem_len = closest_h1.stem_len
        h2_stem_len = closest_h2.stem_len

        if h2_stem_len > h1_stem_len:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_stem_len, h2_stem_len = h2_stem_len, h1_stem_len

        h1_leaf_parent = closest_h1.leaf_parent
        h2_leaf_parent = closest_h2.leaf_parent

        while h1_stem_len != h2_stem_len:
            closest_h1 = closest_h1.children[0]
            h1_stem_len = closest_h1.stem_len

        if h1_leaf_parent.child_residue_count < h2_leaf_parent.child_residue_count:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_leaf_parent, h2_leaf_parent = h2_leaf_parent, h1_leaf_parent

        hairpin_size_diff = h1_leaf_parent.child_residue_count - h2_leaf_parent.child_residue_count

        h1_stem_coords = closest_h1.stem_coords
        h1_loop_coords = closest_h1.loop_coords(
            skip=hairpin_size_diff
        )

        h2_stem_coords = closest_h2.stem_coords
        h2_loop_coords = closest_h2.loop_coords()

        # select the middle residue from hairpin
        loop_size_1 = (len(h1_loop_coords) / 2) - 1
        loop_size_2 = (len(h2_loop_coords) / 2) - 1

        h1_start = int(loop_size_1)
        h1_end = h1_start + 1
        h2_start = int(loop_size_2)
        h2_end = h2_start + 1

        h1_atoms = np.concatenate((h1_stem_coords[0:1], h2_stem_coords[-2:-1], h1_loop_coords[h1_start:h1_end]), axis=0)
        h2_atoms = np.concatenate((h2_stem_coords[0:1], h2_stem_coords[-2:-1], h2_loop_coords[h2_start:h2_end]), axis=0)
    else:
        h1_atoms = closest_h1.subtree_coords
        h2_atoms = closest_h2.subtree_coords

    rotran = calculate_rotran(h1_atoms, h2_atoms)
    total_rms, total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, rotran)

    starting_coords_a = h1_atoms
    starting_coords_b = h2_atoms

    inner_nodes_1 = _inner_nodes(rna_tree_1)
    inner_nodes_2 = _inner_nodes(rna_tree_2)

    roulette_pieces = []

    subtree_coords_1 = rna_tree_1.root.subtree_coords
    subtree_coords_2 = rna_tree_2.root.subtree_coords

    try:
        starting_coord_index_a = np.where(subtree_coords_1 == starting_coords_a[2])[0][0]
        starting_coord_index_b = np.where(subtree_coords_2 == starting_coords_b[2])[0][0]
    except Exception as e:
        starting_coord_index_a = np.where(subtree_coords_1 == starting_coords_b[2])[0][0]
        starting_coord_index_b = np.where(subtree_coords_2 == starting_coords_a[2])[0][0]

    residue_distances_a = distance.squareform(distance.pdist(subtree_coords_1, 'euclidean'))
    residue_distances_b = distance.squareform(distance.pdist(subtree_coords_2, 'euclidean'))

    for inner_node_1, inner_node_2 in product(inner_nodes_1, inner_nodes_2):

        avg_size = len(inner_node_1.subtree_residues) + len(inner_node_2.subtree_residues) / 2.0
        avg_distance = (residue_distances_a[inner_node_1.dist_id, starting_coord_index_a] + residue_distances_b[inner_node_2.dist_id, starting_coord_index_b]) / 2.0

        roulette_pieces.append((
            (inner_node_1, inner_node_2),
            avg_distance * dist[inner_node_1.dist_id, inner_node_2.dist_id].astype(float) / avg_size)
        )

    total_piece_value = sum(map(lambda piece: piece[1], roulette_pieces))
    roulette_pieces = [(piece[0], total_piece_value - piece[1]) for piece in roulette_pieces]
    total_piece_value = sum(map(lambda piece: piece[1], roulette_pieces))

    original_roulette_pieces = roulette_pieces[:]

    total_rotran = rotran
    total_smoothing_rotran = smoothing_rotran
    for sampling_id in range(sampling_count):
        piece_count = random.randrange(1, 4)

        selected_pairs = []
        for _ in range(piece_count):
            pick = random.uniform(0, total_piece_value)
            current = 0
            piece = None
            for pair, value in roulette_pieces:
                current += value
                if current > pick:
                    piece = pair, value
                    selected_pairs.append(pair)
                    break

            if piece in roulette_pieces:
                roulette_pieces = list(filter(lambda p: all(pair_i not in piece[0] for pair_i in p[0]), roulette_pieces))

        roulette_pieces = original_roulette_pieces

        nodes_a, nodes_b = list(zip(*selected_pairs))

        coords_a = _coords_from_residues(
            [residue for residues in map(lambda node: node._metadata['pdb_residues'], nodes_a) for residue in residues]
        )
        coords_b = _coords_from_residues(
            [residue for residues in map(lambda node: node._metadata['pdb_residues'], nodes_b) for residue in residues]
        )

        coords_a = np.concatenate((coords_a, starting_coords_a), axis=0)
        coords_b = np.concatenate((coords_b, starting_coords_b), axis=0)

        rotran = calculate_rotran(coords_a, coords_b)
        new_total_rms, new_total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, rotran)

        if total_rms and total_rms < new_total_rms:
            continue

        total_rms, total_psi, total_rotran, total_smoothing_rotran = new_total_rms, new_total_psi, rotran, smoothing_rotran

    # apply total_rotran and final_smoothing_rotran to pdb_2_structure

    if save_args:
        _save_structure(total_rotran, total_smoothing_rotran, save_args)

    return total_rms, total_psi, total_rotran[0], total_rotran[1]


def method_d(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    longest_paths_1 = _longest_path(rna_tree_1)
    longest_paths_2 = _longest_path(rna_tree_2)

    if len(longest_paths_1[0]) < len(longest_paths_2[0]):
        longest_paths_1, longest_paths_2 = longest_paths_2, longest_paths_1
        dist = dist.T
        structure_1_coordinates, structure_2_coordinates = structure_2_coordinates, structure_1_coordinates

        if save_args:
            save_args['structure1'], save_args['structure2'] = save_args['structure2'], save_args['structure1']
            save_args['chain1'], save_args['chain2'] = save_args['chain2'], save_args['chain1']
            save_args['filename1'], save_args['filename2'] = save_args['filename2'], save_args['filename1']

    longest_path_2_len = len(longest_paths_2[0])
    len_diff = len(longest_paths_1[0]) - longest_path_2_len

    min_dist = None
    selected_offset = -1
    selected_longest_path_1, selected_longest_path_2 = None, None
    for k, (longest_path_1, longest_path_2) in enumerate(product(longest_paths_1, longest_paths_2)):
        if len(longest_path_1) == len(longest_path_2):
            total_dist = 0
            for node_1, node_2 in zip(longest_path_1, longest_path_2):
                total_dist += dist[node_1.dist_id, node_2.dist_id]
            total_dist /= len(longest_path_2)
            if not min_dist or total_dist < min_dist:
                min_dist = total_dist
                selected_offset = -1
                selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2
        else:
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1[offset:-(len_diff-offset)], longest_path_2):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_2)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_offset = offset
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2

    longest_path_1, longest_path_2 = selected_longest_path_1, selected_longest_path_2
    if selected_offset > -1:
        longest_path_1 = longest_path_1[selected_offset:-(len_diff-selected_offset)]

    total_rms, total_psi, total_rotran, final_smoothing_rotran = None, None, None, None
    for i in range(2, max(longest_path_2_len, 3)):
        middle_idx = int(longest_path_2_len / i)
        selected_residues_1 = [longest_path_1[0], longest_path_1[-1]]
        selected_residues_2 = [longest_path_2[0], longest_path_2[-1]]

        for j in range(1, i):
            new_residue_1 = longest_path_1[middle_idx * j]
            new_residue_2 = longest_path_2[middle_idx * j]
            if new_residue_1 in selected_residues_1 or new_residue_2 in selected_residues_2:
                continue
            selected_residues_1.append(new_residue_1)
            selected_residues_2.append(new_residue_2)

        coords_1 = _coords_from_residues(
            [residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_1) for residue in residues]
        )
        coords_2 = _coords_from_residues(
            [residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_2) for residue in residues]
        )

        rotran = calculate_rotran(coords_1, coords_2)
        new_total_rms, new_total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, rotran)
        if not total_rms or new_total_rms < total_rms:
            total_rms, total_psi, total_rotran, final_smoothing_rotran = new_total_rms, new_total_psi, rotran, smoothing_rotran

    if save_args:
        _save_structure(total_rotran, final_smoothing_rotran, save_args)

    return total_rms, total_psi, total_rotran[0], total_rotran[1]


def method_e(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    widths_1 = defaultdict(list)
    widths_2 = defaultdict(list)

    for leaf in rna_tree_1.leafs:
        widths_1[leaf.parent].append(leaf)
    widths_1 = list(widths_1.values())

    for leaf in rna_tree_2.leafs:
        widths_2[leaf.parent].append(leaf)
    widths_2 = list(widths_2.values())

    aligned_widths_1 = []
    aligned_widths_2 = []

    widths_dist = np.zeros((len(widths_1), len(widths_2)))
    widths_parts = {}

    for i, width_1 in enumerate(widths_1):
        for j, width_2 in enumerate(widths_2):
            min_width_distance, min_width_1, min_width_2, remove_width_1, remove_width_2 = np.sum(dist) + 1, None, None, None, None
            if len(width_1) > len(width_2):
                len_diff = len(width_1) - len(width_2)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1[offset:-(len_diff - offset)], width_2)
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            elif len(width_1) < len(width_2):
                len_diff = len(width_2) - len(width_1)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1, width_2[offset:-(len_diff - offset)])
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            else:
                min_width_distance, min_width_1, min_width_2 = _max_node_list_distance(dist, width_1, width_2)

            widths_dist[i, j] = min_width_distance
            widths_parts[(i, j)] = (min_width_1, min_width_2)

    while widths_dist.shape[0] > 0 and widths_dist.shape[1] > 0:
        indices = np.unravel_index(np.argmin(widths_dist), dims=widths_dist.shape)
        min_width_1, min_width_2 = widths_parts[indices]
        aligned_widths_1.append(min_width_1)
        aligned_widths_2.append(min_width_2)
        widths_dist = np.delete(widths_dist, (indices[0]), axis=0)
        widths_dist = np.delete(widths_dist, (indices[1]), axis=1)

    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    current_aligned_widths_1 = []
    current_aligned_widths_2 = []
    for nodes1, nodes2 in zip(aligned_widths_1, aligned_widths_2):
        current_aligned_widths_1 += nodes1
        current_aligned_widths_2 += nodes2

        width_1_residues = [node._metadata['pdb_residues'][0] for node in current_aligned_widths_1]
        width_2_residues = [node._metadata['pdb_residues'][0] for node in current_aligned_widths_2]

        width_1_coords = _coords_from_residues(width_1_residues)
        width_2_coords = _coords_from_residues(width_2_residues)
        total_rotran = calculate_rotran(width_1_coords, width_2_coords)
        total_rms, total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, total_rotran)
        if not final_total_rms or total_psi >= final_total_psi:
            final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = total_rotran, total_rms, total_psi, smoothing_rotran
        else:
            for node1 in nodes1:
                current_aligned_widths_1.remove(node1)
            for node2 in nodes2:
                current_aligned_widths_2.remove(node2)

    if save_args:
        _save_structure(final_total_rotran, final_smoothing_rotran, save_args)

    return final_total_rms, final_total_psi, final_total_rotran[0], final_total_rotran[1]


def method_f(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    widths_1 = defaultdict(list)
    widths_2 = defaultdict(list)

    for leaf in rna_tree_1.leafs:
        widths_1[leaf.parent].append(leaf)
    widths_1 = list(widths_1.values())

    for leaf in rna_tree_2.leafs:
        widths_2[leaf.parent].append(leaf)
    widths_2 = list(widths_2.values())

    aligned_widths_1 = []
    aligned_widths_2 = []

    widths_dist = np.zeros((len(widths_1), len(widths_2)))
    widths_parts = {}

    for i, width_1 in enumerate(widths_1):
        for j, width_2 in enumerate(widths_2):
            min_width_distance, min_width_1, min_width_2, remove_width_1, remove_width_2 = np.sum(dist) + 1, None, None, None, None
            if len(width_1) > len(width_2):
                len_diff = len(width_1) - len(width_2)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1[offset:-(len_diff - offset)], width_2)
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            elif len(width_1) < len(width_2):
                len_diff = len(width_2) - len(width_1)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1, width_2[offset:-(len_diff - offset)])
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            else:
                min_width_distance, min_width_1, min_width_2 = _max_node_list_distance(dist, width_1, width_2)

            widths_dist[i, j] = min_width_distance
            widths_parts[(i, j)] = (min_width_1, min_width_2)

    while widths_dist.shape[0] > 0 and widths_dist.shape[1] > 0:
        indices = np.unravel_index(np.argmin(widths_dist), dims=widths_dist.shape)
        min_width_1, min_width_2 = widths_parts[indices]
        aligned_widths_1.append([node._metadata['pdb_residues'][0] for node in min_width_1])
        aligned_widths_2.append([node._metadata['pdb_residues'][0] for node in min_width_2])
        widths_dist = np.delete(widths_dist, (indices[0]), axis=0)
        widths_dist = np.delete(widths_dist, (indices[1]), axis=1)

    herpins_1 = rna_tree_1.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)
    herpins_2 = rna_tree_2.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)

    closest_h1, closest_h2 = None, None
    min_dist = dist.max() + 1
    for h1, h2 in product(herpins_1, herpins_2):
        if dist[h1.dist_id, h2.dist_id] < min_dist:
            closest_h1, closest_h2 = h1, h2
            min_dist = dist[h1.dist_id, h2.dist_id]

    if min_dist > 0:

        h1_stem_len = closest_h1.stem_len
        h2_stem_len = closest_h2.stem_len

        if h2_stem_len > h1_stem_len:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_stem_len, h2_stem_len = h2_stem_len, h1_stem_len

        h1_leaf_parent = closest_h1.leaf_parent
        h2_leaf_parent = closest_h2.leaf_parent

        while h1_stem_len != h2_stem_len:
            closest_h1 = closest_h1.children[0]
            h1_stem_len = closest_h1.stem_len

        if h1_leaf_parent.child_residue_count < h2_leaf_parent.child_residue_count:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_leaf_parent, h2_leaf_parent = h2_leaf_parent, h1_leaf_parent

        h1_stem_residues = closest_h1.stem_residues
        h1_loop_residues = closest_h1.loop_residues[h1_leaf_parent.child_residue_count - h2_leaf_parent.child_residue_count:]

        h1_residues = h1_stem_residues + h1_loop_residues
        h2_residues = closest_h2.subtree_residues
    else:
        h1_residues = closest_h1.subtree_residues
        h2_residues = closest_h2.subtree_residues

    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    potential_orderings = [
        ([h1_residues] + aligned_widths_1, [h2_residues] + aligned_widths_2),
        ([h1_residues] + aligned_widths_1[::-1], [h2_residues] + aligned_widths_2[::-1]),
        (aligned_widths_1 + [h1_residues], aligned_widths_2 + [h2_residues]),
        (aligned_widths_1[::-1] + [h1_residues], aligned_widths_2[::-1] + [h2_residues])
    ]

    all_total_data = []
    for ordered_aligned_widths_1, ordered_aligned_widths_2, in potential_orderings:
        current_aligned_widths_1 = []
        current_aligned_widths_2 = []
        for nodes1, nodes2 in zip(ordered_aligned_widths_1, ordered_aligned_widths_2):
            current_aligned_widths_1 += nodes1
            current_aligned_widths_2 += nodes2

            width_1_residues = current_aligned_widths_1
            width_2_residues = current_aligned_widths_2

            width_1_coords = _coords_from_residues(width_1_residues)
            width_2_coords = _coords_from_residues(width_2_residues)
            total_rotran = calculate_rotran(width_1_coords, width_2_coords)
            total_rms, total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, total_rotran)
            if not final_total_rms or total_psi >= final_total_psi:
                final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = total_rotran, total_rms, total_psi, smoothing_rotran
            else:
                for node1 in nodes1:
                    current_aligned_widths_1.remove(node1)
                for node2 in nodes2:
                    current_aligned_widths_2.remove(node2)
        all_total_data.append((final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran))
        final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = max(all_total_data, key=lambda t: t[2])

    if save_args:
        _save_structure(final_total_rotran, final_smoothing_rotran, save_args)

    return final_total_rms, final_total_psi, final_total_rotran[0], final_total_rotran[1]


def method_g(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    widths_1 = defaultdict(list)
    widths_2 = defaultdict(list)

    for leaf in rna_tree_1.leafs:
        widths_1[leaf.parent].append(leaf)
    widths_1 = list(widths_1.values())

    for leaf in rna_tree_2.leafs:
        widths_2[leaf.parent].append(leaf)
    widths_2 = list(widths_2.values())

    aligned_widths_1 = []
    aligned_widths_2 = []

    widths_dist = np.zeros((len(widths_1), len(widths_2)))
    widths_parts = {}

    for i, width_1 in enumerate(widths_1):
        for j, width_2 in enumerate(widths_2):
            min_width_distance, min_width_1, min_width_2, remove_width_1, remove_width_2 = np.sum(dist) + 1, None, None, None, None
            if len(width_1) > len(width_2):
                len_diff = len(width_1) - len(width_2)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1[offset:-(len_diff - offset)], width_2)
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            elif len(width_1) < len(width_2):
                len_diff = len(width_2) - len(width_1)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1, width_2[offset:-(len_diff - offset)])
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            else:
                min_width_distance, min_width_1, min_width_2 = _max_node_list_distance(dist, width_1, width_2)

            widths_dist[i, j] = min_width_distance
            widths_parts[(i, j)] = (min_width_1, min_width_2)

    i = 0
    while widths_dist.shape[0] > 0 and widths_dist.shape[1] > 0 and i < 10:
        i += 1
        indices = np.unravel_index(np.argmin(widths_dist), dims=widths_dist.shape)
        min_width_1, min_width_2 = widths_parts[indices]
        aligned_widths_1.append([node._metadata['pdb_residues'][0] for node in min_width_1])
        aligned_widths_2.append([node._metadata['pdb_residues'][0] for node in min_width_2])
        widths_dist = np.delete(widths_dist, (indices[0]), axis=0)
        widths_dist = np.delete(widths_dist, (indices[1]), axis=1)

    # ---- Hairpin section
    herpins_1 = rna_tree_1.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)
    herpins_2 = rna_tree_2.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)

    closest_h1, closest_h2 = None, None
    min_dist = dist.max() + 1
    for h1, h2 in product(herpins_1, herpins_2):
        if dist[h1.dist_id, h2.dist_id] < min_dist:
            closest_h1, closest_h2 = h1, h2
            min_dist = dist[h1.dist_id, h2.dist_id]

    if min_dist > 0:

        h1_stem_len = closest_h1.stem_len
        h2_stem_len = closest_h2.stem_len

        if h2_stem_len > h1_stem_len:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_stem_len, h2_stem_len = h2_stem_len, h1_stem_len

        h1_leaf_parent = closest_h1.leaf_parent
        h2_leaf_parent = closest_h2.leaf_parent

        while h1_stem_len != h2_stem_len:
            closest_h1 = closest_h1.children[0]
            h1_stem_len = closest_h1.stem_len

        if h1_leaf_parent.child_residue_count < h2_leaf_parent.child_residue_count:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_leaf_parent, h2_leaf_parent = h2_leaf_parent, h1_leaf_parent

        h1_stem_residues = closest_h1.stem_residues
        h1_loop_residues = closest_h1.loop_residues[h1_leaf_parent.child_residue_count - h2_leaf_parent.child_residue_count:]

        h1_residues = h1_stem_residues + h1_loop_residues
        h2_residues = closest_h2.subtree_residues
    else:
        h1_residues = closest_h1.subtree_residues
        h2_residues = closest_h2.subtree_residues

    # ---- Paths section
    longest_paths_1 = _longest_path(rna_tree_1)
    longest_paths_2 = _longest_path(rna_tree_2)

    min_dist = None
    selected_longest_path_1, selected_longest_path_2 = None, None
    for k, (longest_path_1, longest_path_2) in enumerate(product(longest_paths_1, longest_paths_2)):
        if len(longest_path_1) == len(longest_path_2):
            total_dist = 0
            for node_1, node_2 in zip(longest_path_1, longest_path_2):
                total_dist += dist[node_1.dist_id, node_2.dist_id]
            total_dist /= len(longest_path_2)
            if not min_dist or total_dist < min_dist:
                min_dist = total_dist
                selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2
        elif len(longest_paths_1[0]) > len(longest_paths_2[0]):
            len_diff = len(longest_paths_1[0]) - len(longest_paths_2[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1[offset:-(len_diff - offset)], longest_path_2):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_2)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1[offset:-(len_diff - offset)], longest_path_2
        else:
            len_diff = len(longest_paths_2[0]) - len(longest_paths_1[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1, longest_path_2[offset:-(len_diff - offset)]):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_1)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2[offset:-(len_diff - offset)]

    longest_path_1, longest_path_2 = selected_longest_path_1, selected_longest_path_2

    path_residues_1 = []
    path_residues_2 = []
    for i in range(2, max(len(longest_path_1), 3)):
        middle_idx = int(len(longest_path_1) / i)
        selected_residues_1 = [longest_path_1[0], longest_path_1[-1]]
        selected_residues_2 = [longest_path_2[0], longest_path_2[-1]]

        for j in range(1, i):
            new_residue_1 = longest_path_1[middle_idx * j]
            new_residue_2 = longest_path_2[middle_idx * j]
            if new_residue_1 in selected_residues_1 or new_residue_2 in selected_residues_2:
                continue
            selected_residues_1.append(new_residue_1)
            selected_residues_2.append(new_residue_2)

        path_residues_1.append([residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_1) for residue in residues])
        path_residues_2.append([residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_2) for residue in residues])

    # --- Compound section
    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    pairings1 = [h1_residues] + aligned_widths_1 + path_residues_1
    pairings2 = [h2_residues] + aligned_widths_2 + path_residues_2

    all_total_data = []
    for permutation in permutations(range(len(pairings1))):
        ordered_aligned_widths_1 = []
        ordered_aligned_widths_2 = []
        for idx in permutation:
            ordered_aligned_widths_1.append(pairings1[idx])
            ordered_aligned_widths_2.append(pairings2[idx])

        current_aligned_widths_1 = []
        current_aligned_widths_2 = []
        for nodes1, nodes2 in zip(ordered_aligned_widths_1, ordered_aligned_widths_2):
            current_aligned_widths_1 += nodes1
            current_aligned_widths_2 += nodes2

            width_1_coords = _coords_from_residues(current_aligned_widths_1)
            width_2_coords = _coords_from_residues(current_aligned_widths_2)
            total_rotran = calculate_rotran(width_1_coords, width_2_coords)
            total_rms, total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, total_rotran)
            if not final_total_rms or total_psi >= final_total_psi:
                final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = total_rotran, total_rms, total_psi, smoothing_rotran
            else:
                for node1 in nodes1:
                    current_aligned_widths_1.remove(node1)
                for node2 in nodes2:
                    current_aligned_widths_2.remove(node2)
        all_total_data.append((final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran))
        final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = max(all_total_data, key=lambda t: t[2])

    if save_args:
        _save_structure(final_total_rotran, final_smoothing_rotran, save_args)

    return final_total_rms, final_total_psi, final_total_rotran[0], final_total_rotran[1]


def method_h(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    widths_1 = defaultdict(list)
    widths_2 = defaultdict(list)

    for leaf in rna_tree_1.leafs:
        widths_1[leaf.parent].append(leaf)
    widths_1 = list(widths_1.values())

    for leaf in rna_tree_2.leafs:
        widths_2[leaf.parent].append(leaf)
    widths_2 = list(widths_2.values())

    aligned_widths_1 = []
    aligned_widths_2 = []

    widths_dist = np.zeros((len(widths_1), len(widths_2)))
    widths_parts = {}

    for i, width_1 in enumerate(widths_1):
        for j, width_2 in enumerate(widths_2):
            min_width_distance, min_width_1, min_width_2, remove_width_1, remove_width_2 = np.sum(dist) + 1, None, None, None, None
            if len(width_1) > len(width_2):
                len_diff = len(width_1) - len(width_2)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1[offset:-(len_diff - offset)], width_2)
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            elif len(width_1) < len(width_2):
                len_diff = len(width_2) - len(width_1)
                for offset in range(len_diff):
                    width_distance, next_width_1, next_width_2 = _max_node_list_distance(dist, width_1, width_2[offset:-(len_diff - offset)])
                    if width_distance < min_width_distance or (width_distance == min_width_distance and len(next_width_1) > len(min_width_1)):
                        min_width_distance, min_width_1, min_width_2 = width_distance, next_width_1, next_width_2
            else:
                min_width_distance, min_width_1, min_width_2 = _max_node_list_distance(dist, width_1, width_2)

            widths_dist[i, j] = min_width_distance
            widths_parts[(i, j)] = (min_width_1, min_width_2)

    while widths_dist.shape[0] > 0 and widths_dist.shape[1] > 0:
        indices = np.unravel_index(np.argmin(widths_dist), dims=widths_dist.shape)
        min_width_1, min_width_2 = widths_parts[indices]
        aligned_widths_1.append([node._metadata['pdb_residues'][0] for node in min_width_1])
        aligned_widths_2.append([node._metadata['pdb_residues'][0] for node in min_width_2])
        widths_dist = np.delete(widths_dist, (indices[0]), axis=0)
        widths_dist = np.delete(widths_dist, (indices[1]), axis=1)

    # ---- Hairpin section
    herpins_1 = rna_tree_1.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)
    herpins_2 = rna_tree_2.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)

    closest_h1, closest_h2 = None, None
    min_dist = dist.max() + 1
    for h1, h2 in product(herpins_1, herpins_2):
        if dist[h1.dist_id, h2.dist_id] < min_dist:
            closest_h1, closest_h2 = h1, h2
            min_dist = dist[h1.dist_id, h2.dist_id]

    if min_dist > 0:

        h1_stem_len = closest_h1.stem_len
        h2_stem_len = closest_h2.stem_len

        if h2_stem_len > h1_stem_len:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_stem_len, h2_stem_len = h2_stem_len, h1_stem_len

        h1_leaf_parent = closest_h1.leaf_parent
        h2_leaf_parent = closest_h2.leaf_parent

        while h1_stem_len != h2_stem_len:
            closest_h1 = closest_h1.children[0]
            h1_stem_len = closest_h1.stem_len

        if h1_leaf_parent.child_residue_count < h2_leaf_parent.child_residue_count:
            closest_h1, closest_h2 = closest_h2, closest_h1
            h1_leaf_parent, h2_leaf_parent = h2_leaf_parent, h1_leaf_parent

        h1_stem_residues = closest_h1.stem_residues
        h1_loop_residues = closest_h1.loop_residues[h1_leaf_parent.child_residue_count - h2_leaf_parent.child_residue_count:]

        h1_residues = h1_stem_residues + h1_loop_residues
        h2_residues = closest_h2.subtree_residues
    else:
        h1_residues = closest_h1.subtree_residues
        h2_residues = closest_h2.subtree_residues

    # ---- Paths section
    longest_paths_1 = _longest_path(rna_tree_1)
    longest_paths_2 = _longest_path(rna_tree_2)

    min_dist = None
    selected_longest_path_1, selected_longest_path_2 = None, None
    for k, (longest_path_1, longest_path_2) in enumerate(product(longest_paths_1, longest_paths_2)):
        if len(longest_path_1) == len(longest_path_2):
            total_dist = 0
            for node_1, node_2 in zip(longest_path_1, longest_path_2):
                total_dist += dist[node_1.dist_id, node_2.dist_id]
            total_dist /= len(longest_path_2)
            if not min_dist or total_dist < min_dist:
                min_dist = total_dist
                selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2
        elif len(longest_paths_1[0]) > len(longest_paths_2[0]):
            len_diff = len(longest_paths_1[0]) - len(longest_paths_2[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1[offset:-(len_diff - offset)], longest_path_2):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_2)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1[offset:-(len_diff - offset)], longest_path_2
        else:
            len_diff = len(longest_paths_2[0]) - len(longest_paths_1[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1, longest_path_2[offset:-(len_diff - offset)]):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_1)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2[offset:-(len_diff - offset)]

    longest_path_1, longest_path_2 = selected_longest_path_1, selected_longest_path_2

    path_residues_1 = []
    path_residues_2 = []
    for i in range(2, max(len(longest_path_1), 3)):
        middle_idx = int(len(longest_path_1) / i)
        selected_residues_1 = [longest_path_1[0], longest_path_1[-1]]
        selected_residues_2 = [longest_path_2[0], longest_path_2[-1]]

        for j in range(1, i):
            new_residue_1 = longest_path_1[middle_idx * j]
            new_residue_2 = longest_path_2[middle_idx * j]
            if new_residue_1 in selected_residues_1 or new_residue_2 in selected_residues_2:
                continue
            selected_residues_1.append(new_residue_1)
            selected_residues_2.append(new_residue_2)

        path_residues_1.append([residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_1) for residue in residues])
        path_residues_2.append([residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_2) for residue in residues])

    # --- Compound section
    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    summed_path_residues_1 = []
    summed_path_residues_2 = []
    for path1, path2 in zip(path_residues_1, path_residues_2):
        for node1, node2 in zip(path1, path2):
            if node1 not in summed_path_residues_1 and node2 not in summed_path_residues_2:
                summed_path_residues_1.append(node1)
                summed_path_residues_2.append(node2)

    summed_path_residues_1_r = []
    summed_path_residues_2_r = []
    for path1, path2 in zip(path_residues_1[::-1], path_residues_2[::-1]):
        for node1, node2 in zip(path1, path2):
            if node1 not in summed_path_residues_1 and node2 not in summed_path_residues_2:
                summed_path_residues_1_r.append(node1)
                summed_path_residues_2_r.append(node2)

    summed_path_residues_1 = [[node] for node in summed_path_residues_1]
    summed_path_residues_2 = [[node] for node in summed_path_residues_2]
    summed_path_residues_1_r = [[node] for node in summed_path_residues_1_r]
    summed_path_residues_2_r = [[node] for node in summed_path_residues_2_r]

    h1_residues = _ndarray_from_residues(h1_residues)
    h2_residues = _ndarray_from_residues(h2_residues)
    aligned_widths_1 = [_ndarray_from_residues(l) for l in aligned_widths_1]
    aligned_widths_2 = [_ndarray_from_residues(l) for l in aligned_widths_2]
    summed_path_residues_1 = [_ndarray_from_residues(l) for l in summed_path_residues_1]
    summed_path_residues_2 = [_ndarray_from_residues(l) for l in summed_path_residues_2]
    summed_path_residues_1_r = [_ndarray_from_residues(l) for l in summed_path_residues_1_r]
    summed_path_residues_2_r = [_ndarray_from_residues(l) for l in summed_path_residues_2_r]

    potential_orderings = [
        ([h1_residues] + aligned_widths_1 + summed_path_residues_1, [h2_residues] + aligned_widths_2 + summed_path_residues_2),
        ([h1_residues] + summed_path_residues_1 + aligned_widths_1, [h2_residues] + summed_path_residues_2 + aligned_widths_2),
        (aligned_widths_1 + [h1_residues] + summed_path_residues_1, aligned_widths_2 + [h2_residues] + summed_path_residues_2),
        (aligned_widths_1 + summed_path_residues_1 + [h1_residues], aligned_widths_2 + summed_path_residues_2 + [h2_residues]),
        (summed_path_residues_1 + [h1_residues] + aligned_widths_1, summed_path_residues_2 + [h2_residues] + aligned_widths_2),
        (summed_path_residues_1 + aligned_widths_1 + [h1_residues], summed_path_residues_2 + aligned_widths_2 + [h2_residues]),
        ([h1_residues] + aligned_widths_1 + summed_path_residues_1_r, [h2_residues] + aligned_widths_2 + summed_path_residues_2_r),
        ([h1_residues] + summed_path_residues_1_r + aligned_widths_1, [h2_residues] + summed_path_residues_2_r + aligned_widths_2),
        (aligned_widths_1 + [h1_residues] + summed_path_residues_1_r, aligned_widths_2 + [h2_residues] + summed_path_residues_2_r),
        (aligned_widths_1 + summed_path_residues_1_r + [h1_residues], aligned_widths_2 + summed_path_residues_2_r + [h2_residues]),
        (summed_path_residues_1_r + [h1_residues] + aligned_widths_1, summed_path_residues_2_r + [h2_residues] + aligned_widths_2),
        (summed_path_residues_1_r + aligned_widths_1 + [h1_residues], summed_path_residues_2_r + aligned_widths_2 + [h2_residues]),
    ]

    all_total_data = []

    for i, (ordered_aligned_widths_1, ordered_aligned_widths_2) in enumerate(potential_orderings):
        current_aligned_widths_1 = []
        current_aligned_widths_2 = []
        k = 0
        for nodes1, nodes2 in zip(ordered_aligned_widths_1, ordered_aligned_widths_2):
            k += 1

            width_1_coords = np.vstack(current_aligned_widths_1 + nodes1)
            width_2_coords = np.vstack(current_aligned_widths_2 + nodes2)
            total_rotran = calculate_rotran(width_1_coords, width_2_coords)
            total_rms, total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, total_rotran)
            if not final_total_rms or total_psi > final_total_psi or (np.isclose(total_psi, final_total_psi) and total_rms < final_total_rms):
                current_aligned_widths_1 += nodes1
                current_aligned_widths_2 += nodes2
                final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = total_rotran, total_rms, total_psi, smoothing_rotran
        all_total_data.append((final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran))
        final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = max(all_total_data, key=lambda t: t[2])

    if save_args:
        _save_structure(final_total_rotran, final_smoothing_rotran, save_args)

    return final_total_rms, final_total_psi, final_total_rotran[0], final_total_rotran[1]


def method_i(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    # best hairpin
    # longest path
    # root leaf-children
    # combine

    # ---- Hairpin section
    herpins_1 = rna_tree_1.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)
    herpins_2 = rna_tree_2.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)

    closest_h1, closest_h2 = None, None
    min_dist = dist.max() + 1
    for h1, h2 in product(herpins_1, herpins_2):
        if dist[h1.dist_id, h2.dist_id] < min_dist:
            closest_h1, closest_h2 = h1, h2
            min_dist = dist[h1.dist_id, h2.dist_id]

    if min_dist > 0:
        if closest_h1.label == 'root':
            closest_h1 = closest_h1.children[0]

        if closest_h2.label == 'root':
            closest_h2 = closest_h2.children[0]

        h1_stem_residues, h2_stem_residues = [closest_h1], [closest_h2]
        current_node = closest_h1
        while True:
            child = current_node.children[0]
            if not child.is_leaf:
                h1_stem_residues.append(child)
                current_node = child
            else:
                break
        current_node = closest_h2
        while True:
            child = current_node.children[0]
            if not child.is_leaf:
                h2_stem_residues.append(child)
                current_node = child
            else:
                break

        h1_loop_residues = closest_h1.leaf_parent.children
        h2_loop_residues = closest_h2.leaf_parent.children

        min_stem_dist = None
        min_stem_1, min_stem_2, min_loop_1, min_loop_2 = None, None, None, None

        if len(h1_stem_residues) == len(h2_stem_residues):
            min_stem_1, min_stem_2 = h1_stem_residues, h2_stem_residues
        elif len(h1_stem_residues) > len(h2_stem_residues):
            len_diff = len(h1_stem_residues) - len(h2_stem_residues)
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(h1_stem_residues[offset:-(len_diff - offset)], h2_stem_residues):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(h2_stem_residues)
                if not min_stem_dist or total_dist < min_dist:
                    min_stem_dist = total_dist
                    min_stem_1, min_stem_2 = h1_stem_residues[offset:-(len_diff - offset)], h2_stem_residues
        else:
            len_diff = len(h2_stem_residues) - len(h1_stem_residues)
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(h1_stem_residues, h2_stem_residues[offset:-(len_diff - offset)]):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(h1_stem_residues)
                if not min_stem_dist or total_dist < min_dist:
                    min_stem_dist = total_dist
                    min_stem_1, min_stem_2 = h1_stem_residues, h2_stem_residues[offset:-(len_diff - offset)]

        min_loop_distance, min_loop_1, min_loop_2 = np.sum(dist) + 1, None, None
        if len(h1_loop_residues) > len(h2_loop_residues):
            len_diff = len(h1_loop_residues) - len(h2_loop_residues)
            for offset in range(len_diff):
                loop_distance, next_loop_1, next_loop_2 = _max_node_list_distance(dist, h1_loop_residues[offset:-(len_diff - offset)], h2_loop_residues)
                if loop_distance < min_loop_distance or (loop_distance == min_loop_distance and len(next_loop_1) > len(min_loop_1)):
                    min_loop_distance, min_loop_1, min_loop_2 = loop_distance, next_loop_1, next_loop_2
        elif len(h1_loop_residues) < len(h2_loop_residues):
            len_diff = len(h2_loop_residues) - len(h1_loop_residues)
            for offset in range(len_diff):
                loop_distance, next_loop_1, next_loop_2 = _max_node_list_distance(dist, h1_loop_residues, h2_loop_residues[offset:-(len_diff - offset)])
                if loop_distance < min_loop_distance or (loop_distance == min_loop_distance and len(next_loop_1) > len(min_loop_1)):
                    min_loop_distance, min_loop_1, min_loop_2 = loop_distance, next_loop_1, next_loop_2
        else:
            min_loop_distance, min_loop_1, min_loop_2 = _max_node_list_distance(dist, h1_loop_residues, h2_loop_residues)

        h1_residues, h1_residues_r, h2_residues = [], [], []
        for node in min_stem_1:
            h1_residues += node._metadata['pdb_residues']
            h1_residues_r += node._metadata['pdb_residues'][::-1]
        for node in min_stem_2:
            h2_residues += node._metadata['pdb_residues']

        h1_residues += [node._metadata['pdb_residues'][0] for node in min_loop_1]
        h1_residues_r += [node._metadata['pdb_residues'][0] for node in min_loop_1[::-1]]
        h2_residues += [node._metadata['pdb_residues'][0] for node in min_loop_2]

    else:
        h1_residues = closest_h1.subtree_residues
        h1_residues_r = closest_h1.subtree_residues
        h2_residues = closest_h2.subtree_residues

    # ---- Paths section
    longest_paths_1 = _longest_path(rna_tree_1)
    longest_paths_2 = _longest_path(rna_tree_2)

    min_dist = None
    selected_longest_path_1, selected_longest_path_2 = None, None
    for k, (longest_path_1, longest_path_2) in enumerate(product(longest_paths_1, longest_paths_2)):
        if len(longest_path_1) == len(longest_path_2):
            total_dist = 0
            for node_1, node_2 in zip(longest_path_1, longest_path_2):
                total_dist += dist[node_1.dist_id, node_2.dist_id]
            total_dist /= len(longest_path_2)
            if not min_dist or total_dist < min_dist:
                min_dist = total_dist
                selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2
        elif len(longest_paths_1[0]) > len(longest_paths_2[0]):
            len_diff = len(longest_paths_1[0]) - len(longest_paths_2[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1[offset:-(len_diff - offset)], longest_path_2):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_2)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1[offset:-(len_diff - offset)], longest_path_2
        else:
            len_diff = len(longest_paths_2[0]) - len(longest_paths_1[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1, longest_path_2[offset:-(len_diff - offset)]):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_1)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2[offset:-(len_diff - offset)]

    longest_path_1, longest_path_2 = selected_longest_path_1, selected_longest_path_2

    path_residues_1 = []
    path_residues_2 = []
    for i in range(2, max(len(longest_path_1), 3)):
        middle_idx = int(len(longest_path_1) / i)
        selected_residues_1 = [longest_path_1[0], longest_path_1[-1]]
        selected_residues_2 = [longest_path_2[0], longest_path_2[-1]]

        for j in range(1, i):
            new_residue_1 = longest_path_1[middle_idx * j]
            new_residue_2 = longest_path_2[middle_idx * j]
            if new_residue_1 in selected_residues_1 or new_residue_2 in selected_residues_2:
                continue
            selected_residues_1.append(new_residue_1)
            selected_residues_2.append(new_residue_2)

        path_residues_1.append([residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_1) for residue in residues])
        path_residues_2.append([residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_2) for residue in residues])
        path_residues_1.append([residue for residues in map(lambda node: node._metadata['pdb_residues'][::-1], selected_residues_1) for residue in residues])
        path_residues_2.append([residue for residues in map(lambda node: node._metadata['pdb_residues'], selected_residues_2) for residue in residues])

    # --- Root children-leaf section
    leafs_1 = rna_tree_1.root.children
    leafs_2 = rna_tree_2.root.children

    min_leaf_distance, min_leaf_1, min_leaf_2, = np.sum(dist) + 1, None, None
    if len(leafs_1) > len(leafs_2):
        len_diff = len(leafs_1) - len(leafs_2)
        for offset in range(len_diff):
            width_distance, next_leaf_1, next_leaf_2 = _max_node_list_distance(dist, leafs_1[offset:-(len_diff - offset)], leafs_2)
            if width_distance < min_leaf_distance or (width_distance == min_leaf_distance and len(next_leaf_1) > len(min_leaf_1)):
                min_leaf_distance, min_leaf_1, min_leaf_2 = width_distance, next_leaf_1, next_leaf_2
    elif len(leafs_1) < len(leafs_2):
        len_diff = len(leafs_2) - len(leafs_1)
        for offset in range(len_diff):
            width_distance, next_leaf_1, next_leaf_2 = _max_node_list_distance(dist, leafs_1, leafs_2[offset:-(len_diff - offset)])
            if width_distance < min_leaf_distance or (width_distance == min_leaf_distance and len(next_leaf_1) > len(min_leaf_1)):
                min_leaf_distance, min_leaf_1, min_leaf_2 = width_distance, next_leaf_1, next_leaf_2
    else:
        min_leaf_distance, min_leaf_1, min_leaf_2 = _max_node_list_distance(dist, leafs_1, leafs_2)

    # --- Compound section
    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    summed_path_residues_1 = []
    summed_path_residues_2 = []
    for path1, path2 in zip(path_residues_1, path_residues_2):
        for node1, node2 in zip(path1, path2):
            if node1 not in summed_path_residues_1 and node2 not in summed_path_residues_2:
                summed_path_residues_1.append(node1)
                summed_path_residues_2.append(node2)

    summed_path_residues_1_r = []
    summed_path_residues_2_r = []
    for path1, path2 in zip(path_residues_1[::-1], path_residues_2[::-1]):
        for node1, node2 in zip(path1, path2):
            if node1 not in summed_path_residues_1 and node2 not in summed_path_residues_2:
                summed_path_residues_1_r.append(node1)
                summed_path_residues_2_r.append(node2)

    summed_path_residues_1 = [[node] for node in summed_path_residues_1]
    summed_path_residues_2 = [[node] for node in summed_path_residues_2]
    summed_path_residues_1_r = [[node] for node in summed_path_residues_1_r]
    summed_path_residues_2_r = [[node] for node in summed_path_residues_2_r]

    h1_residues = [_ndarray_from_residues(h1_residues)]
    h1_residues_r = [_ndarray_from_residues(h1_residues_r)]
    h2_residues = [_ndarray_from_residues(h2_residues)]
    aligned_leafs_1 = _ndarray_from_residues([node._metadata['pdb_residues'][0] for node in min_leaf_1])
    aligned_leafs_2 = _ndarray_from_residues([node._metadata['pdb_residues'][0] for node in min_leaf_2])
    summed_path_residues_1 = [_ndarray_from_residues(l) for l in summed_path_residues_1]
    summed_path_residues_2 = [_ndarray_from_residues(l) for l in summed_path_residues_2]
    summed_path_residues_1_r = [_ndarray_from_residues(l) for l in summed_path_residues_1_r]
    summed_path_residues_2_r = [_ndarray_from_residues(l) for l in summed_path_residues_2_r]

    potential_orderings = [
        (h1_residues + [aligned_leafs_1] + summed_path_residues_1, h2_residues + [aligned_leafs_2] + summed_path_residues_2),
        (h1_residues + summed_path_residues_1 + [aligned_leafs_1], h2_residues + summed_path_residues_2 + [aligned_leafs_2]),
        ([aligned_leafs_1] + h1_residues + summed_path_residues_1, [aligned_leafs_2] + h2_residues + summed_path_residues_2),
        ([aligned_leafs_1] + summed_path_residues_1 + h1_residues, [aligned_leafs_2] + summed_path_residues_2 + h2_residues),
        (summed_path_residues_1 + h1_residues + [aligned_leafs_1], summed_path_residues_2 + h2_residues + [aligned_leafs_2]),
        (summed_path_residues_1 + [aligned_leafs_1] + h1_residues, summed_path_residues_2 + [aligned_leafs_2] + h2_residues),
        (h1_residues + [aligned_leafs_1] + summed_path_residues_1_r, h2_residues + [aligned_leafs_2] + summed_path_residues_2_r),
        (h1_residues + summed_path_residues_1_r + [aligned_leafs_1], h2_residues + summed_path_residues_2_r + [aligned_leafs_2]),
        ([aligned_leafs_1] + h1_residues + summed_path_residues_1_r, [aligned_leafs_2] + h2_residues + summed_path_residues_2_r),
        ([aligned_leafs_1] + summed_path_residues_1_r + h1_residues, [aligned_leafs_2] + summed_path_residues_2_r + h2_residues),
        (summed_path_residues_1_r + h1_residues + [aligned_leafs_1], summed_path_residues_2_r + h2_residues + [aligned_leafs_2]),
        (summed_path_residues_1_r + [aligned_leafs_1] + h1_residues, summed_path_residues_2_r + [aligned_leafs_2] + h2_residues),
        (h1_residues_r + [aligned_leafs_1] + summed_path_residues_1, h2_residues + [aligned_leafs_2] + summed_path_residues_2),
        (h1_residues_r + summed_path_residues_1 + [aligned_leafs_1], h2_residues + summed_path_residues_2 + [aligned_leafs_2]),
        ([aligned_leafs_1] + h1_residues_r + summed_path_residues_1, [aligned_leafs_2] + h2_residues + summed_path_residues_2),
        ([aligned_leafs_1] + summed_path_residues_1 + h1_residues_r, [aligned_leafs_2] + summed_path_residues_2 + h2_residues),
        (summed_path_residues_1 + h1_residues_r + [aligned_leafs_1], summed_path_residues_2 + h2_residues + [aligned_leafs_2]),
        (summed_path_residues_1 + [aligned_leafs_1] + h1_residues_r, summed_path_residues_2 + [aligned_leafs_2] + h2_residues),
        (h1_residues_r + [aligned_leafs_1] + summed_path_residues_1_r, h2_residues + [aligned_leafs_2] + summed_path_residues_2_r),
        (h1_residues_r + summed_path_residues_1_r + [aligned_leafs_1], h2_residues + summed_path_residues_2_r + [aligned_leafs_2]),
        ([aligned_leafs_1] + h1_residues_r + summed_path_residues_1_r, [aligned_leafs_2] + h2_residues + summed_path_residues_2_r),
        ([aligned_leafs_1] + summed_path_residues_1_r + h1_residues_r, [aligned_leafs_2] + summed_path_residues_2_r + h2_residues),
        (summed_path_residues_1_r + h1_residues_r + [aligned_leafs_1], summed_path_residues_2_r + h2_residues + [aligned_leafs_2]),
        (summed_path_residues_1_r + [aligned_leafs_1] + h1_residues_r, summed_path_residues_2_r + [aligned_leafs_2] + h2_residues),
    ]

    all_total_data = []
    for i, (ordered_aligned_nodes_1, ordered_aligned_nodes_2) in enumerate(potential_orderings):
        current_aligned_widths_1 = []
        current_aligned_widths_2 = []
        k = 0
        for nodes1, nodes2 in zip(ordered_aligned_nodes_1, ordered_aligned_nodes_2):
            k += 1

            width_1_coords = np.vstack(current_aligned_widths_1 + nodes1)
            width_2_coords = np.vstack(current_aligned_widths_2 + nodes2)
            total_rotran = calculate_rotran(width_1_coords, width_2_coords)
            total_rms, total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, total_rotran)
            if not final_total_rms or total_psi > final_total_psi or (np.isclose(total_psi, final_total_psi) and total_rms < final_total_rms):
                current_aligned_widths_1 += nodes1
                current_aligned_widths_2 += nodes2
                final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = total_rotran, total_rms, total_psi, smoothing_rotran

        all_total_data.append((final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran))
        final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = max(all_total_data, key=lambda t: t[2])

    if save_args:
        _save_structure(final_total_rotran, final_smoothing_rotran, save_args)

    return final_total_rms, final_total_psi, final_total_rotran[0], final_total_rotran[1]


def method_j(structure_1_coordinates, structure_2_coordinates, rna_tree_1, rna_tree_2, dist, save_args=None):
    # improved method i
    # best hairpin
    # longest path
    # root leaf-children
    # combine

    # ---- Hairpin section
    herpins_1 = rna_tree_1.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)
    herpins_2 = rna_tree_2.apply_subtree_predicate(herpin_init_predicate, herpin_result_predicate)

    closest_h1, closest_h2 = None, None
    min_dist = dist.max() + 1
    for h1, h2 in product(herpins_1, herpins_2):
        if dist[h1.dist_id, h2.dist_id] < min_dist:
            closest_h1, closest_h2 = h1, h2
            min_dist = dist[h1.dist_id, h2.dist_id]

    if min_dist > 0:

        if closest_h1.label == 'root':
            closest_h1 = closest_h1.children[0]

        if closest_h2.label == 'root':
            closest_h2 = closest_h2.children[0]

        h1_stem_residues, h2_stem_residues = [closest_h1], [closest_h2]
        current_node = closest_h1
        while True:
            child = current_node.children[0]
            if not child.is_leaf:
                h1_stem_residues.append(child)
                current_node = child
            else:
                break
        current_node = closest_h2
        while True:
            child = current_node.children[0]
            if not child.is_leaf:
                h2_stem_residues.append(child)
                current_node = child
            else:
                break

        h1_loop_residues = closest_h1.leaf_parent.children
        h2_loop_residues = closest_h2.leaf_parent.children

        min_stem_dist = None
        min_stem_1, min_stem_2, min_loop_1, min_loop_2 = None, None, None, None

        if len(h1_stem_residues) == len(h2_stem_residues):
            min_stem_1, min_stem_2 = h1_stem_residues, h2_stem_residues
        elif len(h1_stem_residues) > len(h2_stem_residues):
            len_diff = len(h1_stem_residues) - len(h2_stem_residues)
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(h1_stem_residues[offset:-(len_diff - offset)], h2_stem_residues):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(h2_stem_residues)
                if not min_stem_dist or total_dist < min_dist:
                    min_stem_dist = total_dist
                    min_stem_1, min_stem_2 = h1_stem_residues[offset:-(len_diff - offset)], h2_stem_residues
        else:
            len_diff = len(h2_stem_residues) - len(h1_stem_residues)
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(h1_stem_residues, h2_stem_residues[offset:-(len_diff - offset)]):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(h1_stem_residues)
                if not min_stem_dist or total_dist < min_dist:
                    min_stem_dist = total_dist
                    min_stem_1, min_stem_2 = h1_stem_residues, h2_stem_residues[offset:-(len_diff - offset)]

        min_loop_distance, min_loop_1, min_loop_2 = np.sum(dist) + 1, None, None
        if len(h1_loop_residues) > len(h2_loop_residues):
            len_diff = len(h1_loop_residues) - len(h2_loop_residues)
            for offset in range(len_diff):
                loop_distance, next_loop_1, next_loop_2 = _max_node_list_distance(dist, h1_loop_residues[offset:-(len_diff - offset)], h2_loop_residues)
                if loop_distance < min_loop_distance or (loop_distance == min_loop_distance and len(next_loop_1) > len(min_loop_1)):
                    min_loop_distance, min_loop_1, min_loop_2 = loop_distance, next_loop_1, next_loop_2
        elif len(h1_loop_residues) < len(h2_loop_residues):
            len_diff = len(h2_loop_residues) - len(h1_loop_residues)
            for offset in range(len_diff):
                loop_distance, next_loop_1, next_loop_2 = _max_node_list_distance(dist, h1_loop_residues, h2_loop_residues[offset:-(len_diff - offset)])
                if loop_distance < min_loop_distance or (loop_distance == min_loop_distance and len(next_loop_1) > len(min_loop_1)):
                    min_loop_distance, min_loop_1, min_loop_2 = loop_distance, next_loop_1, next_loop_2
        else:
            min_loop_distance, min_loop_1, min_loop_2 = _max_node_list_distance(dist, h1_loop_residues, h2_loop_residues)

        h1_residues, h1_residues_r, h2_residues = [], [], []
        for node in min_stem_1:
            h1_residues += node._metadata['pdb_residues']
            h1_residues_r += node._metadata['pdb_residues'][::-1]
        for node in min_stem_2:
            h2_residues += node._metadata['pdb_residues']

        h1_residues += [node._metadata['pdb_residues'][0] for node in min_loop_1]
        h1_residues_r += [node._metadata['pdb_residues'][0] for node in min_loop_1[::-1]]
        h2_residues += [node._metadata['pdb_residues'][0] for node in min_loop_2]

    else:
        h1_residues = closest_h1.subtree_residues
        h1_residues_r = closest_h1.subtree_residues
        h2_residues = closest_h2.subtree_residues

    # ---- Paths section
    longest_paths_1 = _longest_path(rna_tree_1)
    longest_paths_2 = _longest_path(rna_tree_2)

    min_dist = None
    selected_longest_path_1, selected_longest_path_2 = None, None
    for k, (longest_path_1, longest_path_2) in enumerate(product(longest_paths_1, longest_paths_2)):
        if len(longest_path_1) == len(longest_path_2):
            total_dist = 0
            for node_1, node_2 in zip(longest_path_1, longest_path_2):
                total_dist += dist[node_1.dist_id, node_2.dist_id]
            total_dist /= len(longest_path_2)
            if not min_dist or total_dist < min_dist:
                min_dist = total_dist
                selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2
        elif len(longest_paths_1[0]) > len(longest_paths_2[0]):
            len_diff = len(longest_paths_1[0]) - len(longest_paths_2[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1[offset:-(len_diff - offset)], longest_path_2):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_2)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1[offset:-(len_diff - offset)], longest_path_2
        else:
            len_diff = len(longest_paths_2[0]) - len(longest_paths_1[0])
            for offset in range(len_diff):
                total_dist = 0
                for node_1, node_2 in zip(longest_path_1, longest_path_2[offset:-(len_diff - offset)]):
                    total_dist += dist[node_1.dist_id, node_2.dist_id]
                total_dist /= len(longest_path_1)
                if not min_dist or total_dist < min_dist:
                    min_dist = total_dist
                    selected_longest_path_1, selected_longest_path_2 = longest_path_1, longest_path_2[offset:-(len_diff - offset)]

    longest_path_1, longest_path_2 = selected_longest_path_1, selected_longest_path_2

    min_dist = None
    selected_nodes_1, selected_nodes_2 = [], []
    for i in range(2, max(len(longest_path_1), 3)):
        middle_idx = int(len(longest_path_1) / i)
        nodes_1 = [longest_path_1[0], longest_path_1[-1]]
        nodes_2 = [longest_path_2[0], longest_path_2[-1]]

        for j in range(1, i):
            new_node_1 = longest_path_1[middle_idx * j]
            new_node_2 = longest_path_2[middle_idx * j]
            if new_node_1 in nodes_1 or new_node_2 in nodes_2:
                continue
            nodes_1.append(new_node_1)
            nodes_2.append(new_node_2)

        total_dist = 0
        for node_1, node_2 in zip(nodes_1, nodes_2):
            total_dist += dist[node_1.dist_id, node_2.dist_id]
        total_dist /= len(nodes_1)

        if not min_dist or min_dist > total_dist:
            min_dist = total_dist
            selected_nodes_1, selected_nodes_2 = nodes_1, nodes_2

    # --- Root children-leaf section
    leafs_1 = rna_tree_1.root.children
    leafs_2 = rna_tree_2.root.children

    min_leaf_distance, min_leaf_1, min_leaf_2, = np.sum(dist) + 1, None, None
    if len(leafs_1) > len(leafs_2):
        len_diff = len(leafs_1) - len(leafs_2)
        for offset in range(len_diff):
            width_distance, next_leaf_1, next_leaf_2 = _max_node_list_distance(dist, leafs_1[offset:-(len_diff - offset)], leafs_2)
            if width_distance < min_leaf_distance or (width_distance == min_leaf_distance and len(next_leaf_1) > len(min_leaf_1)):
                min_leaf_distance, min_leaf_1, min_leaf_2 = width_distance, next_leaf_1, next_leaf_2
    elif len(leafs_1) < len(leafs_2):
        len_diff = len(leafs_2) - len(leafs_1)
        for offset in range(len_diff):
            width_distance, next_leaf_1, next_leaf_2 = _max_node_list_distance(dist, leafs_1, leafs_2[offset:-(len_diff - offset)])
            if width_distance < min_leaf_distance or (width_distance == min_leaf_distance and len(next_leaf_1) > len(min_leaf_1)):
                min_leaf_distance, min_leaf_1, min_leaf_2 = width_distance, next_leaf_1, next_leaf_2
    else:
        min_leaf_distance, min_leaf_1, min_leaf_2 = _max_node_list_distance(dist, leafs_1, leafs_2)

    # --- Compound section
    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    h1_residues = [_ndarray_from_residues(h1_residues)]
    h1_residues_r = [_ndarray_from_residues(h1_residues_r)]
    h2_residues = [_ndarray_from_residues(h2_residues)]
    aligned_leafs_1 = [_ndarray_from_residues([node._metadata['pdb_residues'][0] for node in min_leaf_1])]
    aligned_leafs_2 = [_ndarray_from_residues([node._metadata['pdb_residues'][0] for node in min_leaf_2])]
    selected_nodes_1 = [_ndarray_from_residues([node._metadata['pdb_residues'][0] for node in selected_nodes_1])]
    selected_nodes_2 = [_ndarray_from_residues([node._metadata['pdb_residues'][0] for node in selected_nodes_2])]

    potential_orderings = [
        (h1_residues + aligned_leafs_1 + selected_nodes_1, h2_residues + aligned_leafs_2 + selected_nodes_2),
        (h1_residues_r + aligned_leafs_1 + selected_nodes_1, h2_residues + aligned_leafs_2 + selected_nodes_2),
        (h1_residues + selected_nodes_1, h2_residues + selected_nodes_2),
        (h1_residues_r + selected_nodes_1, h2_residues + selected_nodes_2),
        (aligned_leafs_1 + selected_nodes_1, aligned_leafs_2 + selected_nodes_2),
        (selected_nodes_1, selected_nodes_2)
    ]

    all_total_data = []
    for i, (ordered_aligned_nodes_1, ordered_aligned_nodes_2) in enumerate(potential_orderings):
        current_aligned_widths_1 = []
        current_aligned_widths_2 = []
        k = 0
        for nodes1, nodes2 in zip(ordered_aligned_nodes_1, ordered_aligned_nodes_2):
            k += 1

            width_1_coords = np.vstack(current_aligned_widths_1 + nodes1)
            width_2_coords = np.vstack(current_aligned_widths_2 + nodes2)
            total_rotran = calculate_rotran(width_1_coords, width_2_coords)
            total_rms, total_psi, smoothing_rotran = calculate_rms_with_rotran(structure_1_coordinates, structure_2_coordinates, total_rotran)
            if not final_total_rms or total_psi > final_total_psi or (np.isclose(total_psi, final_total_psi) and total_rms < final_total_rms):
                current_aligned_widths_1 += nodes1
                current_aligned_widths_2 += nodes2
                final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = total_rotran, total_rms, total_psi, smoothing_rotran

        all_total_data.append((final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran))
        final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = None, None, None, None

    final_total_rotran, final_total_rms, final_total_psi, final_smoothing_rotran = max(all_total_data, key=lambda t: t[2])

    if save_args:
        _save_structure(final_total_rotran, final_smoothing_rotran, save_args)

    return final_total_rms, final_total_psi, final_total_rotran[0], final_total_rotran[1]
