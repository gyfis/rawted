from Bio.PDB import Structure, PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy.spatial import KDTree
import numpy as np
import warnings
import math
from typing import Tuple


warnings.simplefilter('ignore', PDBConstructionWarning)


def calculate_rms(atoms1: np.ndarray, atoms2: np.ndarray) -> float:

    si = SVDSuperimposer()
    si.set(atoms1, atoms2)
    si.run()

    return si.get_rms()


def calculate_rotran(atoms1: np.ndarray, atoms2: np.ndarray) -> tuple:

    si = SVDSuperimposer()
    si.set(atoms1, atoms2)
    si.run()

    return si.get_rotran()


def calculate_rms_with_rotran(atoms1: np.ndarray, atoms2: np.ndarray, rotran: tuple) -> Tuple[float, float, Tuple]:
    # apply rotran on atoms2
    atoms2 = np.dot(atoms2, rotran[0]) + rotran[1]

    # fix atoms1 in place, use KDTree for distances
    kd_tree_a = KDTree(atoms1)
    kd_tree_b = KDTree(atoms2)
    # get nearest neighbours for atoms2 in atoms1 using KDTree, and also the other way around
    distances, indexes_a = kd_tree_b.query(atoms1)
    _, indexes_b = kd_tree_a.query(atoms2)

    # pick only the pairs that are both closest to each other
    closest_indexes = np.where(indexes_b[indexes_a] == list(range(len(indexes_a))))
    smoothing_rotran = calculate_rotran(atoms1[closest_indexes], atoms2[indexes_a[closest_indexes]])
    atoms2 = np.dot(atoms2, smoothing_rotran[0]) + smoothing_rotran[1]

    kd_tree_b = KDTree(atoms2)
    distances, indexes_a = kd_tree_b.query(atoms1)
    _, indexes_b = kd_tree_a.query(atoms2)

    # pick only the pairs that are both closest to each other
    closest_indexes = np.where(indexes_b[indexes_a] == list(range(len(indexes_a))))
    distances = distances[closest_indexes]

    # return RMSD ( sqrt(1/N * SUM(distance(xi, yi)^2)) )
    post_rms = math.sqrt(np.sum(np.power(distances, 2)) / float(len(distances)))

    psi = 100.0 * np.sum(distances <= 4.0) / float(min(len(atoms1), len(atoms2)))

    return post_rms, psi, smoothing_rotran


class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.id == self.chain:
            return 1
        return 0


def save_structure(structure, chain, filename):
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename, ChainSelect(chain))


def load_pdb_structure(filename: str) -> Structure:
    return PDBParser().get_structure('', open(filename))


def get_chains(pdb_structure: Structure) -> list:
    return list(chain.id for chain in pdb_structure[0].child_list)
