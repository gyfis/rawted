from Bio.PDB import Chain
import numpy as np
import re


def enum(*sequential, **named):
    return type('Enum', (), dict(zip(sequential, range(len(sequential))), **named))


def bracket_pairs(bracket):
    brackets = {'(': ')', '[': ']', '<': '>', '{': '}',
                ')': '(', ']': '[', '>': '<', '}': '{'}
    return brackets[bracket]


def pdb_id_from_nt(nt):
    chainless_nt_id = nt['nt_id'][nt['nt_id'].find('.') + 1:]
    split_nt_id = (chainless_nt_id.split('^') + [' '])
    nt_id = re.search('[0-9]+$', split_nt_id[0]).group()
    nt_type = split_nt_id[0][:-len(nt_id)].strip('/') or ' '
    return nt_type, int(nt_id), split_nt_id[1]


def pdb_residues(pdb_chain, nt):
    nt_id = pdb_id_from_nt(nt)
    for residue in pdb_chain:
        r_id = residue.id
        resname = residue.resname.strip()
        if nt_id == (resname, r_id[1], r_id[2]):
            return residue
    raise LookupError('residue not found')


def coordinates_from_pdb_chain(pdb_chain: Chain) -> np.ndarray:
    return np.array([residue.child_dict['P'].coord for residue in pdb_chain if 'P' in residue.child_dict])
