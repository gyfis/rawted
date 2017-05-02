import argparse
import re
from biopython_wrapper import load_pdb_structure, get_chains
from dssr_wrapper import DSSRWrapper, clean_dssr_output
from general_helpers import coordinates_from_pdb_chain
from methods import run_method
from rna_tree import tree_from_version
from zss_wrapper import zss_with_descriptor


def validate_tree_args(trees: list):
    tree_regex = re.compile('(v1|v2(\d+,\d+,\d+(,\d+)?)?)')

    for tree in trees:
        if not tree_regex.match(tree):
            print('Input argument tree {} does not match required format. Aborting...'.format(tree))
            exit()


def validate_method_args(methods: list):
    for method in methods:
        if method.upper() not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            print('Input argument method {} does not match any known method. Aborting...'.format(method))
            exit()


def validate_args(args):
    validate_tree_args(args.trees)
    validate_method_args(args.methods)


def run_rawted(args):
    trees = args.trees
    methods = args.methods

    structure_file1 = args.structure1
    chain1 = args.chain1
    structure_file2 = args.structure2
    chain2 = args.chain2

    save_folder = args.save_folder
    pdb_structure1 = load_pdb_structure(structure_file1)
    pdb_structure2 = load_pdb_structure(structure_file2)

    dssr1 = DSSRWrapper.load(structure_file1)
    dssr2 = DSSRWrapper.load(structure_file2)
    clean_dssr_output()

    if not chain1:
        chain1 = get_chains(pdb_structure2)[0]
    else:
        chains = get_chains(pdb_structure1)
        if chain1 not in chains:
            print('Input argument chain1 {} does not match any chains in the input structure ({}). Aborting...'.format(
                chain1, chains)
            )
            exit()

    if not chain2:
        chain2 = get_chains(pdb_structure2)[0]
    else:
        chains = get_chains(pdb_structure2)
        if chain2 not in chains:
            print('Input argument chain2 {} does not match any chains in the input structure ({}). Aborting...'.format(
                chain2, chains)
            )
            exit()

    pdb_chain1 = pdb_structure1[0][chain1]
    pdb_chain2 = pdb_structure2[0][chain2]
    dssr_nts1 = dssr1.nts_for_chain(chain1)
    dssr_nts2 = dssr2.nts_for_chain(chain2)

    for tree_descriptor in trees:
        tree_version = tree_descriptor.split(',')[0]
        tree1 = tree_from_version(tree_version).from_nts(dssr_nts1, pdb_chain1)
        tree2 = tree_from_version(tree_version).from_nts(dssr_nts2, pdb_chain2)

        ted_matrix = zss_with_descriptor(tree1, tree2, tree_descriptor)

        for method in methods:
            save_args = None
            if save_folder:
                save_args = {
                    'structure1': pdb_structure1.copy(),
                    'structure2': pdb_structure2.copy(),
                    'chain1': chain1,
                    'chain2': chain2,
                    'filename1': '{}/{}_{}_s1.pdb'.format(save_folder, tree_descriptor.replace(',', '-'), method.lower()),
                    'filename2': '{}/{}_{}_s2.pdb'.format(save_folder, tree_descriptor.replace(',', '-'), method.lower())
                }

            rmsd, psi, _, _ = run_method(
                method,
                coordinates_from_pdb_chain(pdb_chain1), coordinates_from_pdb_chain(pdb_chain2),
                tree1, tree2,
                ted_matrix,
                save_args=save_args
            )
            print('--------------------')
            print('Rawted finished.')
            print('Structure 1: {}'.format(structure_file1))
            print('Structure 2: {}'.format(structure_file2))
            print('Tree descriptor: {}'.format(tree_descriptor))
            print('Method: {}'.format(method))
            print('--------------------')
            print('>> Results <<')
            print('RMSD: {}'.format(round(rmsd, 4)))
            print('PSI: {}'.format(round(psi, 4)))
            print('--------------------')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('structure1', type=str, help='Structure 1 to process')
    parser.add_argument('chain1', nargs='?', type=str, help='Chain of structure 1 to process')
    parser.add_argument('structure2', type=str, help='Structure 2 to process')
    parser.add_argument('chain2', nargs='?', type=str, help='Chain of structure 2 to process')
    parser.add_argument('--trees', '-t', nargs='*', type=str, default=['v1'], help='Tree arguments')
    parser.add_argument('--methods', '-m', nargs='*', type=str, default=['J'], help='RAWTED methods to use')
    parser.add_argument('--save_folder', '-s', nargs='?', type=str, help='Save folder')

    args = parser.parse_args()

    validate_args(args)

    run_rawted(args)


if __name__ == '__main__':
    main()
