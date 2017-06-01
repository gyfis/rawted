# RAWTED
This repository holds a source code for the Rna Alignment With Tree Edit Distance method developed as a Master Thesis in 2016-2017 by me (Tomas Hromada).

## How to use
Usage is very simple.
1. Download source code
2. Install requirements (numpy, scipy, BioPython) `pip3 install -r requirements.txt`
3. Download DSSR tool (x3dna-dssr binary) from https://x3dna.org , and put into `bin/` folder at the root of the project
4. Run `python3 rawted.py [args]`

## Arguments of the program
- `filename of .pdb tertiary RNA structure file 1`
- `chain 1 (optional)`
- `filename of .pdb tertiary RNA structure file 2`
- `chain 2 (optional)`
- `--trees, -t [tree arguments] (optional)` - specifies what kind of trees to use for the alignment
- `--methods, -m [method arguments] (optional)` - specifies which methods to use for the alignment
- `--save_folder, -s (optional)` - save folder for the aligned structure, saves only when specified

Note that when the chains are not specified, any chain from the structure can be taken for the alignment.
The default tree variant is `v1`, the default method is `J`

### Trees
There are two major version of trees, `v1` and `v2`, with `v2` using three or four optional numeric arguments for weights and penalty

### Methods
There are 10 methods usable with the RAWTED, identified by the alphabet letters `A` to `J`

## Call examples

`python3 rawted.py structure1.pdb structure2.pdb --trees v1 v2 v2,10,0,0 v2,10,1,1,100 --methods A B D E H I J`

`python3 rawted.py structure1.pdb A structure2.pdb C --t v2,10,0,0 -m A B D -s out_dir`

The PDB files can be found and downloaded at the RCSB Protein Data Bank.

## Known issues

When using the argument `--save_folder, -s`, some structures fail to save correctly and instead throw an exception.
This is caused by the presence of disordered atoms in the structure and the usage of BioPython.
The issue is tracked in https://github.com/biopython/biopython/issues/455.
