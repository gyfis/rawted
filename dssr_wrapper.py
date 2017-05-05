import subprocess
import json
from typing import Dict, Tuple, Iterable
from path_helpers import absolute_path, check_or_create_absolute_dir


_dssr_output_dir = 'dssr_output'


def _build_dssr_args(input_filename: str = None,
                     output_filename: str = None,
                     use_json: bool = True,
                     cleanup: bool = False) -> Tuple:
    args = absolute_path('bin/x3dna-dssr'),

    if cleanup:
        return args + ('--cleanup', )

    if use_json:
        args += '--json',

    if input_filename:
        args += 'i={}'.format(input_filename),

    if output_filename:
        args += 'o={}'.format(output_filename),

    return args


def _run_dssr(args: tuple) -> str:
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    return popen.stdout.read()


def _prepare_args_dssr(filename: str, output_filename: str = None) -> tuple:
    return _build_dssr_args(input_filename=filename, output_filename=output_filename)


def _load_input_file(input_file: str) -> Dict:
    output_file = absolute_dssr_path(input_file)
    _run_dssr(_prepare_args_dssr(input_file, output_filename=output_file))
    return json.load(open(output_file))


def clean_dssr_output():
    _run_dssr(_build_dssr_args(use_json=False, cleanup=True))


def _remove_pseudoknots(dbn: str) -> Tuple[str, Dict]:
    knot_starts = {}

    knot_pairings = {']': '[', '}': '{', '>': '<'}
    regulars = ['.', '(', ')']

    knot_pairs = {}

    knots_removed = ''

    for i, c in enumerate(dbn):
        if c in regulars:
            knots_removed += c

        if c in knot_pairings.values():
            if c not in knot_starts:
                knot_starts[c] = []
            knot_starts[c].append(i)

        if c in knot_pairings:
            if knot_pairings[c] not in knot_starts:
                raise Exception('String does not have valid bpn format')
            if c not in knot_pairs:
                knot_pairs[c] = []
            knot_pairs[c].append((knot_starts[knot_pairings[c]][-1], i))
            knot_starts[knot_pairings[c]] = knot_starts[knot_pairings[c]][:-1]

    return knots_removed, knot_pairs


def absolute_dssr_path(input_file: str) -> str:
    check_or_create_absolute_dir(_dssr_output_dir)
    return absolute_path(_dssr_output_dir, input_file.split('/')[-1], '_', 'output.pdb.json')


class DSSRWrapper:

    @classmethod
    def load(cls, filename: str):
        return DSSRWrapper(filename)

    def __init__(self, input_file: str):
        self._dssr_output = _load_input_file(input_file)

        self._dbn_wo_pseudoknots, self._pseudoknots = _remove_pseudoknots(self._get_dbn_for_chain())

    def _get_pairs(self, names: Iterable):
        return [p for p in self._dssr_output['pairs'] if p['name'] in names]

    def _get_pairs_inverse(self, names: Iterable):
        return [p for p in self._dssr_output['pairs'] if p['name'] not in names]

    def _get_dbn_for_chain(self, chain: str = None):
        return self._dssr_output['dbn'][chain if chain else 'all_chains']['sstr']

    @property
    def wc_pairs(self):
        return self._get_pairs('WC', )

    @property
    def non_wc_pairs(self):
        return self._get_pairs_inverse('WC', )

    @property
    def dbn(self):
        return self._get_dbn_for_chain()

    @property
    def nts(self):
        return self._dssr_output['nts']

    @property
    def dbn_cleaned(self):
        return self._dbn_wo_pseudoknots

    @property
    def pseudoknots(self):
        return self._pseudoknots

    def nts_for_chain(self, chain_id: str):
        return [nt for nt in self.nts if nt['chain_name'] == chain_id]
