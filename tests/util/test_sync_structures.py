
import pytest

from eee.util import load_structure
from eee.util.data import AA_1TO3

from eee.util.sync_structures import _align_seq
from eee.util.sync_structures import _align_structures
from eee.util.sync_structures import sync_structures

import numpy as np

import os
import glob

def test__align_seq(ensembles,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    for k in ensembles:
        
        # Load the specified structure files
        dfs = []
        for f in ensembles[k]:
            dfs.append(load_structure(f))
        
        # Get lists of all CA atoms and residues
        seqs = []
        for df in dfs:
            mask = np.logical_and(df.atom == "CA",df["class"] == "ATOM")
            this_df = df.loc[mask,:]
            seqs.append(list(this_df["resid"]))

        column_contents, column_indexes = _align_seq(seqs,keep_temporary=True)

        # Make sure write output is being created
        assert max([len(c) for c in column_contents]) == len(ensembles[k])
        assert len(column_indexes) == len(ensembles[k])
        assert len(glob.glob("*.fasta")) == 2
        
        # Make sure the input fasta is correct
        input_fasta = glob.glob("tmp-align*input.fasta")[0]
        input_seqs = []
        with open(input_fasta) as f:
            for line in f:
                if line.startswith(">"):
                    input_seqs.append([])
                else:
                    input_seqs[-1].extend(list(line.strip()))
        
        assert len(input_seqs) == len(ensembles[k])
        for i in range(len(input_seqs)):
            assert np.array_equal([AA_1TO3[s] for s in input_seqs[i]],
                                seqs[i])

        # Make sure the output fasta is reasonable
        output_fasta = glob.glob("tmp-align*output.fasta")[0]
        output_seqs = []
        with open(output_fasta) as f:
            for line in f:
                if line.startswith(">"):
                    output_seqs.append([])
                else:
                    output_seqs[-1].extend(list(line.strip()))

        # Only one length allowed; same as length of column_contents
        lengths = np.unique([len(s) for s in output_seqs])
        assert len(lengths) == 1
        assert lengths[0] == len(column_contents)

        # make sure muscle_binary is interpreted correctly
        with pytest.raises(FileNotFoundError):
            _align_seq(seqs,muscle_binary="not_real")

        # Delete temporary files
        for f in glob.glob("*.fasta"):
            os.remove(f)

        # Make sure keep_temporary flag is interpreted correctly
        column_contents, column_indexes = _align_seq(seqs,keep_temporary=False)
        assert len(glob.glob("*.fasta")) == 0

    os.chdir(current_dir)


def test__align_structures(ensembles,tmpdir):

    current_dir = os.getcwd()
    os.chdir(tmpdir)

    for k in ensembles:

        # Load the specified structure files
        dfs = []
        for f in ensembles[k]:
            dfs.append(load_structure(f))

        
    os.chdir(current_dir)