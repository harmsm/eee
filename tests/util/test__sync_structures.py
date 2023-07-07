
import pytest

from eee.util import load_structure
from eee.util.data import AA_1TO3

from eee.util._sync_structures import _clean_structures
from eee.util._sync_structures import _run_muscle
from eee.util._sync_structures import _align_seq
from eee.util._sync_structures import _check_residues
from eee.util._sync_structures import _align_structures
from eee.util._sync_structures import sync_structures

import numpy as np

import os
import glob

def test__clean_structures():
    pass

def test__run_muscle():
    pass

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

        aligned_dfs = _align_seq(dfs,keep_temporary=True)

        # Make sure write output is being created
        assert len(aligned_dfs) == len(dfs)
        all_rows = np.sum([len(out_df) for out_df in aligned_dfs])
        for out_df in aligned_dfs:
            assert "alignment_site" in out_df.columns
            assert "shared_fx" in out_df.columns
            assert np.min(out_df["shared_fx"]) >= 0
            assert np.max(out_df["shared_fx"]) <= 1
            assert np.min(out_df["alignment_site"]) >= 0
            assert np.max(out_df["alignment_site"]) <= all_rows

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
        all_indexes = []
        for out_df in aligned_dfs:
            all_indexes.extend(out_df["alignment_site"])

        assert lengths[0] - 1 == np.nanmax(all_indexes)

        # make sure muscle_binary is interpreted correctly
        with pytest.raises(FileNotFoundError):
            _align_seq(dfs,muscle_binary="not_real")

        # Delete temporary files
        for f in glob.glob("*.fasta"):
            os.remove(f)

        # Make sure keep_temporary flag is interpreted correctly
        #column_contents, column_indexes 
        aligned_dfs= _align_seq(dfs,keep_temporary=False)
        assert len(glob.glob("*.fasta")) == 0

    os.chdir(current_dir)

def test__check_residues():
    pass

def test__align_structures(ensembles,tmpdir):
    pass

def test_sync_structures():
    pass