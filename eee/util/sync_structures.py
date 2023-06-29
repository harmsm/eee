"""
"""

from eee.util.data import AA_3TO1
from eee.util import write_pdb
from eee.util import load_structure

import numpy as np

import os
import random
import string
import subprocess
import re


def _align_seq(seq_list,muscle_binary="muscle",verbose=False):
    """
    Use muscle to align sequences from rcsb files. Returns which sequences align
    with which column and the column indexes for each site in each sequence.
    The alignment:
    
    MAST-
    -ASTP
    
    Would yield: 
    + column_contents: [[0],[0,1],[0,1],[0,1],[1]]
    + column_indexes: [[0,1,2,3],[1,2,3,4]]
    
    Parameters
    ----------
    seq_list : list
        list of lists. each sub-list should be an amino acid sequence using 
        3-letter amino acid codes. Example: ["MET","ALA",...]
    muscle_binary : str, default="muscle"
        path to muscle binary
    verbose : bool, default=True
        whether or not to print out muscle output
        
    Returns
    -------
    column_contents : list
        list of lists. list containing which sequences have a non-gap character
        at this position. 
    column_indexes : list
        list of lists. indexes correspond to input list. sub-lists are the 
        alignment columns for the amino acids input to seq_list.
    """

    # Construct temporary files
    r = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    base = f"tmp_{r}"
    input_fasta = f"tmp-align_{base}_input.fasta"
    output_fasta = f"tmp-align_{base}_output.fasta"
    
    # Write input fasta
    with open(input_fasta,'w') as f:
        for i, s in enumerate(seq_list):
            seq_string = "".join([AA_3TO1[aa] for aa in s])
            f.write(f">seq{i}\n{seq_string}\n")
        
    # Try both versions of muscle command line (v. 5, v. 3)
    cmd_list = [[muscle_binary,"-align",input_fasta,"-output",output_fasta],
                [muscle_binary,"-in",input_fasta,"-out",output_fasta]]
    
    # Run muscle
    successful = False
    for cmd in cmd_list:

        # Run muscle, capturing output to avoid cluttering terminal
        popen = subprocess.Popen(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True)
        for line in popen.stdout:
            if verbose:
                print(line,end="",flush=True)
    
        # If successful, stop trying to align
        return_code = popen.wait()
        if return_code == 0:
            successful = True
            break
    
    # If we did not actually do alignment, throw error
    if not successful:
        err = f"Alignment failed. Is muscle in the path?\n"
        raise RuntimeError(err)

    # Read output fasta file
    output = {}
    with open(output_fasta) as f:
        
        buffer = []
        for line in f:
            if line.startswith(">"):
                if len(buffer) != 0:
                    output[key] = buffer[:]
                    buffer = []

                key = int(line[4:])
            else:
                buffer.extend(list(line.strip()))
    
        output[key] = buffer[:]

    # Delete temporary files
    os.remove(input_fasta)
    os.remove(output_fasta)
        
    # Get alignment columns for each site
    keys = list(output.keys())
    keys.sort()
    
    column_indexes = [[] for _ in range(len(keys))]
    column_contents = [[] for _ in range(len(output[keys[0]]))]
    for k in keys:
        
        counter = 0
        alignment = output[key]
        
        for s in seq_list[k]:
            aa = AA_3TO1[s]
            while aa != alignment[counter]:
                counter += 1
            column_indexes[k].append(counter)
            
        for i, c in enumerate(alignment):
            if c != "-":
                column_contents[i].append(k)
            
    return column_contents, column_indexes
    

def _align_structures(dfs,lovoalign_binary="lovoalign",verbose=True):
    """
    Align the structures in the dataframes using lovoalign. Align all structures
    individually to the first dataframe.
    
    Parameters
    ----------
    dfs : list
        list of pandas dataframes containing structures
    lovoalign_binary : default = "lovoalign"
        lovoalign binary
    verbose : bool, defualt = True
        whether or not to print lovoalign output
    
    Returns
    -------
    dfs : list
        list of pandas dataframes containing structures with x, y, z coordinates
        updated
    """
    
    # No structures to align
    if len(dfs) < 2:
        return dfs
    
    # root for temporary files
    out_base = "".join([random.choice(string.ascii_letters) for _ in range(10)])

    # Write out pdb files to align
    files = []
    for i, df in enumerate(dfs):
        files.append(f"tmp-align_{i}_{out_base}.pdb")
        write_pdb(df,files[-1])
        
    # Go through all but first file
    for i, f in enumerate(files[1:]):
        
        # Align file to first file using lovoalign
        out_file = f"tmp-out_{out_base}.pdb"
        cmd = [lovoalign_binary,"-p1",f,"-p2",files[0],"-o",out_file]
        
        popen = subprocess.Popen(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True)
        for line in popen.stdout:
            if verbose:
                print(line,end="",flush=True)
        
        # # Check for success
        return_code = popen.wait()
        if return_code != 0:
            err = "Could not align structures!\n"
            raise RuntimeError(err)
        
        # Move coordinates from aligned structure into dfs
        new_df = load_structure(out_file)
        dfs[i+1].loc[:,"x"] = new_df["x"]
        dfs[i+1].loc[:,"y"] = new_df["y"]
        dfs[i+1].loc[:,"z"] = new_df["z"]
        
        # Remove output file
        #os.remove(out_file)
        
    # Remove all temporary files
    #for f in files:
    #    os.remove(f)
        
    return dfs
    

def sync_structures(out_dir,*args):
    
    # See if the output directory exists
    if os.path.exists(out_dir):
        err = f"output directory {out_dir} already exists.\n"
        raise FileExistsError(err)
        
    # Load the specified structure files
    pdb_files = args[:]
    dfs = []
    for f in pdb_files:
        dfs.append(load_structure(f))
    
    # Get lists of all CA atoms and residues
    seqs = []
    residues = []
    for df in dfs:
        df["_resid_key"] = list(zip(df["chain"],df["resid"],df["resid_num"]))
        
        mask = np.logical_and(df.atom == "CA",df["class"] == "ATOM")
        this_df = df.loc[mask,:]
        
        seqs.append(list(this_df["resid"]))
        residues.append(list(this_df["_resid_key"]))

    # Align the sequences using muscle
    column_contents, column_indexes = _align_seq(seqs)
    
    # Create an array indicating whether a residue is shared across all 
    # structures (column_mask)
    column_mask = np.zeros(len(column_contents),dtype=bool)
    num_structures = len(dfs)
    for i in range(len(column_contents)):
        if len(column_contents[i]) == num_structures:
            column_mask[i] = True
        
    # Go through the alignment for each structure
    for i in range(len(column_indexes)):
        
        this_df = dfs[i]
        this_df["matched"] = False
        
        for j in range(len(column_indexes[i])):
 
            is_matched = column_mask[j]
            
            # Record whether each residue is matched between the two structures
            this_resid = residues[i][j]
            this_resid_mask = this_df["_resid_key"] == this_resid
            this_df.loc[this_resid_mask,"matched"] = is_matched
                    
    # Remove "_resid_key" convenience column
    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop(columns="_resid_key")
        
    # Map "matched" to the b-factor column
    for i in range(len(dfs)):
        bfactors = np.zeros(len(dfs[i]),dtype=float)
        bfactors[dfs[i]["matched"]] = 1
        dfs[i].loc[:,"b"] = bfactors
    
    
    # Align structures in 3D
    _align_structures(dfs)
    
    os.mkdir(out_dir)
    
    for i in range(len(dfs)):
        
        # Make output file names have path to original files as names, replacing
        # path separators with __ and appending _clean.pdb. This is to make sure
        # all output file names are unique and can be mapped back to the input 
        # file names. test/this/out.pdb will be placed in 
        # out_dir/test__this__out_clean.pdb
        f = re.sub(os.sep,"__",pdb_files[i])
        f = f"{f}_clean.pdb"
        f = os.path.join(out_dir,f)
        
        write_pdb(dfs[i],f)
    

    return dfs
    