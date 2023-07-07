"""
"""

from eee.util.data import AA_3TO1
from eee.util import write_pdb
from eee.util import load_structure

import numpy as np
import pandas as pd

import os
import random
import string
import subprocess
import re
import glob
import shutil

def _clean_structures(dfs,foldx_binary="foldx",verbose=False):
    """
    Run the structures through foldx to build missing atoms in sidechains. Will
    delete residues with incomplete backbones. 
    """

    new_dfs = []

    out_base = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    tmp_dir = f"tmp-dir_{out_base}"
    os.mkdir(tmp_dir)
    os.chdir(tmp_dir)
    for df in dfs:
        write_pdb(df,"tmp-input.pdb",overwrite=True)

        cmd = [foldx_binary,
               "-c","PDBFile",
               "--fixSideChains","1",
               "--pdb","tmp-input.pdb"]
        
        popen = subprocess.Popen(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True)
        for line in popen.stdout:
            if verbose:
                print(line,end="",flush=True)
        
        # Check for success
        return_code = popen.wait()
        if return_code != 0:
            os.chdir("..")
            err = "Could not clean structure using foldx!\n"
            raise RuntimeError(err)
        
        shutil.move("PF_tmp-input.fxout","tmp-output.pdb")
        new_df = load_structure("tmp-output.pdb")

        # foldx will drop all hetatms. bring them back in
        hetatm_df = df.loc[df["class"] == "HETATM",:]
        new_df = pd.concat((new_df,hetatm_df))
        new_df.index = np.arange(len(new_df.index),dtype=int)

        new_dfs.append(new_df)

    os.chdir("..")

    return new_dfs

def _run_muscle(seq_list,
                muscle_binary="muscle",
                verbose=False,
                keep_temporary=False):

    # Construct temporary files
    r = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    base = f"tmp_{r}"
    input_fasta = f"tmp-align_{base}_input.fasta"
    output_fasta = f"tmp-align_{base}_output.fasta"

    # Write input fasta
    with open(input_fasta,'w') as f:
        for i, s in enumerate(seq_list):
            f.write(f">seq{i}\n{''.join(s)}\n")

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
        err = f"Alignment failed!\n"
        raise RuntimeError(err)

    # Read output fasta file
    output = {}
    with open(output_fasta) as f:
        
        for line in f:
            if line.startswith(">"):
                key = int(line[4:])
                output[key] = []
            else:
                output[key].extend(list(line.strip()))

    # Delete temporary files
    if not keep_temporary:
        os.remove(input_fasta)
        os.remove(output_fasta)

    # Get alignment columns for each site
    keys = list(output.keys())
    keys.sort()

    return [output[k] for k in keys]


def _align_seq(dfs,
               muscle_binary="muscle",
               verbose=False,
               keep_temporary=False):
    """
    Use muscle to align sequences from rcsb files. 
    
    Parameters
    ----------
    dfs : list
        list of pandas dataframes containing structures
    muscle_binary : str, default="muscle"
        path to muscle binary
    verbose : bool, default=True
        whether or not to print out muscle output
    keep_temporary : bool, default=False
        do not delete temporary files
        
    Returns
    -------
    dfs : list
        list of pandas dataframes with structures updated to have the shared_fx
        and alignment_site columns. 
    """

    seq_list = []
    for df in dfs:
        mask = np.logical_and(df.atom == "CA",df["class"] == "ATOM")
        this_df = df.loc[mask,:]        
        seq_list.append([AA_3TO1[aa] for aa in this_df["resid"]])

    output = _run_muscle(seq_list=seq_list,
                         muscle_binary=muscle_binary,
                         verbose=verbose,
                         keep_temporary=keep_temporary)
    

    # Convert output into column indexes and column contents. For example: 
    # MAST-
    # -ASTP
    # Would yield: 
    # + column_contents: [[0],[0,1],[0,1],[0,1],[1]]
    # + column_indexes: [[0,1,2,3],[1,2,3,4]]
    column_indexes = [[] for _ in range(len(output))]
    column_contents = [[] for _ in range(len(output[0]))]
    for i in range(len(output)):
        
        counter = 0
        alignment = output[i]
    
        for s in seq_list[i]:
            while s != alignment[counter]:
                counter += 1
            column_indexes[i].append(counter)
            
        for j, c in enumerate(alignment):
            if c != "-":
                column_contents[j].append(i)
            

    # Get lists of all CA atoms and residues
    residues = []
    for df in dfs:
        df["_resid_key"] = list(zip(df["chain"],df["resid"],df["resid_num"]))
        
        mask = np.logical_and(df.atom == "CA",df["class"] == "ATOM")
        this_df = df.loc[mask,:]

        residues.append(list(this_df["_resid_key"]))

    # Create an array indicating the fraction of structures sharing the site
    shared_column = np.zeros(len(column_contents),dtype=float)
    num_structures = len(dfs)
    for i in range(len(column_contents)):
        shared_column[i] = len(column_contents[i])/num_structures

    # Go through the alignment for each structure
    for i in range(len(column_indexes)):
        
        this_df = dfs[i]
        this_df["shared_fx"] = 0.0
        
        for j in range(len(column_indexes[i])):
            
            idx = column_indexes[i][j]

            # Record how many structures share the column
            this_resid = residues[i][j]
            this_resid_mask = this_df["_resid_key"] == this_resid
            this_df.loc[this_resid_mask,"alignment_site"] = idx
            this_df.loc[this_resid_mask,"shared_fx"] = shared_column[idx]
                    
    # Remove "_resid_key" convenience column
    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop(columns="_resid_key")
        
    return dfs


def _check_residues(df):
    """
    Check identical amino acids. 
    If Cys and Ser --> mutate Ser to Cys
    Check for missing backbone residues
    """

    # Get lists of all CA atoms and residues
    residues = []
    for df in dfs:
        df["_resid_key"] = list(zip(df["chain"],df["resid"],df["resid_num"]))
        
        mask = np.logical_and(df.atom == "CA",df["class"] == "ATOM")
        this_df = df.loc[mask,:]

        residues.append(list(this_df["_resid_key"]))


    pass


def _align_structures(dfs,
                      lovoalign_binary="lovoalign",
                      verbose=False,
                      keep_temporary=False):
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
    keep_temporary : bool, default=False
        do not delete temporary files
    
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
        out_file = f"tmp-out_{out_base}-{i+1}.pdb"
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

        dfs[i+1].loc[:,"x"] = np.array(new_df["x"])
        dfs[i+1].loc[:,"y"] = np.array(new_df["y"])
        dfs[i+1].loc[:,"z"] = np.array(new_df["z"])

        # Remove output file
        if not keep_temporary:
            os.remove(out_file)
        
    # Remove all temporary files
    if not keep_temporary:
        for f in files:
            os.remove(f)
        
    return dfs


        



def sync_structures(structure_files,
                    out_dir,
                    allowed_hetam=None,
                    overwrite=False,
                    verbose=False):
    """
    Take a set of structures, clean up, align, and figure out which sites are
    shared among all structures. Output is a directory with pdb files and a
    report describing structures. The b-factor column of each pdb file has the
    number of structures in which that specific site is seen. (HETATM b-factor 
    is always 0). 

    Parameters
    ----------
    structure_files : list
        list of structure files to use for the calculation. These files should 
        be in RCSB cif (preferred) or pdb format.
    out_dir : str
        output directory for the cleaned up files in pdb format. This directory
        should either not exist or be empty. 
    overwrite : bool, default=False
        overwrite an existing output directory
    verbose : bool, default=False
        write out all output to standard output
    """
    
    # See if the output directory exists
    exists = False
    if os.path.exists(out_dir):
        if os.path.isdir(out_dir):
            if len(glob.glob(os.path.join(out_dir,"*"))) > 0:
                exists = True
        else:
            exists = True

    if exists:
        if not overwrite:
            err = f"output directory {out_dir} already exists.\n"
            raise FileExistsError(err)
        else:
            shutil.rmtree(out_dir)
    
    # Make new directory. 
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    # Load the specified structure files
    dfs = []
    for f in structure_files:
        dfs.append(load_structure(f))

    # Clean up structures --> build missing atoms or delete residues with
    # missing backbone atoms. 
    dfs = _clean_structures(dfs,verbose=verbose)

    # Figure out which residues are shared between what structures
    dfs = _align_seq(dfs,verbose=verbose)

    

    # Align structures in 3D
    dfs = _align_structures(dfs,verbose=verbose)

    for i in range(len(dfs)):
        
        # Make output file names have path to original files as names, replacing
        # path separators with __ and appending _clean.pdb. This is to make sure
        # all output file names are unique and can be mapped back to the input 
        # file names. test/this/out.pdb will be placed in 
        # out_dir/test__this__out_clean.pdb
        f = re.sub(os.sep,"__",structure_files[i])
        f = f"{f}_clean.pdb"
        f = os.path.join(out_dir,f)
        
        write_pdb(dfs[i],
                  f,
                  bfactor_column="shared_fx",
                  occ_column="alignment_site")
    

    return dfs
    