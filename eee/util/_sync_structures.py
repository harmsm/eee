"""
Functions for taking raw RCSB output with several structures and creating input
for an EEE calculation. 
"""

from eee.util.data import AA_3TO1
from eee.util import write_pdb
from eee.util import load_structure
from eee.util import logger

import numpy as np
import pandas as pd

import os
import random
import string
import subprocess
import glob
import shutil

def _clean_structures(dfs,
                      foldx_binary="foldx",
                      verbose=False,
                      keep_temporary=False):
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
        
        try:
            popen = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
        except FileNotFoundError:
            err = f"could not find {foldx_binary} in the path.\n"
            raise RuntimeError(err)
        
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
    if not keep_temporary:
        shutil.rmtree(tmp_dir)

    return new_dfs

def _run_muscle(seq_list,
                muscle_binary="muscle",
                verbose=False,
                keep_temporary=False):
    """
    Actually run muscle.
    """

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
        try:
            popen = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
        except FileNotFoundError:
            err = f"could not find {muscle_binary} in the path.\n"
            raise RuntimeError(err)
        
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
    Use muscle to align sequences from rcsb files, then do some clean up. 
    Renumber residues so they match between structures. If a site has a mixture
    of CYS and SER across structures, mutate the SER to CYS. 
    
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
        list of pandas dataframes with structures updated to have the shared_fx,
        alignment_site, and identical_aa columns. 
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

    # Check sequence identity. identical_aa is True if the amino acids are the 
    # same at that site (or a mix of Ser and Cys), False if they differ. Gaps
    # do not count as different. 
    ser_to_cys = {}
    identical_aa = np.ones(len(column_contents),dtype=float)
    for i in range(len(column_contents)):

        struct_seen = column_contents[i]
        aa_seen = list(set([output[j][i] for j in struct_seen]))

        # All same
        if len(aa_seen) == 1:
            continue

        # If mixture of Cys and Ser seen, mutate SER --> CYS
        if set(aa_seen) == set(["C","S"]):
            ser_to_cys[i] = []
            for j in range(len(column_contents[i])):
                if output[j][i] == "S":
                    ser_to_cys[i].append(j)
        else:
            identical_aa[i] = False

    # Get lists of all CA atoms and residues
    residues = []
    for df in dfs:

        df["_resid_key"] = list(zip(df["chain"],df["resid"],df["resid_num"]))
        
        mask = np.logical_and(df.atom == "CA",df["class"] == "ATOM")
        this_df = df.loc[mask,:]

        residues.append(list(this_df["_resid_key"]))

    # Create an array indicating the fraction of structures sharing the site. 
    shared_column = np.zeros(len(column_contents),dtype=float)
    num_structures = len(dfs)
    for i in range(len(column_contents)):
        shared_column[i] = len(column_contents[i])/num_structures

    # Go through the alignment for each structure
    for i in range(len(column_indexes)):
        
        this_df = dfs[i]
        this_df["shared_fx"] = 0.0
        this_df["identical_aa"] = 0.0
        
        for j in range(len(column_indexes[i])):
            
            idx = column_indexes[i][j]

            # Record how many structures share the column
            this_resid = residues[i][j]
            this_resid_mask = this_df["_resid_key"] == this_resid

            # Change residue numbering
            this_df.loc[this_resid_mask,"resid_num"] = f"{idx + 1:d}"

            # Record shared fraction
            this_df.loc[this_resid_mask,"shared_fx"] = shared_column[idx]

            # Record identical amino acids. 
            this_df.loc[this_resid_mask,"identical_aa"] = identical_aa[idx]        

            # Mutate ser to cys from sites with mix of ser and cys across the
            # structures
            if j in ser_to_cys:
                if i in ser_to_cys[j]:
                    if verbose:
                        logger.log(f"Introducing S{j}C into structure {i}")
                        
                    this_df.loc[this_resid_mask,"resid"] = "CYS"
                    atom_mask = np.logical_and(this_resid_mask,
                                               this_df["atom"] == "OG")
                    this_df.loc[atom_mask,"atom"] = "SG"

                    
    # Remove "_resid_key" convenience column
    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop(columns="_resid_key")
        
    return dfs


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
    verbose : bool, default = True
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
        no_het_df = df.loc[df["class"] == "ATOM",:]

        files.append(f"tmp-align_{i}_{out_base}.pdb")
        write_pdb(no_het_df,files[-1])
        
    # Go through all but first file
    for i, f in enumerate(files[1:]):
        
        # Align file to first file using lovoalign
        out_file = f"tmp-out_{out_base}-{i+1}.pdb"
        cmd = [lovoalign_binary,"-p1",f,"-p2",files[0],"-o",out_file]
        
        try:
            popen = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
        except FileNotFoundError:
            err = f"could not find {lovoalign_binary} in the path.\n"
            raise RuntimeError(err)
        
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
        mask = dfs[i+1]["class"] == "ATOM"

        dfs[i+1].loc[mask,"x"] = np.array(new_df["x"])
        dfs[i+1].loc[mask,"y"] = np.array(new_df["y"])
        dfs[i+1].loc[mask,"z"] = np.array(new_df["z"])

        # Remove output file
        if not keep_temporary:
            os.remove(out_file)
        
    # Remove all temporary files
    if not keep_temporary:
        for f in files:
            os.remove(f)
        
    return dfs


def _create_unique_filenames(files):
    """
    This wacky block of code trims back filenames, right to left, until they
    are unique. This solves edge case where someone puts in files with same
    name from different directory (like 1stn.pdb and ../test/1stn.pdb). This
    loop would create output files "1stn.pdb" and "test__1stn.pdb". 
    """

    found_filenames = False
    counter = -1
    while not found_filenames:
        name_mapper = []

        if len(list(set(files))) != len(files):
            err = "structure_files must have unique filenames!"
            raise ValueError(err)

        for i in range(len(files)):    
            real_path = "__".join(files[i].split(os.path.sep)[counter:])
            if real_path not in name_mapper:
                name_mapper.append(real_path)
                found_filenames = True
            else:
                counter -= 1
                found_filenames = False
                break

    name_mapper = dict([(files[i],name_mapper[i]) for i in range(len(files))])

    return name_mapper


def sync_structures(structure_files,
                    out_dir,
                    overwrite=False,
                    verbose=False,
                    keep_temporary=False):
    """
    Take a set of structures, clean up, align, and figure out which sites are
    shared among all structures. Output is a directory with pdb files and a
    report describing structures. The residue numbers are replaced with their 
    sites in the alignment (meaning residue numbers compare between structures).
    The b-factor column of each pdb file has the fraction of structures in which
    that specific site is seen. The occupancy column is 1 if the amino acids are
    same at the site for all structures, 0 if the amino acids are different. 
    (Note: at sites with a mix of Cys and Ser across structures, the Ser 
    residues are mutated to Cys). HETATM entries will always have 0 occupancy
    and b-factors. 

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
    keep_temporary : bool, default=False
        do not delete temporary files
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
    logger.log("Cleaning up structures with FoldX.")
    dfs = _clean_structures(dfs,verbose=verbose,keep_temporary=keep_temporary)

    # Figure out which residues are shared between what structures
    logger.log("Aligning sequences using muscle.")
    dfs = _align_seq(dfs,verbose=verbose,keep_temporary=keep_temporary)

    # Align structures in 3D
    logger.log("Aligning structures using lovoalign.")
    dfs = _align_structures(dfs,verbose=verbose,keep_temporary=keep_temporary)

    # Create a unique output name for each structure file
    name_mapper = _create_unique_filenames(structure_files)

    # Write out file names. 
    logger.log(f"Writing output to '{out_dir}'.")
    for i in range(len(structure_files)):

        f = f"{name_mapper[structure_files[i]]}_clean.pdb"
        f = os.path.join(out_dir,f)
        
        write_pdb(dfs[i],
                  f,
                  bfactor_column="shared_fx",
                  occ_column="identical_aa")
    

    return dfs
    