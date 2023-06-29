"""
Write a pdb file from a dataframe with atom information.
"""
import os

def write_pdb(df,pdb_file,overwrite=False):
    """
    Write a pdb file given a pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with structural data (generally created using load_structure)
    pdb_file : str
        name of pdb file to write
    overwrite : bool, default=False
        overwrite the pdb file if it exists
    """
    
    if os.path.exists(pdb_file):
        if not overwrite:
            err = f"pdb_file {pdb_file} already exists.\n"
            raise FileExistsError(err)
        else:
            if os.path.isfile(pdb_file):
                os.remove(pdb_file)
            else:
                err = f"pdb_file {pdb_file} exists but is not a regular file.\n"
                err += "Cannot overwrite.\n"
                raise FileExistsError(err)

    df = df.copy()
    
    last_chain = None
    last_class = None
    with open(pdb_file,'w') as f:
        
        counter = 1
        for i in df.index:
            
            row = df.loc[i,:]
            
            chain = row['chain']
            if last_chain is None:
                last_chain = chain
                
            atom_class = row['class']
            if last_class is None:
                last_class = atom_class
                
            if chain != last_chain and last_class == "ATOM":
                f.write("TER\n")
                last_chain = chain 
                
            f.write(f"{row['class']:6s}{counter:5d}")
        
            atom = row["atom"]
            if len(atom) < 5:
                atom = f" {atom:<4s}"
            
            f.write(f" {atom}{row['resid']:3s} {row['chain']}{row['resid_num']:4s}")
            f.write(f"    {row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}")
            f.write(f"{row['occ']:6.2f}{row['b']:6.2f}{row['elem']:>12s}\n")
        
            last_class = atom_class
            counter += 1
            
        f.write("END\n")
