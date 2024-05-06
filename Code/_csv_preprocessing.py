import pickle
import os
import pandas as pd
from tqdm import tqdm

def list_files(directory) -> list:
    """given the path of a directory return the list of its files

    Args:
        directory (_type_): path to the directory to explore

    Returns:
        list: list of filenames
    """
    l=[]
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                l.append(entry.name)
    return l

def generate_table_dict(csv_folder_path: str, outpath: str=None, table_set: set=None, log_path: str=None) -> dict:
    """_summary_

    Args:
        gittables_folder_path (str): path to the folder containing the tables in the csv format
        outpath (str): folder where to save the generated table_dict
        log_path (str, optional): path to the directory where to save the names of the dropped tables. Defaults to None.

    Returns:
        dict: a table_dict
    """
    filenames = list_files(csv_folder_path)
    log = []
    table_dict = {}
    for i in tqdm(range(len(filenames))):
        t = None
        path = csv_folder_path + '/' + filenames[i]
        try:
            t = pd.read_csv(path, sep=',', header=None)
        except:
            try:
                t = pd.read_csv(path, sep='#', header=None)
            except:
                log.append(path)
        if isinstance(t, pd.DataFrame):
            table_dict[str(filenames[i])] = t
        

    if log_path:
        with open(log_path, 'w') as file:
            # Write the string to the file
            file.write('\n'.join(log))
    if isinstance(table_set, set):
        table_dict = {k:table_dict[k] for k in table_set}

    if isinstance(outpath, str):
        with open(outpath, 'wb') as f:
            pickle.dump(table_dict, f)
    return table_dict

def generate_table_dict_from_different_folders(folders: list, outpath: str) -> dict:
    table_dict = {}
    for d in folders:
        table_dict.update(generate_table_dict(d))
    with open(outpath, 'wb') as f:
        pickle.dump(table_dict, f)
