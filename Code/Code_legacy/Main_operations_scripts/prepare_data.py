import os
import pandas as pd
from Code._csv_preprocessing import *
import zipfile
import pandas as pd
import os
import shutil
from tqdm import tqdm

def process_zip(root: str, zip_name: str, out_dir: str) -> list:
    tmp_dir = root + '/tmp/'+zip_name
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with zipfile.ZipFile(root+'/zips/'+zip_name, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    files = list(os.listdir(tmp_dir))
    dropped = []
    for file in files:
        try:
            shutil.move(tmp_dir+'/'+file, out_dir+'/'+zip_name+'_'+file)
        except:
            dropped.append(file)
    return dropped

def extract_csv_gittables(root: str, gittables_csv_directory: str, zip_files: str) -> list:
    if not os.path.exists(gittables_csv_directory):
        os.makedirs(gittables_csv_directory)
    if not os.path.exists(root+'/tmp'):
        os.makedirs(root+'/tmp')
    if not os.path.exists(root+'/zips'):
        os.makedirs(root+'/zips')
    print('Unzipping gittables')
    with zipfile.ZipFile(root+'/'+zip_files, 'r') as zip_ref:
        zip_ref.extractall(root+'/zips')
    zips = list(os.listdir(root+'/zips'))
    print(type(zips))
    dropped = []
    
    print('Creating collection')
    for zip in tqdm(zips):
        dropped+=process_zip(root=root, zip_name=zip, out_dir=gittables_csv_directory)
    return dropped

def get_table_list(df: pd.DataFrame) -> list:
    out = []
    for r in tqdm(range(df.shape[0])):
        out.append(df.iloc[r]['r_id'])
        out.append(df.iloc[r]['s_id'])
    return out

def get_table_set(tables: list) -> set:
    out = []
    for t in tables:
        out += get_table_list(t)
    return set(out)

def generate_sets_table_querying(df: pd.DataFrame, root: str) -> None:
    query_set = []
    data_lake = []
    for r in tqdm(range(df.shape[0])):
        query_set.append(df.iloc[r]['r_id'])
        data_lake.append(df.iloc[r]['s_id'])
    query_set = set(query_set)
    data_lake = set(data_lake)
    with open(root+'/query_set_1k.pkl', 'wb') as f:
        pickle.dump(query_set, f)
    with open(root+'/data_lake_10k.pkl', 'wb') as f:
        pickle.dump(data_lake, f)

if __name__ == '__main__':
    """
        Input: 
            * root directories for gittables and wikilast containing:
                - train.csv
                - test.csv
                - valid.csv
            * directories containing all the csv files in wikilast and gittables
        Output:
            * inside the root directories are created:
                - 2x table_dict.pkl
                - queryt_set_1k.pkl
                - data_lake_10k.pkl
    """
    root = ''                       # Insert the full name of the root directory here
    gittables_csv_directory = ''    # Insert the full name of the directory where to save the csv tables of gittables
    gittables_root_zip = ''         # Insert the full name of the directory containing the zip containing the tables of gittables
    zip_files_gittables_path = ''   # Path to the zip file containing the files downloaded from GitTables
    wikilast_csv_directory = ''     # Insert the full name of the directory containing the csv tables of wikilast

    extract_csv_gittables(root=gittables_root_zip, gittables_csv_directory=gittables_csv_directory, zip_files=zip_files_gittables_path)

    root_gittables = root+'/gittables_root'
    root_wikilast = root+'/wikilast_root'
    
    table_querying_tables = pd.read_csv(root_gittables+'/table_querying.csv')
    train_gittables = pd.read_csv(root_gittables+'/train.csv')
    test_gittables = pd.read_csv(root_gittables+'/test.csv')
    valid_gittables = pd.read_csv(root_gittables+'/valid.csv')
    gittables_table_set = get_table_set([train_gittables, test_gittables, valid_gittables, table_querying_tables])


    train_wikilast = pd.read_csv(root_wikilast+'/train.csv')
    test_wikilast = pd.read_csv(root_wikilast+'/test.csv')
    valid_wikilast = pd.read_csv(root_wikilast+'/valid.csv')
    wikilast_table_set = get_table_set([train_wikilast, test_wikilast, valid_wikilast])

    print('Building table_dict for GitTables')
    generate_table_dict(csv_folder_path=gittables_csv_directory, outpath=root_gittables+'/table_dict.pkl', table_set=gittables_table_set)
    print('Building table_dict for WikiLast')
    generate_table_dict(csv_folder_path=wikilast_csv_directory, outpath=root_wikilast+'/table_dict.pkl', table_set=wikilast_table_set)
    generate_sets_table_querying(table_querying_tables, root_gittables)
    print('Data preparation is complete')
