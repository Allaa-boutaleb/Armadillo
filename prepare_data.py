import os
import pandas as pd
from Code._csv_preprocessing import *

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
    root = '/home/francesco.pugnaloni/tmp'
    root_gittables = root+'/gittables_root'
    root_wikilast = root+'/wikilast_root'
    gittables_csv_directory = '/home/francesco.pugnaloni/tmp/gittables'
    wikilast_csv_directory = '/home/francesco.pugnaloni/tmp/wikilast'
    
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
