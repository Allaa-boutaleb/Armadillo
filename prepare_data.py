import os
import pandas as pd
from Code._csv_preprocessing import *
from Code._generate_graph_dict import *

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
                - table_dict.pkl
                - graph_dict.pkl
    """
    root_gittables = '/home/francesco.pugnaloni/tmp/gittables_root'
    root_wikilast = '/home/francesco.pugnaloni/tmp/wikilast_root'
    gittables_csv_directory = '/home/francesco.pugnaloni/tmp/gittables'
    wikilast_csv_directory = '/home/francesco.pugnaloni/tmp/wikilast'
    
    train_gittables = pd.read_csv(root_gittables+'/train.csv')
    test_gittables = pd.read_csv(root_gittables+'/test.csv')
    valid_gittables = pd.read_csv(root_gittables+'/valid.csv')
    gittables_table_set = get_table_set([train_gittables, test_gittables, valid_gittables])


    train_wikilast = pd.read_csv(root_wikilast+'/train.csv')
    test_wikilast = pd.read_csv(root_wikilast+'/test.csv')
    valid_wikilast = pd.read_csv(root_wikilast+'/valid.csv')
    wikilast_table_set = get_table_set([train_wikilast, test_wikilast, valid_wikilast])

    print('Building table_dict for GitTables')
    generate_table_dict(csv_folder_path=gittables_csv_directory, outpath=root_gittables+'/table_dict.pkl', table_set=gittables_table_set)
    print('Building table_dict for WikiLast')
    generate_table_dict(csv_folder_path=wikilast_csv_directory, outpath=root_wikilast+'/table_dict.pkl', table_set=wikilast_table_set)

    print('Building graph_dict for GitTables')
    generate_graph_dictionary(table_dict_path=root_gittables+'/table_dict.pkl', out_path=root_gittables+'/graph_dict.pkl')
    print('Building graph_dict for WikiLast')
    generate_graph_dictionary(table_dict_path=root_wikilast+'/table_dict.pkl', out_path=root_wikilast+'/graph_dict.pkl')
