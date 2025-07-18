import os
from Code._csv_preprocessing import generate_table_dict_from_different_folders

if __name__ == '__main__':
    # --- Configuration ---
    # Set the following flags to True to process the corresponding dataset.
    PROCESS_GITTABLES = False
    PROCESS_WIKITABLES = True
    PROCESS_QUERYING_DATA = False

    # --- Path Definitions ---
    # Root directory containing all dataset folders
    root = 'data'
    root_git = os.path.join(root, 'gittables')
    root_wiki = os.path.join(root, 'wikitables')
    # [cite_start]The querying dataset uses tables from the GitTables corpus [cite: 345]
    root_table_querying = root_git 
    
    # --- Data Processing ---
    
    if PROCESS_GITTABLES:
        print("🚀 Processing GitTables dataset...")
        output_dir = os.path.join(root_git, 'dictionaries')
        os.makedirs(output_dir, exist_ok=True)
        
        source_folders = [
            os.path.join(root_git, 'train'),
            os.path.join(root_git, 'test'),
            os.path.join(root_git, 'valid')
        ]
        
        generate_table_dict_from_different_folders(
            folders=source_folders,
            outpath=os.path.join(output_dir, 'table_dict.pkl'),
            anon=False
        )
        print("✅ GitTables processing complete.")

    if PROCESS_WIKITABLES:
        print("\n🚀 Processing WikiTables dataset...")
        output_dir = os.path.join(root_wiki, 'dictionaries')
        os.makedirs(output_dir, exist_ok=True)

        source_folders = [
            os.path.join(root_wiki, 'train'),
            os.path.join(root_wiki, 'test'),
            os.path.join(root_wiki, 'valid')
        ]
        
        generate_table_dict_from_different_folders(
            folders=source_folders,
            outpath=os.path.join(output_dir, 'table_dict.pkl'),
            anon=False
        )
        print("✅ WikiTables processing complete.")

    if PROCESS_QUERYING_DATA:
        print("\n🚀 Processing Table Querying dataset...")
        output_dir = os.path.join(root_table_querying, 'dictionaries')
        os.makedirs(output_dir, exist_ok=True)
        
        source_folders = [os.path.join(root_table_querying, 'tables')]
        
        generate_table_dict_from_different_folders(
            folders=source_folders,
            outpath=os.path.join(output_dir, 'table_dict_querying.pkl'),
            anon=False
        )
        print("✅ Table Querying processing complete.")