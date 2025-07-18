import pickle
import time
from .hypergraph import Hypergraph
from tqdm import tqdm

def generate_hypergraph_dictionary(table_dict_path: str | dict, out_path: str = None, save_hypergraph_dict: bool = True) -> dict:
    """
    Generate a dictionary of hypergraph objects from a dictionary of tables.

    Args:
        table_dict_path (str | dict): Path to the pickled table dictionary or the dictionary itself.
        out_path (str, optional): Path to save the output hypergraph dictionary. Defaults to None.
        save_hypergraph_dict (bool, optional): If True, saves the dictionary to out_path. Defaults to True.

    Returns:
        dict: The generated dictionary mapping table names to Hypergraph objects.
    """
    start_time = time.time()
    
    # --- Load Table Dictionary ---
    print("🚀 Loading table dictionary...")
    if isinstance(table_dict_path, str):
        try:
            with open(table_dict_path, 'rb') as f:
                table_dict = pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Table dictionary not found at: {table_dict_path}")
    else:
        table_dict = table_dict_path
    print(f"✅ Table dictionary loaded in: {time.time() - start_time:.2f}s\n")

    # --- Generate Hypergraphs ---
    generation_start_time = time.time()
    print("🕸️  Generating hypergraphs...")
    
    hypergraph_dict = {}
    for table_name, table_df in tqdm(table_dict.items()):
        try:
            # Instantiate the new Hypergraph object for each table.
            hypergraph_dict[table_name] = Hypergraph(df=table_df, table_name=table_name)
        except Exception as e:
            # If a table fails (e.g., it's empty), set its entry to None and print a warning.
            hypergraph_dict[table_name] = None
            print(f"\n⚠️  Warning: Could not process table '{table_name}'. Reason: {e}")

    print(f"✅ Hypergraph generation finished in: {time.time() - generation_start_time:.2f}s")

    # --- Save Output ---
    if save_hypergraph_dict and isinstance(out_path, str):
        print(f"\n💾 Saving output to {out_path}...")
        with open(out_path, 'wb') as f:
            pickle.dump(hypergraph_dict, f)
        print("✅ Output saved.")
    
    print(f"\n✨ Total execution time: {time.time() - start_time:.2f}s")
    return hypergraph_dict