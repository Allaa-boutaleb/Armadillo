import pandas as pd
import torch
from Code.node_embeddings import Hash_embedding_buffer

class Hypergraph:
    """
    Represents a table as a hypergraph, designed for use with Hypergraph Neural Networks.
    This implementation follows the HYPERDILLO proposal.

    The structure is defined as:
    - Nodes (V): One for each cell in the table.
    - Hyperedges (E): One for each row and one for each column. A hyperedge connects all the cell nodes it contains.
    """
    def __init__(self, df: pd.DataFrame, table_name: str, verbose: bool=False):
        """
        Constructs the hypergraph representation from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input table.
            table_name (str): The name of the table.
            verbose (bool, optional): If True, prints debug information. Defaults to False.
        """
        self.table_name = table_name
        
        # Drop empty rows or columns which would create zero-degree hyperedges.
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        n_rows, n_cols = df.shape
        num_nodes = n_rows * n_cols
        num_hyperedges = n_rows + n_cols

        if num_nodes == 0:
            raise ValueError('Cannot generate a hypergraph from an empty DataFrame')

        # --- 1. Node Feature Initialization (x) ---
        # Each cell becomes a node, initialized with its SHA-256 hash.
        embedding_buffer = Hash_embedding_buffer()
        cell_embeddings = []
        for r in range(n_rows):
            for c in range(n_cols):
                cell_value = df.iloc[r, c]
                # Treat NaN values as the string 'NULL' to match Armadillo's logic.
                sentence = 'NULL' if pd.isnull(cell_value) else str(cell_value)
                embedding_buffer(sentence)
        
        # `x` is the node feature matrix of shape [num_nodes, feature_dim]
        self.x = embedding_buffer.pop_embeddings()

        # --- 2. Hyperedge Index Construction (hyperedge_index) ---
        # This tensor defines the connections between nodes and hyperedges.
        # Shape: [2, total_connections], where row 0 is node_idx and row 1 is hyperedge_idx.
        node_indices = []
        hyperedge_indices = []

        # Create row-hyperedges (indices 0 to n_rows-1)
        for r in range(n_rows):
            for c in range(n_cols):
                node_idx = r * n_cols + c
                node_indices.append(node_idx)
                hyperedge_indices.append(r)

        # Create column-hyperedges (indices n_rows to n_rows+n_cols-1)
        for c in range(n_cols):
            for r in range(n_rows):
                node_idx = r * n_cols + c
                node_indices.append(node_idx)
                hyperedge_indices.append(n_rows + c)
        
        self.hyperedge_index = torch.tensor([node_indices, hyperedge_indices], dtype=torch.long)

        # --- 3. Hyperedge Feature Initialization (hyperedge_attr) ---
        # Each hyperedge's feature is the mean of its constituent node features.
        hyperedge_features = []

        # Features for row-hyperedges
        for r in range(n_rows):
            start_node_idx = r * n_cols
            end_node_idx = start_node_idx + n_cols
            row_node_features = self.x[start_node_idx:end_node_idx]
            hyperedge_features.append(torch.mean(row_node_features, dim=0))

        # Features for column-hyperedges
        for c in range(n_cols):
            col_node_indices = [r * n_cols + c for r in range(n_rows)]
            col_node_features = self.x[col_node_indices]
            hyperedge_features.append(torch.mean(col_node_features, dim=0))

        # `hyperedge_attr` is the hyperedge feature matrix of shape [num_hyperedges, feature_dim]
        self.hyperedge_attr = torch.stack(hyperedge_features)