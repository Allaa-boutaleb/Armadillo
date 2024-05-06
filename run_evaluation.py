from Code._generate_graph_dict import generate_graph_dictionary
from Code._embed_all_no_paral import run_experiment
from Code._training_pipeline import run_Armadillo_experiment_split
from Code._overlap_of_a_pair_effe_effi_processing import recompute_embeddings_overlaps_overlap_computation_time, repeat_test_emb_already_computed
from Code._performance_overlap_computation import predict_overlap_compute_AE
from Code._table_stats_computation import compute_tables_stats
import pandas as pd
import pickle
import time

def evaluate(model_GitTables: str, table_dict_GitTables: str | dict, graph_dict_GitTables: str | dict, model_WikiLast: str, table_dict_WikiLast: str | dict, graph_dict_WikiLast: str | dict,
             table_stats_GitTables_out: str, table_stats_WikiLast_out: str) -> None:
    #prepare data for charts

    #plot chart MAE compared to baseline

    #plot chart MAE per bin compared to baseline

    #plot chart NDCG

    #plot chart overlap computation time

    #plot chart embedding generation time

    #compute table stats
    print('Computing table stats for GitTables')
    compute_tables_stats(table_dict_GitTables, table_stats_GitTables_out)
    print('Computing table stats for WikiLast')
    compute_tables_stats(table_dict_WikiLast, table_stats_WikiLast_out)
