from Code._embed_all_no_paral import run_experiment
from Code._overlap_of_a_pair_effe_effi_processing import recompute_embeddings_overlaps_overlap_computation_time, repeat_test_emb_already_computed, add_new_column_prediction_armadillo
from Code._performance_overlap_computation import *
from Code._table_stats_computation import compute_tables_stats
from Code._table_querying import *
import os

def evaluate(model_GitTables: str, table_dict_GitTables: str | dict, model_WikiLast: str, table_dict_WikiLast: str | dict, table_stats_GitTables_out: str, 
             gittables_unlabeled: str, wikilast_unlabeled: str, table_stats_WikiLast_out: str, chart_data_path: str, charts_path: str) -> None:
    #prepare data for charts
    #Embedding generation time
    print('Test: embedding all of the tables in GitTables using armadillo_gittables')
    run_experiment(model_file=model_GitTables, table_dict_path=table_dict_GitTables, iters=3, experiment_data_file_path=chart_data_path+'/embedding_gen_time_gittables.pkl', embedding_file=chart_data_path+'/embedding_file_git_on_git.pkl')
    print('Test: embedding all of the tables in GitTables using armadillo_wikilast')
    run_experiment(model_file=model_WikiLast, table_dict_path=table_dict_GitTables, iters=1, embedding_file=chart_data_path+'/embedding_file_wiki_on_git.pkl')

    print('Test: embedding all of the tables in WikiLast using armadillo_wikilast')
    run_experiment(model_file=model_WikiLast, table_dict_path=table_dict_WikiLast, iters=3, experiment_data_file_path=chart_data_path+'/embedding_gen_time_wikilast.pkl', embedding_file=chart_data_path+'/embedding_file_wiki_on_wiki.pkl')
    print('Test: embedding all of the tables in WikiLast using armadillo_gittables')
    run_experiment(model_file=model_GitTables, table_dict_path=table_dict_WikiLast, iters=1, embedding_file=chart_data_path+'/embedding_file_git_on_wiki.pkl')

    #Major comparison
    recompute_embeddings_overlaps_overlap_computation_time(unlabeled_dataset=gittables_unlabeled, model_file=model_GitTables, table_dict=table_dict_GitTables, output_file=chart_data_path+'/effe_effi_gittables.csv')
    repeat_test_emb_already_computed(old_file=chart_data_path+'/effe_effi_gittables.csv', embeddings_dict='', out_path=chart_data_path+'/effe_effi_gittables.csv')
    add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_gittables.csv', embedding_dict=chart_data_path+'/embedding_file_git_on_git.pkl', out_path=chart_data_path+'/effe_effi_gittables.csv', label='gittables')
    add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_gittables.csv', embedding_dict=chart_data_path+'/embedding_file_wiki_on_git.pkl', out_path=chart_data_path+'/effe_effi_gittables.csv', label='wikilast')


    recompute_embeddings_overlaps_overlap_computation_time(unlabeled_dataset=wikilast_unlabeled, model_file=model_WikiLast, table_dict=table_dict_WikiLast, output_file=chart_data_path+'/effe_effi_wikilast.csv')
    repeat_test_emb_already_computed(old_file=chart_data_path+'/effe_effi_wikilast.csv', embeddings_dict='', out_path=chart_data_path+'/effe_effi_wikilast.csv')
    add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_wikilast.csv', embedding_dict=chart_data_path+'/embedding_file_git_on_wiki.pkl', out_path=chart_data_path+'/effe_effi_wikilast.csv', label='gittables')
    add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_wikilast.csv', embedding_dict=chart_data_path+'/embedding_file_wiki_on_wiki.pkl', out_path=chart_data_path+'/effe_effi_wikilast.csv', label='wikilast')


    #NDCG table querying

    # add_armadillo_predictions_to_sloth_file()
    # compute_ndcg_at_k()



    #plot chart MAE compared to baseline
    print('Generating MAE boxplot for GitTables')
    plot_box_group(chart_data_path+'/effe_effi_gittables.csv', label_list=['AE_armadillo', 'armadillo_wikitables_AE','AE_josie','AE_jsim'], out_pdf=charts_path+'/mae_comparison_baseline_gittables.pdf')
    print('Generating MAE boxplot for WikiLast')
    plot_box_group(chart_data_path+'/effe_effi_wikilast.csv', label_list=['armadillo_wikitables_AE','armadillo_gittables_AE','o_set_sim_AE','jsim_AE'],out_pdf=charts_path+'/mae_comparison_baseline_wikilast.pdf')

    #plot chart MAE per bin compared to baseline
    print('Generating MAE per bin barplot for GitTables')
    compare_models_hist(chart_data_path+'/effe_effi_gittables.csv', out_pdf=charts_path+'/mae_comparison_baseline_gittables.pdf', font_scale=1.1)
    print('Generating MAE per bin barplot for WikiLast')
    compare_models_hist(chart_data_path+'/effe_effi_wikilast.csv', out_pdf=charts_path+'/mae_comparison_baseline_wikilast.pdf', font_scale=1.1)

    #plot chart NDCG
    print('Generating ndcg barplot')
    compare_models_ndcg(chart_data_path+'/ndcg_data.csv', font_scale=1.1, out_pdf=charts_path+'/ndcg_score.pdf')

    #plot chart overlap computation time
    print('Generating scatterplot for comparing overlap computation time on GitTables')
    show_scatter_t_exec_sloth_arm(chart_data_path+'/effe_effi_gittables.csv', logx=True, logy=True, output_pdf=charts_path+'/overlap_pair_t_exc_comparison_arm_sloth_gittabels.pdf')
    print('Generating scatterplot for comparing overlap computation time on WikiLast')
    show_scatter_t_exec_sloth_arm(chart_data_path+'/effe_effi_wikilast.csv', logx=True, logy=True, output_pdf=charts_path+'/overlap_pair_t_exc_comparison_arm_sloth_wikilast.pdf')

    #plot chart embedding generation time with increasing table areas
    print('Generating scatterplot to show how embedding generation time varies with increasing table areas on GitTables')
    visualize_scatter_plot(chart_data_path+'/embedding_gen_time_gittables.pkl', logx=True, logy=True, out_pdf=charts_path+'/emb_gen_t_exec_gittables.pdf', font_size=15)
    print('Generating scatterplot to show how embedding generation time varies with increasing table areas on WikiLast')
    visualize_scatter_plot(chart_data_path+'/embedding_gen_time_wikilast.pkl', logx=True, logy=True, out_pdf=charts_path+'/emb_gen_t_exec_wikitables.pdf', font_size=15)

    #compute table stats
    print('Computing table stats for GitTables')
    compute_tables_stats(table_dict_GitTables, table_stats_GitTables_out)
    print('Computing table stats for WikiLast')
    compute_tables_stats(table_dict_WikiLast, table_stats_WikiLast_out)

if __name__ == '__main__':
    root = '/home/francesco.pugnaloni/tmp'
    root_gittables = root+'/gittables_root'
    root_wikilast = root+'/wikilast_root'

    armadillo_gittables = ''
    armadillo_wikilast = ''

    if not os.path.exists(root+'/chart_data'):
        os.makedirs(root+'/chart_data')

    if not os.path.exists(root+'/charts'):
        os.makedirs(root+'/charts')

    evaluate(model_GitTables=armadillo_gittables, table_dict_GitTables=root_gittables+'/table_dict.pkl', graph_dict_GitTables=root_gittables+'/graph_dict.pkl',
             model_WikiLast=armadillo_wikilast, table_dict_WikiLast=root_wikilast+'/table_dict.pkl', graph_dict_WikiLast=root_wikilast+'/graph_dict.pkl',
             chart_data_path=root+'/chart_data', charts_path=root+'/charts')
