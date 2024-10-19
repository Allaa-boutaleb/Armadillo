# Armadillo
This repository contains the artifacts that regard Armadillo, an approach based on graph neural networks for the generation of table embeddings whose cosine similarity approximates the overlap ratio between tables.

To reproduce the results of the paper, please follow the instructions in the following sections.

## Datasets
We propose two triple datasets composed of pairs of tables and their overlap ratio. The tables are extracted from two sources, namely [GitTables](), a collection of tables extracted from GitHub repositories, and WikiTables, i.e., the web tables existent in Wikipedia at the time of its last snapshot considered in the [IANVS]() project. 

The triple datasets comprise a total of 1.32 million samples and their triples are divided into training, testing, and validation data as follows:
| Dataset                  | Training triples | Testing triples  | Validation triples | 
| ------------------------ |-----------------:| ----------------:| ------------------:|
| GitTables_triple_dataset | 500 000          | 100 000          | 100 000            |
| WikiTables_triple_dataset  | 500 000          | 60 000           | 60 000             |

Note that there are no repetitions of tables in the train-test-validation split, so a model trained on these datasets is granted not only to be evaluated on previously unseen couples but also on completely new tables.

In addition to them, taking care of not including tables that appear inside the training and validation triples, we extracted at random from GitTables two sets of tables containing 100 and 10 000 tables each and built a third triple datasets that emulates a table querying scenario, where each of the 100 query tables is compared to the 10 000 tables of the other set, containing a total of 1 000 000 pairs.

The triple datasets and the processed tables that we extracted from GitTables and Wikipedia in .csv format, are available [here](). (Note that due to anonymity requirements all of the links are disabled and will be added after acceptance of the paper. In this repository, we provide in the `Dataset` folders the .csv files containing the triple datasets, but due to the size of the files containing the actual tables we were unable to upload them)

## Model weights
This repository contains the weights of the two versions of Armadillo used to produce the results in the paper and the weights of all of the scaling heads of the baselines. All of them are trained on the GitTables_triple_dataset and the WikiLast_triple_dataset and are stored in the directory called `Models`.

## Running the experiments
Before running the experiments perform the following preliminary operations:
* Our experiments were performed in a conda environment, use the `Armadillo.yml` file to replicate our configuration.
* Download the directory containing the triple datasets and the tables of WikiTables and GitTables.
* Decompress the files containing the .csv tables of WikiTables and GitTables.
* Download [this]() version of TURL and insert the repository into `Baselines/turl`
* Download [this]() version of EmbDI and insert the repository into `Baselines/embdi`

We provide scripts to automate the replication of our experiments:
* `1_data_preparation.py`: generates some preliminary artifacts necessary for running the experiments. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory, to the `gittables_csv_directory` variable the path to an empty directory where all the .csv files of GitTables will be extracted, and to the `wikilast_csv_directory` the path to the directory containing the .csv files of WikiLast. Also create a directory containing the zip file containing the zipped .csv of GitTables and assign to the variable `gittables_root_zip` the path to this directory and to the variable `zip_files_gittables_path` the path to the actual .zip file.
* `2_run_training_armadillo.py`: retrain a new Armadillo model from scratch using the triple datasets. Note that the retraining is heavily demanding in terms of resources, in particular, it is suggested to have at least 50 GB of RAM and free disk space when using the WikiLast_triple_dataset and 200 GB of RAM and free disk space when using the GitTables_triples_dataset. Before running the script open the file and assign to the `root_dataset` variable the path to the directory containing the triple datasets for train, test, and validation to use. The trained model can be found inside the `root_dataset` directory as `model.pth`.
* `3_generate_embeddings_baselines.py`: for each baseline embedding model generate table embeddings of the tables in GitTables and WikiTables.
* `4_train_scaling_heads_baselines.py`: for each baseline embedding model, train a scaling head to scale the embeddings generated using the third script into the correct format.
* `run_evaluation.py`: runs the experiments presented in the paper using `Armadillo_git` and `Armadillo_wiki` and generates the charts related to them as .pdf files. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory. The script generates two new directories inside `root`, namely `charts` and `chart_data`, the first one contains the charts as .pdf files, while the other one the files with the results used to generate them.
