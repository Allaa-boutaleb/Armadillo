# Armadillo
This repository contains the artifacts that regard Armadillo, an approach based on graph neural networks for the generation of table embeddings whose cosine similarity approximates the overlap ratio between tables.

To reproduce the results of the paper, please follow the instructions in the following sections.

## Datasets
We propose two triple datasets composed of pairs of tables and their overlap ratio. The tables are extracted from two sources, namely [GitTables](https://gittables.github.io/), a collection of tables extracted from GitHub repositories, and WikiTables, i.e., the web tables existent in Wikipedia at the time of its last snapshot considered in the [IANVS](https://hpi.de/naumann/projects/data-profiling-and-analytics/change-exploration.html) project. 

The triple datasets comprise a total of 1.32 million samples and their triples are divided into training, testing, and validation data as follows:
| Dataset                  | Training triples | Testing triples  | Validation triples | 
| ------------------------ |-----------------:| ----------------:| ------------------:|
| GitTables_triple_dataset | 500 000          | 100 000          | 100 000            |
| WikiTables_triple_dataset  | 500 000          | 60 000           | 60 000             |

Note that there are no repetitions of tables in the train-test-validation split, so a model trained on these datasets is granted not only to be evaluated on previously unseen couples but also on completely new tables.

In addition to them, taking care of not including tables that appear inside the training and validation triples, we extracted at random from GitTables two sets of tables containing 100 and 10 000 tables each and built a third triple dataset that emulates a table querying scenario, where each of the 100 query tables is compared to the 10 000 tables of the other set, containing a total of 1 000 000 pairs.

The triple datasets and the processed tables that we extracted from GitTables and Wikipedia in .csv format, are available [here](https://my.hidrive.com/share/6tuees3os3). (Note that due to anonymity requirements, all of the links are disabled and will be added after acceptance of the paper. In this repository, we provide in the `Dataset` folders the .csv files containing the triple datasets, but due to the size of the archive containing the actual tables we were unable to upload them)

## Model weights
This repository contains the weights of the two versions of Armadillo used to produce the results in the paper and the weights of all of the scaling heads of the baselines. All of them are trained on the GitTables_triple_dataset and the WikiTables_triple_dataset and are stored in the directory called `Models`.

## Running the experiments
Before running the experiments perform the following preliminary operations:
* Our experiments were performed in a conda environment, use the `Armadillo.yml` file to replicate our configuration.
* Download the directory containing the triple datasets and the tables of WikiTables and GitTables.
* Decompress the files containing the .csv tables of WikiTables and GitTables.
* Download [this]() version of TURL and insert the repository into `Baselines/turl`
* Download [this]() version of EmbDI and insert the repository into `Baselines/embdi`

We provide eight scripts and one notebook to replicate our experiments:
* `1_data_preparation.py`: generates the preliminary artifacts necessary for running the experiments. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory.
* `2_run_training_armadillo.py`: retrain a new Armadillo model from scratch using the triple datasets. Note that the training is heavily demanding in terms of resources, in particular, it is suggested to have at least 50 GB of RAM and free disk space when using the WikiTables_triple_dataset and 200 GB of RAM and free disk space when using the GitTables_triples_dataset. Before running the script open the file and assign to the `root_dataset` variable the path to the directory containing the triple datasets for train, test, and validation to use. 
* `3_generate_embeddings_baselines.py`: for each baseline embedding model generate table embeddings of the tables in GitTables and WikiTables. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory.
* `4_train_scaling_heads_baselines.py`: for each baseline embedding model, train a scaling head to scale the embeddings generated using the third script into the correct format. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory.
* `5_evaluate_scaling_models.py`: for each baseline embedding model, run the evaluation. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory.
* `6_evaluate_jaccard_bag.py`: evaluates the baselines based on the Jaccard under bag semantics. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory.
* `7_evaluate_armadillo.py`: Evaluates armadillo. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory.
* `8_prepare_plot_data.py`: prepares a data frame containing the data necessary for generating the charts. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory.
* `9_plots.ipynb`: contains the code for visualizing the results.
