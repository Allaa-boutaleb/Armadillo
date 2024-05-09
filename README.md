# Armadillo
This repository contains the artifacts that regard Armadillo, an approach based on graph neural networks for the generation of table embeddings whose cosine similarity approximates the overlap ratio between tables.

To reproduce the results of the paper, please follow the instructions in the following sections.

## Datasets
We propose two triple datasets composed of pairs of tables and their overlap ratio. The tables are extracted from two sources, namely [GitTables](https://github.com/madelonhulsebos/gittables), a collection of tables extracted from GitHub repositories, and WikiLast, i.e., the web tables existent in Wikipedia at the time of its last snapshot considered in the [IANVS](https://hpi.de/naumann/projects/data-profiling-and-analytics/change-exploration.html) project. 

The triple datasets comprise a total of 1.32 million samples and their triples are divided into training, testing, and validation data as follows:
| Dataset                  | Training triples | Testing triples  | Validation triples | 
| ------------------------ |:----------------:| ----------------:| ------------------ |
| GitTables_triple_dataset | 500 000          | 100 000          | 100 000            |
| WikiLast_triple_dataset  | 500 000          | 60 000           | 60 000             |

Note that there are no repetitions of tables in the train-test-validation split, so a model trained on these datasets is granted not only to be evaluated on previously unseen couples but also on completely new tables.

The two triple datasets and a directory containing the tables in WikiLast in the .csv format are available [here](https://my.hidrive.com/share/6tuees3os3).

The GitTables table collection is available [here](https://zenodo.org/records/6515973).

## Model weights
This repository contains the weights of the two versions of Armadillo used to produce the results in the paper. They are trained on the GitTables_triple_dataset and the WikiLast_triple_dataset respectively and are stored in the files named `Armadillo_GitTables.pth` and `Armadillo_WikiLast.pth` that can be found in the directory called `Models`.

## Running the experiments
Before running the experiments perform the following preliminary operations:
* Our experiments were performed in a conda environment, use the `Armadillo.yml` file to replicate our configuration.
* Download the GitTables table collection and save all of the tables in the .csv format in a directory.
* Download the directory containing the triple datasets and the tables of WikiLast.
* Decompress the file containing the .csv tables of WikiLast.

We provide three scripts to automate the replication of our experiments:
* `prepare_data.py`: generates some preliminary artifacts necessary for running the experiments. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory, to the `gittables_csv_directory` variable the path to the directory containing the .csv files of GitTables, and to the `wikilast_csv_directory` the path to the directory containing the .csv files of WikiLast.
* `run_training.py`: retrain a new Armadillo model from scratch using the triple datasets. Note that the retraining is heavily demanding in terms of resources, in particular, it is suggested to have at least 50 GB of RAM and free disk space when using the WikiLast_triple_dataset and 200 GB of RAM and free disk space when using the GitTables_triples_dataset. Before running the script open the file and assign to the `root_dataset` variable the path to the directory containing the triple datasets for train, test, and validation to use. The trained model can be found inside the `root_dataset` directory as `model.pth`.
* `run_evaluation.py`: runs all the experiments presented in the paper and generates the charts related to them as .pdf files. Before running the script open the file and assign to the `root` variable the path to the `triple_datasets` directory. The script generates two new directories inside `root`, namely `charts` and `chart_data`, the first one contains the charts as .pdf files, while the other one the files with the results used to generate them.
