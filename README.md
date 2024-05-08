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

The two triple datasets and a directory containing the tables in WikiLast in the .csv format are available at [here](//my.hidrive.com/share/6tuees3os3).

The GitTables table collection is available [here](https://zenodo.org/records/6515799).

## Model weights
This repository contains the weights of the two versions of Armadillo used to produce the results in the paper, trained on the GitTables_triple_dataset and the WikiLast_triple_dataset respectively. The files are named `Armadillo_GitTables.pth` and `Armadillo_WikiLast.pth` and can be found in the directory called `Models`.

## Setup environment

## Running the experiments
