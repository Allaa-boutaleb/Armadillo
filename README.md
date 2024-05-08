# Armadillo
This repository contains the artifacts and source code for Armadillo, an approach based on graph neural networks that learns table embeddings whose cosine similarity approximates the overlap ratio between tables.

To reproduce the results of the paper, please follow the instructions in the following sections.

## Datasets
We propose two triple datasets composed of pairs of tables and their overlap ratio. The tables are extracted from two sources, namely [GitTables](https://github.com/madelonhulsebos/gittables), a collection of tables extracted from GitHub repositories, and WikiLast, i.e., the web tables existent in Wikipedia at the time of its last snapshot considered in the [IANVS](https://hpi.de/naumann/projects/data-profiling-and-analytics/change-exploration.html) project. 

The triple datasets comprise a total of 1.32 million samples and their triples are divided into training, testing, and validation data as follows:
| Dataset                  | Training triples | Testing triples  | Validation triples | 
| ------------------------ |:----------------:| ----------------:| ------------------ |
| GitTables_triple_dataset | 500 000          | 100 000          | 100 000            |
| WikiLast_triple_dataset  | 500 000          | 60 000           | 60 000             |

Triple Datasets and WikiLast: https://my.hidrive.com/share/6tuees3os3

GitTables: https://zenodo.org/records/6515799

## Model weights

## Setup environment

## Running the experiments
