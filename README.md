# Dynamic-Weighted-Contrastive-Learning
Dynamic Weighted Contrastive Learning
# BERT-DPCNN-DW-SupCon: High-Value Patent Identification Method Based on BERT-DPCNN Fusion and Dynamic Weighted Contrastive Learning

## Description

This repository contains the public experimental framework of BERT-DPCNN-DW-SupCon, a deep learning model combining BERT-DPCNN feature fusion and Dynamic Weighted Supervised Contrastive Learning (DW-SupCon) for high-value patent identification. The repository includes the public script corresponding to the experimental setting reported in the associated article:

* train_eval.py: public binary high-value patent identification on a lithography patent dataset

The released code provides the model architecture, training loop, dynamic contrastive loss calculation, and evaluation pipeline. Some dataset-specific preprocessing procedures and external resources used in the full experiments are not redistributed in this public release.

## Dataset Information

Note on Data Availability

The experiments in the article involve a proprietary-source dataset.

* The lithography patent dataset was derived from records obtained via the IncoPat platform. Because of third-party platform restrictions and practical constraints, the processed dataset is not directly redistributed in this repository.

Researchers who wish to reproduce the experiments should prepare their own input files according to the data format described in the code. A small sample dataset structure is provided in the `data/` directory to demonstrate the input schema (specifically `text_a` and `label` in `.tsv` format).

## Requirements

To run this framework, the following packages are required:

* tensorflow
* transformers
* pandas
* numpy
* scikit-learn

A minimal requirements.txt can list these packages and their versions.

## Methodology

At a high level, the implementation follows these steps:

1. Load the patent text data and binary labels (high-value vs. standard)
2. Tokenize and encode input text using a pre-trained BERT-base-chinese model
3. Extract multi-scale local phrase features via parallel DPCNN convolutional blocks (kernel sizes 2, 3, and 4) with Mish activation
4. Project the fused global-local representation features into a normalized 256-dimensional space
5. Calculate the joint loss of binary cross-entropy and Dynamic Weighted Supervised Contrastive Loss (DW-SupCon), where hard-sample weights scale dynamically across epochs
6. Train the network using an Adam optimizer with warmup and decay learning rate schedules
7. Evaluate and report classification performance metrics (Accuracy, Precision, Recall, F1-score) on the test set

For full methodological details, please refer to the associated article.

## Citation

If you use this code or the associated framework in your research, please cite the corresponding article.

## Data Sources

* IncoPat Global Patent Database (third-party platform source for the lithography patent records)

## License

This code is released under the MIT License. Please include a LICENSE file in the repository if you choose to distribute it under MIT terms.

## Contributions

This repository is intended to support transparency and partial reproducibility of the associated article. Bug reports and small improvements are welcome, provided that changes are clearly documented and do not conflict with the original experimental setting.
