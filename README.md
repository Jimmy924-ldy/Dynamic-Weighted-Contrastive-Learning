# Dynamic-Weighted-Contrastive-Learning

## BERT-DPCNN-DW-SupCon: High-Value Patent Identification Method Based on BERT-DPCNN Fusion and Dynamic Weighted Contrastive Learning

## Description
This repository contains the public experimental framework of **BERT-DPCNN-DW-SupCon**, a deep learning model combining BERT-DPCNN feature fusion and Dynamic Weighted Supervised Contrastive Learning (DW-SupCon) for high-value patent identification.

The repository includes the public script corresponding to the experimental setting reported in the associated article:
* `qgbertdpcnnwsupcon.py`: Public binary high-value patent identification on a lithography patent dataset.

The released code provides the core model architecture, custom joint training loops, dynamic contrastive loss calculation, and evaluation pipelines. Some dataset-specific preprocessing procedures and external heavy resources used in the full experiments are not redistributed directly in this public repository.

---

## Code Information
The repository is structured as follows:
```text
├── data/
│   ├── train.tsv               # Sample training dataset schema
│   ├── dev.tsv                 # Sample validation dataset schema
│   └── test.tsv                # Sample test dataset schema
├── qgbertdpcnnwsupcon.py       # Main Python script (Model, Training, and Inference)
└── README.md                   # Project documentation
Input Data Requirements
To run the qgbertdpcnnwsupcon.py script, you need to prepare your dataset in a tab-separated values (.tsv) format. At a minimum, the input files must contain the following columns:

text_a: The textual content of the patent (e.g., patent abstract, claims, or titles used for semantic extraction).

label: The binary target label (1 representing high-value patents, 0 representing standard/non-high-value patents).

The script also requires a local directory containing the pre-trained weights of bert-base-chinese specified via the internal parameters.

Dataset Information & Data Availability
Note on Data Availability
The experiments in the article involve a lithography patent dataset derived from records obtained via the IncoPat platform. To comply with the journal's Open Data Policy while respecting third-party platform redistribution restrictions:

The raw, fully anonymized, and de-identified dataset used to validate the statistical analysis in this study has been uploaded as a Supplemental File (Raw_Data_.xlsx) alongside the manuscript submission (or available via the designated public remote repository DOI link specified in the paper).

For security and licensing reasons, the full production dataset is not hosted directly inside this public GitHub repository. However, minimal dummy sample structures are provided in the data/ directory to demonstrate the required input schema.

Data Collection & Labeling Rules
The lithography patent dataset was retrieved from the IncoPat Global Patent Database, focusing on Chinese patent records (country code "CN") under explicit keyword strategies targeting lithography technologies and equipment.
Using IncoPat's proprietary patent value score (HeXiang value score, ranging from 1 to 10), legal status, and validity, the binary dataset was constructed as follows:

High-value patents (Positive Class): Patents with a HeXiang value score = 10, legal status = granted, and validity = valid.

Non-high-value patents (Negative Class): Patents with a HeXiang value score = 1, and legal status indicating invalidity, withdrawal, abandonment, or lapse.

Usage Instructions
Environment Setup
We recommend setting up an isolated Python environment using Conda:

Bash
# Create and activate a new virtual environment
conda create -n bdpcnn_env python=3.9
conda activate bdpcnn_env

# Install core dependencies
pip install tensorflow transformers pandas numpy scikit-learn
Running the Implementation Pipeline
Download the pre-trained bert-base-chinese model parameters from Hugging Face and place them in your local path.

Update the file directory paths (train_file, dev_file, test_file, and bert_dir) at the beginning of the qgbertdpcnnwsupcon.py file to match your local setup.

Execute the script via your terminal:

Bash
python qgbertdpcnnwsupcon.py
Expected Outputs
Upon running, the script executes the training batches, prints joint training/validation losses alongside classification metrics (Accuracy, Precision, Recall, F1-score) per epoch, saves the optimal weights to the designated path, and exports the final test set inference probabilities into test_results.csv.

Requirements
Operating System / Hardware: Python 3.8+ (Tested running successfully on Python 3.9 with GPU support).

Core Libraries:

tensorflow >= 2.6.0

transformers >= 4.18.0

pandas

numpy

scikit-learn

Methodology
The framework implements high-value patent identification through the following major steps:

Data Loading: Loads the text sequences and corresponding binary labels from the .tsv datasets.

Tokenization: Tokenizes and encodes the inputs into standard BERT-compatible input tokens (input_ids, attention_mask, token_type_ids).

Multi-Scale Feature Fusion: Feeds sequence outputs through a parallel Deep Pyramid Convolutional Neural Network (DPCNN) architecture with block kernels (sizes 2, 3, 4) using Mish Activation to extract local phrase-level semantic features.

Normalized Embedding: Projects representations into a 256-dimensional space with L2-normalization.

Joint Dynamic Optimization: Computes a joint objective utilizing Binary Cross-Entropy Loss and our custom Dynamic Weighted Supervised Contrastive Loss (DW-SupCon), where easy/hard pair mining weights adaptively scale across epochs.

Evaluation: Outputs comprehensive performance metrics (Accuracy, F1-Score, Precision, Recall) evaluated on the test set.

Citation
If you use this code, the methodology, or the experimental framework in your research, please cite the corresponding article:

Plaintext
[Insert the Full Standard Citation / DOI of your Journal Paper here once published]
Data Sources
IncoPat Global Patent Database: Third-party proprietary platform source for the underlying lithography patent records (https://www.incopat.com/).

License & Contributions
License: This project is licensed under the MIT License terms.

Contributions: This repository serves to support scientific transparency and exact mathematical reproducibility. Bug reports, optimization suggestions, and issues are welcome, provided updates align with the primary parameters reported in the paper.
