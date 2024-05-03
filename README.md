# Project Title : Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit

## Overview
The aim of the original project is to find the efficient allocation of Intensive Care Unit (ICU) beds, particularly addressing the crucial task of predicting the length of stay for current ICU patients. For this reproduction study, we would like to verify the following hypothesis.

**Replicate TPC, LSTM and Transformer architectures and show that TPC approach outperforms LSTM and Transformer based models by comparing the metrics on two dataset.**

**Show that optimizing for Mean Squared Logarithmic error (MSLE) handles the positive skew in the target class as opposed to Mean Squared Error (MSE).**

Please refer to the paper for further explanation on the data source, methodology, model architecture and results.

The directory structure of this repository is shown below.
```
.
├── MIMIC_preprocessing
│   ├── README.md
│   ├── __init__.py
│   ├── create_all_tables.sql
│   ├── flat_and_labels.py
│   ├── flat_features.sql
│   ├── labels.sql
│   ├── reader.py
│   ├── run_all_preprocessing.py
│   ├── timeseries.py
│   └── timeseries.sql
├── README.md
├── README_MIMIC_data.md
├── README_eICU_data.md
├── __init__.py
├── eICU_preprocessing
│   ├── README.md
│   ├── __init__.py
│   ├── create_all_tables.sql
│   ├── diagnoses.py
│   ├── diagnoses.sql
│   ├── flat_and_labels.py
│   ├── flat_features.sql
│   ├── labels.sql
│   ├── reader.py
│   ├── run_all_preprocessing.py
│   ├── split_train_test.py
│   ├── timeseries.py
│   └── timeseries.sql
├── local_setup.sh
├── models
│   ├── MIMIC
│   │   ├── lstm.py
│   │   ├── mean_median.py
│   │   ├── tpc.py
│   │   └── transformer.py
│   ├── README.md
│   ├── __init__.py
│   ├── eICU
│   │   ├── lstm.py
│   │   ├── mean_median.py
│   │   ├── tpc.py
│   │   └── transformer.py
│   ├── experiment_template.py
│   ├── initialise_arguments.py
│   ├── lstm_model.py
│   ├── mean_median_model.py
│   ├── metrics.py
│   ├── run_lstm.py
│   ├── run_tpc.py
│   ├── run_transformer.py
│   ├── shuffle_train.py
│   ├── tpc_model.py
│   └── transformer_model.py
├── paths.json
└── requirements.txt

6 directories, 51 files
```

## Environment Setup
### eICU
Please refer to README_eICU_data.md for instructions on how to set up the eICU database.
    
### MIMIC-IV
Please refer to README_MIMIC_data.md for instructions on how to set up the MIMIC-IV database.

### Local Setup
To set up the environment locally, run the following command in your terminal:

`conda create -n python36 python=3.6`
`conda activate python36`
`pip install -r requirements.txt`
   
## Running the models
1) Once you have run the pre-processing steps you can run all the models in your terminal. Set the working directory to the TPC-LoS-prediction, and run the following:

    ```
    python -m models.eICU.mean_median
    python -m models.eICU.lstm
    python -m models.eICU.transformer
    python -m models.eICU.tpc
    python -m models.MIMIC.mean_median
    python -m models.MIMIC.lstm
    python -m models.MIMIC.transformer
    python -m models.MIMIC.tpc
    ```
    
## Hardware requirements
1. MIMIC data requires minimum of 50GB RAM to run the pre-processing steps. This takes about 12 hours to run.
2. eICU data requires minimum of 16GB RAM to run the pre-processing steps. This takes about 12 hours to run.
3. The models were trained on a V100 GPU with 16GB RAM. This takes about 16 hours to run.

## References
1. Rocheteau, E., Liò, P., & Hyland, S. (2021). Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit. Retrieved from arXiv preprint server. (Version 4) https://arxiv.org/abs/[Insert arXiv ID]

2. Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series Data. Scientific Data, 6(96), 2019.

```@inproceedings{rocheteau2021,
author = {Rocheteau, Emma and Li\`{o}, Pietro and Hyland, Stephanie},
title = {Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit},
year = {2021},
isbn = {9781450383592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3450439.3451860},
doi = {10.1145/3450439.3451860},
booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
pages = {58–68},
numpages = {11},
keywords = {intensive care unit, length of stay, temporal convolution, mortality, patient outcome prediction},
location = {Virtual Event, USA},
series = {CHIL '21}
}```
