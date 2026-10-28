# Anomaly Detection in 4G Cellular Networks

## Problem Description

This project addresses Kaggle's [Anomaly Detection in 4G Cellular Networks](https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/discussion?sort=recent-comments) competition. The goal is to identify unusual observations in 4G cellular network performance data. Each row represents an observation associated with a network cell and a time, along with numerical metrics. In the training set, the `Unusual` column indicates whether an observation is normal (`0`) or unusual (`1`).

The solution explores the data, converts time into a numeric feature, and compares classification models using hyperparameter search and cross-validation. Since unusual observations may be less frequent, evaluation includes F1, ROC-AUC, PR-AUC, precision and recall. The classification threshold is selected based on the F1 score from out-of-fold predictions. The practical goal is to produce a classification for each row in the test set in the competition's submission format.

### Column descriptions

`UL` means uplink (device-to-network traffic), `DL` means downlink (network-to-device traffic), and `UE` means user equipment, such as a mobile device. The CSV files contain the following columns:

| Column | Description |
| --- | --- |
| `Time` | Observation time in `HH:MM` format. |
| `CellName` | Identifier of the cellular network cell associated with the observation. |
| `PRBUsageUL` | Physical Resource Block (PRB) usage for uplink traffic. |
| `PRBUsageDL` | Physical Resource Block (PRB) usage for downlink traffic. |
| `meanThr_DL` | Mean downlink throughput. |
| `meanThr_UL` | Mean uplink throughput. |
| `maxThr_DL` | Maximum downlink throughput. |
| `maxThr_UL` | Maximum uplink throughput. |
| `meanUE_DL` | Mean number of UEs associated with downlink traffic. |
| `meanUE_UL` | Mean number of UEs associated with uplink traffic. |
| `maxUE_DL` | Maximum number of UEs associated with downlink traffic. |
| `maxUE_UL` | Maximum number of UEs associated with uplink traffic. |
| `maxUE_UL+DL` | Combined maximum UE count for uplink and downlink, as provided in the dataset. |
| `Unusual` | Training target: `1` for an unusual observation and `0` otherwise. This column is not present in the test file. |

The notebook does not specify the measurement units for the throughput or PRB usage fields; use the competition's data description for their exact units.

## Data

The competition data files are not included in this repository. To obtain them:

1. Sign in to the [Kaggle competition page](https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/discussion?sort=recent-comments) and accept the competition rules to enable downloads.
2. Download the data from the competition's **Data** section.
3. Extract these two files into the same folder as `notebook.ipynb`:

	- `ML-MATT-CompetitionQT2021_train.csv`
	- `ML-MATT-CompetitionQT2021_test.csv`

The notebook reads the CSV files using `;` as the delimiter. The training set contains the `Unusual` label; the test set does not. Do not redistribute the competition data if Kaggle's rules prohibit it.

You can also download the files using the Kaggle CLI after installing it and configuring your credentials as described in the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api):

```bash
kaggle competitions download -c anomaly-detection-in-4g-cellular-networks
```

Extract the downloaded ZIP file into the project folder. If the competition is unavailable to your account through the CLI, download the data from the competition page after accepting the rules.