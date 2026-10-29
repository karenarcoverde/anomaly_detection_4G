# Anomaly Detection in 4G Cellular Networks

## Problem Description

This project addresses Kaggle's [Anomaly Detection in 4G Cellular Networks](https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/discussion?sort=recent-comments) competition. Cellular networks are often provisioned to handle peak demand, which can leave radio resources underused at other times. Predicting changes in user traffic can help an operator manage energy and frequency resources more efficiently. The goal of this problem is to detect unusual network activity that could motivate reconfiguring a base station.

Each row represents a sample from one cell at a particular time. The data comes from a real LTE deployment: measurements were collected every 15 minutes across base stations, each with a different number of cells. The training label `Unusual` distinguishes normal activity (`0`), consistent with the behavior typically observed for that time of day, from unusual activity (`1`), which deviates from that behavior. Unusual activity may be associated with events such as strikes, demonstrations, or sports events and may warrant a change in base-station configuration.

The solution explores the data, converts time into a numeric feature, and compares classification models using hyperparameter search and cross-validation. Since unusual observations may be less frequent, evaluation includes F1, ROC-AUC, PR-AUC, precision and recall. The classification threshold is selected based on the F1 score from out-of-fold predictions. The practical goal is to produce a classification for each row in the test set in the competition's submission format.

### Column descriptions

`UL` means uplink (device-to-network traffic), `DL` means downlink (network-to-device traffic), and `UE` means user equipment, such as a mobile device. Traffic and resource-usage metrics describe the 15-minute measurement interval. The CSV files contain the following columns:

| Column | Description |
| --- | --- |
| `Time` | Time of day when the sample was generated, in `HH:MM` format. |
| `CellName` | Text identifier of the cell that generated the sample. The identifier encodes the base station and the cell within it. |
| `PRBUsageUL` | Percentage of uplink Physical Radio Blocks (PRBs) in use during the previous 15 minutes. |
| `PRBUsageDL` | Percentage of downlink Physical Radio Blocks (PRBs) in use during the previous 15 minutes. |
| `meanThr_DL` | Average downlink traffic carried during the previous 15 minutes, in Mbps. |
| `meanThr_UL` | Average uplink traffic carried during the previous 15 minutes, in Mbps. |
| `maxThr_DL` | Maximum downlink traffic measured during the previous 15 minutes, in Mbps. |
| `maxThr_UL` | Maximum uplink traffic measured during the previous 15 minutes, in Mbps. |
| `meanUE_DL` | Average number of user equipment (UE) devices simultaneously active for downlink during the previous 15 minutes. |
| `meanUE_UL` | Average number of UE devices simultaneously active for uplink during the previous 15 minutes. |
| `maxUE_DL` | Maximum number of UE devices simultaneously active for downlink during the previous 15 minutes. |
| `maxUE_UL` | Maximum number of UE devices simultaneously active for uplink during the previous 15 minutes. |
| `maxUE_UL+DL` | Maximum number of UE devices simultaneously active during the previous 15 minutes, regardless of uplink or downlink. |
| `Unusual` | Training target: `1` for an unusual observation and `0` otherwise. This column is not present in the test file. |

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

The notebook is for local project execution only. Do not include or upload a notebook as part of the competition submission; submit only the required CSV file.

## Environment setup

Use Python 3.9 or newer. From the project folder, create and activate a virtual environment, then install the dependencies:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When the environment is active, its name (usually `.venv`) appears in the terminal prompt. To leave it, run:

```bash
deactivate
```

## Train and save the final model with BentoML

With the virtual environment active and the training CSV in the project folder, run:

```bash
python train.py
```

By default, the script reads `ML-MATT-CompetitionQT2021_train.csv` from the project folder. To use a different location or BentoML model name:

```bash
python train.py --data path/to/ML-MATT-CompetitionQT2021_train.csv --model-name anomaly_detection_4g_lgbm
```

The script applies the notebook's data preparation, chooses the classification threshold from three-fold out-of-fold predictions, then fits the selected LightGBM configuration on all cleaned training rows. BentoML saves the trained model in its local model store; the command prints the resulting model tag. The saved model also includes the chosen threshold and ordered feature names as custom objects.

## Serve predictions with BentoML

After training the model, start the BentoML API from the project folder:

```powershell
bentoml serve predict:service --reload
```

The service loads the latest model named `anomaly_detection_4g_lgbm` from
BentoML's local model store. If you trained with a different `--model-name`,
set `BENTOML_MODEL_TAG` to that model's tag before starting the service, for
example:

```powershell
$env:BENTOML_MODEL_TAG = "my_anomaly_model:latest"
bentoml serve predict:service --reload
```

Upload a CSV file as a multipart file to `http://localhost:3000/predict`. The
response is the original CSV with predictions appended in the `Unusual`
column. For example, in PowerShell with `curl.exe`:

```powershell
curl.exe -X POST http://localhost:3000/predict -F "file=@ML-MATT-CompetitionQT2021_test.csv" -o ML-MATT-CompetitionQT2021_test_predicted.csv
```