# Setup & Execution Guide

This guide covers everything needed to run the project end-to-end on a local machine.

---

## System Requirements

- **Python:** 3.10 or higher
- **RAM:** 16 GB recommended (the Random Forest notebook is memory-intensive — 3-fold CV was used in place of 5-fold for this reason)
- **OS:** Windows, macOS, or Linux
- **Ollama:** Required for Notebooks 4 and 6 (see Step 3 below)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/FBlake28/motor-insurance-prediction.git
cd motor-insurance-prediction
```

---

## Step 2: Install Python Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib statsmodels scipy tqdm ollama ipython
```

Or by using the `requirements.txt` file included:

```bash
pip install -r requirements.txt
```

---

## Step 3: Install and Configure Ollama

Ollama is a free LLM model that runs locally. This model was chosen due to its flexible setup and low operational cost.

Notebooks 4 and 6 require [Ollama](https://ollama.com) running locally with the `llama3.2` model pulled. Without this, both notebooks will throw a connection error.

**Install Ollama:** Download from [https://ollama.com/download](https://ollama.com/download)

**Pull the model:**
```bash
ollama pull llama3.2
```

**Start the service** (must be running before executing either notebook):
```bash
ollama serve
```

> If Ollama is already running as a background service after installation, `ollama serve` may not be necessary. You can verify by running `ollama list` — if it returns without error, the service is active.

---

## Step 4: Download the Dataset

The dataset is not included in this repository as it is third-party data.

1. Go to [https://data.mendeley.com/datasets/5cxyb5fp4f/2](https://data.mendeley.com/datasets/5cxyb5fp4f/2)
2. Download the file
3. Place it in the root project directory (same folder as the notebooks)
4. Rename it exactly to:

```
Motor vehicle insurance data.csv
```

> The dataset is semicolon-delimited and Excel-wrapped. Do not open and re-save it in Excel before use — this can alter the delimiter and break the loading step.

---

## Step 5: Run the Notebooks

All notebooks must be run **in order** from the same working directory. Each notebook saves `.pkl` artifact files that downstream notebooks depend on.

| Order | File | Description |
|---|---|---|
| 1 | `01. Data Loading and Cleaning.ipynb` | Preprocessing, feature engineering, PCA, K-Means clustering |
| 2 | `02. Multiple Linear Regression.ipynb` | MLR premium prediction + logistic regression claim classification |
| 3 | `03. Random Forest.ipynb` | RF premium prediction + RF claim classification |
| 4 | `04. LLM.ipynb` | Few-shot LLM prediction via Ollama *(requires Ollama running)* |
| 5 | `05. Comparison and Conclusion.ipynb` | Three-way model comparison, ROC curves, confusion matrices |
| 6 | `06. Agent Demo.ipynb` | LLM agent with tool calling — input any policy in plain English *(requires Ollama running)* |

> **Do not skip notebooks.** Each one writes `.pkl` files (trained models, scalers, metrics, split indices) that later notebooks load. Running out of order will produce `FileNotFoundError`.

---

## Expected Runtime

| Notebook | Approximate Runtime |
|---|---|
| 01 | <1 minute |
| 02 | 1-2 minutes |
| 03 | 5-10 minutes *(RF training is the bottleneck)* |
| 04 | ~10 minutes *(depends on Ollama throughput for 300 rows × 2 tasks)* |
| 05 | 1–2 minutes |
| 06 | ~1 minutes per agent call |

---

## Troubleshooting

**`FileNotFoundError: Motor vehicle insurance data.csv`**
The dataset filename must match exactly, including capitalisation and spacing. Verify the file is in the same directory as the notebooks.

**`ConnectionError` or `ollama` not found**
Ollama is not running or not installed. Complete Step 3 before running Notebooks 4 or 6.

**Kernel crash in Notebook 3**
This is a RAM issue. Try closing other applications, reducing `n_jobs` to `1` in the `RandomForestRegressor` call, or reducing `max_samples` further (e.g. `0.5`).

**`KeyError` or `FileNotFoundError` on a `.pkl` file**
A previous notebook did not complete successfully. Re-run all notebooks from the beginning in order.

**`ModuleNotFoundError`**
A required package is missing. Re-run the `pip install` command in Step 2.
