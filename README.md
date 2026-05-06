# Motor Vehicle Insurance — Premium & Claims Prediction
**BIA 652 · Multivariate Data Analysis · Stevens Institute of Technology**

This project began as a final project requirement and was extended beyond course requirements.

---

## Research Question

> *What vehicle and driver characteristics are the strongest predictors of motor insurance premium and claim occurrence — and how does a traditional statistical model, an ensemble ML model, and an LLM compare in their ability to make that prediction?*

In the age of GenAI, there tends to be a consistent overhype among AI enthusiasts in the ability of GenAI to perform as well as traditional ML models with little-to-no situational fine-tuning. While LLMs are powerful, they are limited by their token count and require external tools — like traditional ML models — to reach their full predictive potential. This project seeks to quantify the gap in model success, determine model specialties, and exemplify a successful GenAI system using function calling.

---

## Dataset

**Spanish Motor Vehicle Insurance Portfolio (2015–2018)**
105,555 rows × 30 columns — semicolon-delimited, sourced from [Mendeley Data](https://data.mendeley.com/datasets/5cxyb5fp4f/2).

> **Note:** This dataset originates from Spain. All attributes follow European standards — date formatting, currency (€), engine power (kW), vehicle dimensions (mm/kg), etc.

---

## Workflow

| Phase | Description |
|---|---|
| **1. EDA & Preprocessing** | Exploratory analysis, feature engineering, reusable preprocessing pipeline |
| **2. Dimensionality Reduction & Segmentation** | PCA on vehicle specs → K-Means clustering → human-readable risk tier labels |
| **3. Predictive Modeling** | Three parallel models predicting Premium and claim occurrence |
| &nbsp;&nbsp;&nbsp;h1 — Random Forest | Ensemble ML model (best regression) |
| &nbsp;&nbsp;&nbsp;h2 — Multiple Linear Regression | Traditional statistical model (best classification) |
| &nbsp;&nbsp;&nbsp;h3 — LLM + Prompt Engineering | Ollama (llama3.2) with few-shot structured prompting |
| **4. Model Comparison** | R², RMSE, and ROC-AUC across all three approaches |
| **5. Agent Demo** | LLM tool-calling wrapper over h1 and h2 — input any policy description in plain English and receive a trained-model-backed underwriting memo |

---

## Results

### Premium Prediction (Regression)
*Target: log-transformed annual premium in €. Mean test premium: €318.45.*

| Model | R² (test) | RMSE (log) | RMSE (€) | Test n |
|---|---|---|---|---|
| Random Forest (h1) | 0.718 | 0.230 | €86.31 | 20,759 |
| Multiple Linear Regression (h2) | 0.487 | 0.310 | €119.17 | 20,759 |
| LLM — llama3.2 (h3) | −2.458 | 0.783 | €2,746.97 | 300* |

### Claim Occurrence (Classification)
*Target: binary claim flag. Balanced classifiers used for MLR and RF. Primary metric: ROC-AUC.*
*Accuracy de-emphasized — inflated by 4.4:1 class imbalance.*

| Model | ROC-AUC | Recall (Claims) | Precision | F1-Score | Accuracy | Test n |
|---|---|---|---|---|---|---|
| Multiple Linear Regression (h2) | **0.919** | 0.793 | 0.584 | 0.673 | 0.855 | 20,759 |
| Random Forest (h1) | 0.873 | 0.551 | 0.489 | 0.518 | 0.808 | 20,759 |
| LLM — llama3.2 (h3) | 0.682 | N/A | N/A | N/A | 0.790 | 300* |

*\*LLM evaluated on a stratified 300-row sample — full test set inference is infeasible at local throughput.*

---

## Takeaways

**Random Forest (h1) wins regression** with R² = 0.718. RF captures nonlinear interactions that MLR cannot — notably, N_doors = 0 rows correspond to motorcycles, which carry a systematically lower premium that RF correctly adjusts for.

**Multiple Linear Regression (h2) wins classification** with ROC-AUC = 0.919. The logistic regression decision boundary responds more cleanly to class weighting, producing superior claim-sorting ability despite RF's slight recall advantage.

**The LLM consistently underperforms** both classical models on both tasks. Its "training data" consisted of six few-shot examples rather than the full 83,000-row training set, the base model (llama3.2) was not fine-tuned for actuarial prediction, and it exhibited a strong bias toward predicting no claim. This is not a failure of LLMs in general — it demonstrates that LLMs are not designed for direct prediction tasks. Their real value is as an orchestration layer: by giving an LLM access to trained ML tools via function calling, it can parse natural language inputs and return interpretable, model-backed outputs in plain English.

The Agent Demo (Notebook 6) implements exactly this pattern.

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally with `llama3.2` pulled.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib statsmodels scipy tqdm ollama
ollama pull llama3.2
```

**Run notebooks in order:**
