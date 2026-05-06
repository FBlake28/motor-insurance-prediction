# BIA 652 - Multivariate Data Analysis
This project began as a final project requirement for a course at Stevens Institute of Technology and was extended beyond course requirements.

---

## Research Question
> *What vehicle and driver characteristics are the strongest predictors of motor insurance premium and claim occurrence — and how does a traditional statistical model, an ensemble ML model, and an LLM compare in their ability to make that prediction?*

In the age of GenAI, there tends to be a consistent "overhype" among AI enthusiasts in the ability of GenAI to perform as well as traditional ML models with little-to-no situational fine-tuning. While LLMs are powerful, they are limited by their token count and require external tools-- like traditional ML models-- to reach their full "predictive" potential. This project seeks to quantify the gap in model success, determine model specialties, and exemplify a successful GenAI system using function calling. This project also seeks to create a functional system to estimate insurance premium cost and claim likelihood based on the provided training data.

---

## Dataset
**Spanish Motor Vehicle Insurance Portfolio (2015–2018)**
105,555 rows × 30 columns — semicolon-delimited, Excel-wrapped.
Source: [Mendeley Data](https://data.mendeley.com/datasets/5cxyb5fp4f/2)

Note that this dataset originates from Spain, meaning that our attributes will be European standard. This include date-time formatting, currency type, distance and size measurements, etc.

---

## Workflow Overview

| Phase | Description |
|---|---|
| **1. EDA & Preprocessing** | Exploratory analysis, feature engineering, reusable preprocessing pipeline |
| **2. Dimensionality Reduction & Segmentation** | PCA on vehicle specs → K-Means clustering with human-readable risk tier labels |
| **3. Predictive Modeling** | Three parallel models predicting Premium and claim occurrence |
| &nbsp;&nbsp;&nbsp;h1 — Multiple Linear Regression | Traditional statistical model |
| &nbsp;&nbsp;&nbsp;h2 — Random Forest | Ensemble ML baseline |
| &nbsp;&nbsp;&nbsp;h3 — LLM + Prompt Engineering | Ollama with structured prompting |
| **4. Model Comparison** | R², RMSE, ROC-AUC across all three approaches |
| **5. Agent Demo** | Using wrapped tools via the LLM tool calling capabilities-- Input your own policy description to receive text-delivered trained model outputs! |

---

======================================================================
Regression Comparison — Premium Prediction (Log_premium target)
======================================================================
                                 R² (test, log)  RMSE (log)   RMSE (€)  Test n
Model                                                                         
Multiple Linear Regression (h2)          0.4869      0.3103   119.1657   20759
Random Forest (h1)                       0.7175      0.2302    86.3081   20759
LLM (llama3.2)                          -2.4581      0.7834  2746.9726     300

Mean test premium for context: €318.45

---

======================================================================
  CLASSIFICATION COMPARISON (MLR & RF: balanced models)
======================================================================
                                  ROC-AUC Recall (Claims) Precision F1-Score  Accuracy  Test n Imbalance corrected
Model                                                                                                             
Multiple Linear Regression (h2)  0.919000          0.7931     0.584   0.6727    0.8550   20759                 Yes
Random Forest (h1)               0.872900          0.5505     0.489   0.5179    0.8075   20759                 Yes
LLM (llama3.2)                   0.682194             N/A       N/A      N/A    0.7900     300                 N/A

  Primary metric: ROC-AUC | Secondary: Recall (insurance — missed claims costly)
  Accuracy de-emphasized — inflated by 4.4:1 class imbalance

 ---

 ## Takeaways:
 In this project, we had two goals: predict insurance premium cost (regression), and predict whether a claim would occur within a year (classification).

Random Forest (h1) wins regression with the highest R-squared value of 0.728. RF is able to capture nonlinear interactions, such as with N_doors that allowed it to pick up on 0-door rows belonging to motorbikes and adjusting the predicted premium value accordingly.

Multiple Linear Regression (h2) wins classification with AUC = 0.915. While RF has slightly higher recall, the MLR is still able to more consistently classify whether a row will result in a claim.

Meanwhile, the LLM approach consistently underperforms both classical models for both classification and regression tasks. There are numerous reasons for this occurring: "training data" for the LLM consisted of few-shot prompts rather than standard training, the LLM model may not have been advanced enough, and a heavy bias towards predicting a claim would result in "no claim." This isn't a failure of LLMs in general, but instead demonstrates the need for LLM's use as a connecting tool rather than a primary classification / regression methodology. By giving LLMs access to tools like RF and MLR via function calling capabilities, LLMs can instead be used to intake and explain results in "plain english" for easy understanding.

A potential project extension could be doing the above and designing an application to assist with improving understanding of RF and MLR conclusions via an LLM-based AI Agent. In fact, let's attempt a rough version of such a system.
