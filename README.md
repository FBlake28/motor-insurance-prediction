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
| **5. Agent Demo** | Using wrapped tools via the LLM tool calling capabilities |

---
