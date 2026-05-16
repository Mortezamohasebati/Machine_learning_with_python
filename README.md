# 🤖 Machine Learning with Python

A collection of end-to-end **Machine Learning classification projects** built with Python and scikit-learn, covering two real-world healthcare and environmental datasets. Each notebook walks through the complete ML pipeline — from raw data exploration and preprocessing all the way to training, evaluating, and comparing **7 different classification models**.

---

## 📂 Repository Structure

```
Machine_learning_with_python/
├── Diabetes_Preprocessing_7_models.ipynb        # Diabetes prediction — full ML pipeline
├── water_potability_prediction_7_models.ipynb   # Water potability prediction — full ML pipeline
└── water_potability.csv                         # Dataset for the water potability project
```

---

## 📓 Projects Overview

### 🩺 Project 1 — Diabetes Prediction

**Notebook:** `Diabetes_Preprocessing_7_models.ipynb`

Predicts whether a patient is diabetic based on diagnostic measurements. Uses the well-known **Pima Indians Diabetes Dataset**.

#### Dataset Features

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration (2-hour OGTT) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight/height²) |
| `DiabetesPedigreeFunction` | Diabetes pedigree function (genetic risk) |
| `Age` | Age in years |
| **`Outcome`** | **Target — 1: Diabetic, 0: Non-diabetic** |

#### Pipeline Steps
- Exploratory Data Analysis (EDA) — distributions, correlations, class balance
- Handling biologically invalid zero values (e.g., zero Glucose, zero BMI) → replaced with `NaN`
- Missing value imputation (mean/median)
- Feature scaling — `StandardScaler` / `MinMaxScaler`
- Train/test split
- Training and evaluating 7 classification models
- Comparing model performance with metrics and visualizations

---

### 💧 Project 2 — Water Potability Prediction

**Notebook:** `water_potability_prediction_7_models.ipynb`  
**Dataset:** `water_potability.csv`

Predicts whether water is **safe to drink** based on physicochemical measurements.

#### Dataset Features

| Feature | Description |
|---------|-------------|
| `ph` | pH level of water (0–14) |
| `Hardness` | Water hardness (mg/L) |
| `Solids` | Total dissolved solids (ppm) |
| `Chloramines` | Chloramines concentration (ppm) |
| `Sulfate` | Sulfate concentration (mg/L) |
| `Conductivity` | Electrical conductivity (μS/cm) |
| `Organic_carbon` | Organic carbon (ppm) |
| `Trihalomethanes` | Trihalomethanes concentration (μg/L) |
| `Turbidity` | Water turbidity (NTU) |
| **`Potability`** | **Target — 1: Potable, 0: Not potable** |

#### Pipeline Steps
- EDA — feature distributions, missing values analysis, class imbalance
- Missing value imputation
- Feature scaling and normalization
- Train/test split
- Training and evaluating 7 classification models
- Model comparison with evaluation metrics

---

## 🧠 Models Used (Both Projects)

Both notebooks train and compare the following 7 classifiers using **scikit-learn**:

| # | Model | Type |
|---|-------|------|
| 1 | **Logistic Regression** | Linear |
| 2 | **K-Nearest Neighbors (KNN)** | Instance-based |
| 3 | **Support Vector Machine (SVM)** | Kernel-based |
| 4 | **Decision Tree** | Tree-based |
| 5 | **Random Forest** | Ensemble (Bagging) |
| 6 | **Gradient Boosting** | Ensemble (Boosting) |
| 7 | **Naive Bayes** | Probabilistic |

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- **Accuracy** — overall correct predictions
- **Precision** — how many predicted positives are actually positive
- **Recall** — how many actual positives were correctly identified
- **F1-Score** — harmonic mean of Precision and Recall
- **Confusion Matrix** — visualizes TP, TN, FP, FN
- **ROC-AUC Curve** — model discrimination ability

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualization and plots |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation metrics |
| `Jupyter Notebook` | Interactive development environment |

---

## ⚙️ Setup & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Mortezamohasebati/Machine_learning_with_python.git
cd Machine_learning_with_python
```

**2. Install required packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**3. Launch Jupyter Notebook:**
```bash
jupyter notebook
```

**4. Open either notebook and run all cells** (`Kernel → Restart & Run All`)

---

## 🔄 ML Pipeline Overview

Both projects follow this standard pipeline:

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Data Preprocessing
 ├─ Handle missing / invalid values
 ├─ Feature scaling (StandardScaler / MinMaxScaler)
 └─ Train / Test split (typically 80/20)
   │
   ▼
Model Training (7 classifiers)
   │
   ▼
Evaluation & Comparison
 ├─ Accuracy, Precision, Recall, F1
 ├─ Confusion Matrix
 └─ ROC-AUC Curve
   │
   ▼
Best Model Selection
```

---

## 📚 Concepts Covered

- Binary classification
- Exploratory Data Analysis (EDA)
- Data cleaning — handling missing and invalid values
- Feature scaling — StandardScaler, MinMaxScaler
- Supervised learning — 7 classification algorithms
- Model evaluation — Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion Matrix interpretation
- Comparing multiple models on the same dataset

---

## 📜 License

This project is open source and free to use for educational and research purposes.
