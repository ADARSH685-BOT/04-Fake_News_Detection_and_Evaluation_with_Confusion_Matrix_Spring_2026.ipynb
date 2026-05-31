<div align="center">

<img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-green?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/NLP-TF--IDF-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>

</div>

---

<div align="center">

# 🕵️ Fake News Detection & Evaluation with Confusion Matrix

### IDEAS Summer Internship Program 2026
**Institute of Data Engineering, Analytics and Science Foundation**

> *Can a machine tell real news from fake? This project proves it can — with 99.5% accuracy.*

</div>

---

## 📌 Project Overview

Misinformation spreads faster than truth. This project builds a complete **end-to-end NLP pipeline** to automatically classify news articles as **Real** or **Fake** using classical machine learning models trained on a large labeled dataset of ~45,000 articles.

The pipeline covers everything from raw text ingestion to model evaluation with confusion matrices, precision-recall analysis, and hyperparameter tuning.

---

## 🎯 Objectives

- ✅ Load and explore a real-world fake news dataset (~45K articles)
- ✅ Clean and preprocess raw news text
- ✅ Convert text to numerical features using **TF-IDF**
- ✅ Train and compare **4 classification models**
- ✅ Evaluate with Accuracy, Precision, Recall, F1 Score
- ✅ Visualise **Confusion Matrices** for all models
- ✅ Tune hyperparameters using **GridSearchCV**
- ✅ Select and justify the best model

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) |
| **Files** | `Fake.csv` + `True.csv` |
| **Total Articles** | ~44,898 |
| **Fake Articles** | 23,481 (sourced from Politifact-identified unreliable sites) |
| **True Articles** | 21,417 (sourced from Reuters.com) |
| **Topic Focus** | Politics & World News |
| **Columns** | `title`, `text`, `subject`, `date` |

> ⚠️ **Dataset not included** in this repo due to size. Download from the Kaggle link above and place `Fake.csv` and `True.csv` inside a `News_dataset/` folder.

---

## 🏗️ Project Pipeline

```
Raw CSVs (Fake.csv + True.csv)
        │
        ▼
  Label Encoding  →  class: 1 (Fake), 0 (True)
        │
        ▼
  Merge + Shuffle  →  44,898 rows combined
        │
        ▼
  Data Cleaning  →  Drop title, subject, date
        │
        ▼
  Text Preprocessing (wordopt)
    • Lowercase
    • Remove URLs
    • Remove non-alphanumeric chars
    • Collapse whitespace
        │
        ▼
  TF-IDF Vectorisation  →  104,782 features
        │
        ▼
  Train / Test Split  →  75% train | 25% test
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Model Training & Evaluation        │
  │  • Logistic Regression              │
  │  • Decision Tree                    │
  │  • Naive Bayes                      │
  │  • Random Forest                    │
  └─────────────────────────────────────┘
        │
        ▼
  GridSearchCV Hyperparameter Tuning
        │
        ▼
  Confusion Matrix + Metrics Report
```

---

## 📈 Model Performance Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.9819 | 0.9851 | 0.9802 | 0.9826 |
| 🏆 **Decision Tree** | **0.9949** | **0.9942** | **0.9961** | **0.9951** |
| Naive Bayes | 0.9320 | 0.9325 | 0.9377 | 0.9351 |
| Random Forest | 0.9890 | 0.9925 | 0.9865 | 0.9895 |

### 🏆 Best Model: Decision Tree
> Decision Tree achieved the highest F1 Score of **0.9951**, making it the most balanced and accurate model for this task. It correctly identifies both fake and real news with minimal false positives and false negatives.

**GridSearchCV Best Parameters for Logistic Regression:**
```python
{'C': 10, 'solver': 'liblinear'}
```

---

## 🗂️ Repository Structure

```
📁 04-Fake_News_Detection_and_Evaluation_with_Confusion_Matrix_Spring_2026/
│
├── 📓 04-Fake_News_Detection_and_Evaluation_with_Confusion_Matrix_Spring_2026.ipynb
│                          ↳ Main project notebook (all code + markdown)
│
├── 📁 News_dataset/        ← Download from Kaggle (not included)
│   ├── Fake.csv
│   └── True.csv
│
├── 🖼️ model_comparison.png      ← Generated on notebook run
├── 🖼️ confusion_matrices.png    ← Generated on notebook run
│
└── 📄 README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ADARSH685-BOT/04-Fake_News_Detection_and_Evaluation_with_Confusion_Matrix_Spring_2026.ipynb.git
cd 04-Fake_News_Detection_and_Evaluation_with_Confusion_Matrix_Spring_2026.ipynb
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn matplotlib numpy
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) and place `Fake.csv` + `True.csv` in a `News_dataset/` folder.

### 4. Update the file paths in the notebook
In **Question 1**, update the paths to match your local dataset location:
```python
fake_df = pd.read_csv('News_dataset/Fake.csv')
true_df = pd.read_csv('News_dataset/True.csv')
```

### 5. Run the notebook
Open in Jupyter Notebook or JupyterLab and run all cells top to bottom.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas | Data loading & manipulation |
| Scikit-learn | ML models, TF-IDF, metrics |
| Matplotlib | Visualisations & confusion matrices |
| NumPy | Numerical operations |
| Jupyter Notebook | Interactive development environment |

---

## 📋 Notebook Structure

| Section | Description | Marks |
|---------|-------------|-------|
| Q1 | Import libraries & load data | 2 |
| Q2 | Create class label column | 1 |
| Q3 | Merge & shuffle DataFrames | 2 |
| Q4 | Data cleaning (drop columns) | 2 |
| Q5 | Text preprocessing function `wordopt()` | 5 |
| Q6 | Feature/target split + train-test split | 2 |
| Q7 | TF-IDF vectorisation | 5 |
| Q8 | Logistic Regression + classification report | 3 |
| Q9 | Decision Tree + accuracy score | 3 |
| Q10 | GridSearchCV hyperparameter tuning | 5 |
| Extra | Naive Bayes, Random Forest, comparison, confusion matrices | — |

---

## 👨‍💻 Author

<div align="center">

**Adarsh Kumar**
Summer Internship 2026 — IDEAS Foundation

[![GitHub](https://img.shields.io/badge/GitHub-ADARSH685--BOT-black?style=flat-square&logo=github)](https://github.com/ADARSH685-BOT)

</div>

---

<div align="center">

*Built with ❤️ for the IDEAS Summer Internship Program 2026*

</div>
