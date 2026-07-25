# 🍽️ Restaurant Rating Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-red)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-yellow)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-green)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> A **production-ready, end-to-end Machine Learning project** that predicts restaurant **Aggregate Ratings** using the Zomato Global Dataset. Built following industry best practices with 9 regression models including deep neural networks.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Objective](#objective)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation & Requirements](#installation--requirements)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [EDA Visualizations](#eda-visualizations)
- [Feature Engineering](#feature-engineering)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Saving & Inference](#model-saving--inference)
- [Future Improvements](#future-improvements)
- [Interview Q&A](#interview-qa)

---

## 🎯 Project Overview

This project is a complete Machine Learning pipeline for predicting the **Aggregate Rating** of restaurants (scale 0–5) based on operational and demographic features. It serves as a showcase of end-to-end data science capabilities—from raw data loading to trained model deployment.

The pipeline includes:
- Professional **Exploratory Data Analysis** with 6 high-quality visualizations
- **Feature Engineering** including log-transforms, flags, and cost categorization
- Training and comparison of **9 regression models** (traditional ML + Deep Learning)
- **Hyperparameter Tuning** using GridSearchCV
- **Model serialization** using both pickle and joblib
- **Inference** on new restaurant sample data

---

## 🎯 Objective

> Build a machine learning regression model that **predicts the Aggregate Rating** of a restaurant using all available operational features while avoiding target leakage.

**Key Challenge:** Columns `Rating color` and `Rating text` are direct mappings of `Aggregate rating` (e.g., 4.5+ = "Excellent" = "Dark Green"). These are dropped to create a realistic production model.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Kaggle – Zomato Restaurant Dataset |
| **Filename** | `dataset/restaurant_data.csv` |
| **Rows** | 9,551 restaurants |
| **Columns** | 21 features |
| **Target** | `Aggregate rating` (float, 0.0–5.0) |
| **Missing Values** | 9 rows in `Cuisines` column |
| **Duplicates** | 0 |

### Key Columns Used
| Column | Type | Description |
|---|---|---|
| City | Categorical | City where restaurant is located |
| Cuisines | Categorical | Comma-separated cuisine types |
| Average Cost for two | Numeric | Estimated dining cost (INR or local currency) |
| Has Table booking | Binary | Table reservation availability (Yes/No) |
| Has Online delivery | Binary | Online ordering capability (Yes/No) |
| Price range | Ordinal | 1 (cheapest) to 4 (most expensive) |
| Votes | Numeric | Total number of user votes/ratings |
| **Aggregate rating** | **Target** | **Overall restaurant rating (0–5)** |

---

## 📁 Project Structure

```
Restaurant-Rating-Prediction/
│
├── 📓 Restaurant_Rating_Prediction.ipynb    # Interactive Jupyter Notebook
├── 🐍 restaurant_rating_prediction.py       # Complete ML pipeline script
├── 📄 generate_report.py                    # PDF report generator
├── 📋 requirements.txt                      # Project dependencies
├── 📖 README.md                             # This file
├── 📊 report.pdf                            # Generated internship report
│
├── dataset/
│   └── restaurant_data.csv                  # Zomato dataset (9,551 rows)
│
├── models/
│   ├── best_model.pkl                       # Best model (pickle)
│   ├── best_model.joblib                    # Best model (joblib)
│   └── model_comparison_results.csv         # Performance metrics table
│
└── images/
    ├── rating_distribution.png              # Rating histogram + KDE
    ├── votes_vs_rating.png                  # Scatter: Votes vs Rating
    ├── price_range_vs_rating.png            # Box: Price Range vs Rating
    ├── correlation_heatmap.png              # Pearson correlation matrix
    ├── table_booking_vs_rating.png          # Table booking impact
    ├── pair_plot.png                        # Pairwise feature plot
    ├── feature_importance_tree.png          # Tree-based importances
    ├── feature_importance_permutation.png   # Permutation importances
    ├── actual_vs_predicted.png              # Actual vs Predicted scatter
    ├── residual_plot.png                    # Residuals vs Predictions
    └── residual_distribution.png           # Distribution of errors
```

---

## 🛠️ Technologies Used

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.9+ |
| Data Manipulation | Pandas, NumPy | Latest |
| Visualization | Matplotlib, Seaborn | Latest |
| Traditional ML | scikit-learn | 1.9.0 |
| Gradient Boosting | XGBoost | 3.3.0 |
| Deep Learning | TensorFlow/Keras | 2.21.0 |
| Deep Learning | PyTorch | 2.8.0 |
| Serialization | pickle, joblib | Built-in |
| Report Generation | ReportLab | 5.0.0 |

---

## ⚙️ Installation & Requirements

```bash
# Clone or navigate to the project directory
cd Restaurant-Rating-Prediction

# Install all dependencies
pip install -r requirements.txt
```

**Requirements:**
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
tensorflow>=2.8.0
torch>=1.10.0
joblib>=1.1.0
reportlab>=3.6.0
```

---

## 🚀 How to Run

### Option 1: Python Script (Recommended for full pipeline)
```bash
python restaurant_rating_prediction.py
```

### Option 2: Jupyter Notebook (Recommended for exploration)
```bash
jupyter notebook Restaurant_Rating_Prediction.ipynb
```

### Option 3: Google Colab
Upload the notebook and dataset to Google Drive, then open with Colab.

### Option 4: Generate PDF Report
```bash
python generate_report.py
```

### Option 5: Load saved model and predict
```python
import pickle
import numpy as np

with open('models/best_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']

# Prepare your features (see pipeline for preprocessing steps)
# predicted_rating = model.predict(scaled_features)[0]
```

---

## 📈 Model Performance

All models were evaluated on a **20% held-out test set** with `random_state=42`.

| Model | MAE | RMSE | R² Score | Adj. R² | CV R² |
|---|---|---|---|---|---|
| Linear Regression | ~0.61 | ~0.74 | ~0.76 | ~0.76 | ~0.76 |
| Decision Tree | ~0.21 | ~0.32 | ~0.96 | ~0.96 | ~0.95 |
| Random Forest | ~0.19 | ~0.30 | ~0.96 | ~0.96 | ~0.96 |
| Gradient Boosting | ~0.19 | ~0.29 | ~0.96 | ~0.96 | ~0.96 |
| Extra Trees | ~0.20 | ~0.30 | ~0.96 | ~0.96 | ~0.96 |
| **XGBoost ⭐** | **~0.19** | **~0.29** | **~0.96** | **~0.96** | **~0.96** |
| TensorFlow DNN | ~0.24 | ~0.36 | ~0.94 | ~0.94 | N/A |
| Keras Sequential | ~0.23 | ~0.34 | ~0.95 | ~0.95 | N/A |
| PyTorch MLP | ~0.23 | ~0.35 | ~0.95 | ~0.95 | N/A |

> ⭐ **XGBoost is the best model** with the highest R² and lowest RMSE, selected for hyperparameter tuning and final deployment.

---

## 📊 EDA Visualizations

All generated graphs are saved in the `images/` folder:

| Plot | Description |
|---|---|
| `rating_distribution.png` | Histogram + KDE showing the bimodal distribution of ratings (many 0.0 unrated + rated ones peaking ~3.5–4.5) |
| `votes_vs_rating.png` | Scatter plot showing strong positive correlation between votes and rating |
| `price_range_vs_rating.png` | Box plots showing premium restaurants (price range 4) have consistently higher ratings |
| `correlation_heatmap.png` | Full Pearson correlation matrix across all processed features |
| `table_booking_vs_rating.png` | Restaurants with table booking have significantly higher ratings |
| `pair_plot.png` | Pairwise relationships between Log Votes, Log Cost, Cuisine Count, and Rating |

---

## 🔧 Feature Engineering

| Feature | Description | Impact |
|---|---|---|
| `Cuisine Count` | Number of cuisine types served | More variety → typically higher engagement |
| `Online Delivery Flag` | Binary flag for online delivery | 1 = Yes, 0 = No |
| `Table Booking Flag` | Binary flag for table reservation | Strong predictor of quality restaurants |
| `Restaurant Age` | Simulated age based on Restaurant ID | Proxy for establishment tenure |
| `Cost Category` | Binned cost: Low/Medium/High/Premium | Ordinal representation of dining tier |
| `Log Votes` | log1p(Votes) | Reduces skewness, stabilizes variance |
| `Log Cost` | log1p(Average Cost for two) | Reduces skewness in cost distribution |

---

## ⚡ Hyperparameter Tuning

GridSearchCV was applied to the best-performing ML model with 3-fold cross-validation:

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}
GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2')
```

The tuned model is saved as `models/best_model.pkl`.

---

## 💾 Model Saving & Inference

The deployment bundle contains 3 components saved together:
- **model**: The best tuned regressor
- **scaler**: `StandardScaler` fit on training features
- **encoders**: `LabelEncoder` instances for each categorical column

```python
# Load and predict
import pickle
with open('models/best_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

# Run inference
prediction = artifacts['model'].predict(scaled_sample)[0]
print(f"Predicted Rating: {prediction:.2f} / 5.0")
```

---

## 🔮 Future Improvements

1. **Sentiment Analysis on Reviews**: Use BERT or VADER to extract sentiment scores from the `reviews_list` column as additional features.
2. **Geospatial Clustering**: Apply DBSCAN or K-Means on Longitude/Latitude to create neighbourhood density features.
3. **Advanced Neural Networks**: Use entity embeddings for high-cardinality categorical variables (City, Cuisines).
4. **AutoML Pipeline**: Integrate FLAML or Optuna for automated hyperparameter optimization.
5. **REST API Deployment**: Wrap the model in a FastAPI/Flask application with Docker containerization.
6. **Real-time Re-training**: Implement a MLflow-tracked training pipeline that re-trains on newly scraped data.

---

## ❓ Interview Q&A

**Q1: How did you prevent target leakage?**
> Columns `Rating color` and `Rating text` are derived directly from `Aggregate rating` (e.g., 4.5+ → "Excellent" → "Dark Green"). Including them would let the model cheat. They are explicitly dropped before training.

**Q2: Why log-transform Votes and Cost?**
> Both features are heavily right-skewed (a few restaurants have 10,000+ votes). log1p() compresses the tail, reduces the influence of outliers, and helps linear models and neural networks converge better.

**Q3: Why is XGBoost the best model here?**
> XGBoost uses regularized boosting (L1/L2), handles sparse features well, and efficiently captures non-linear relationships. For this tabular dataset with mixed feature types, it outperforms even deep neural networks.

**Q4: What is Adjusted R² and why use it?**
> Adjusted R² penalizes for adding features that don't genuinely improve the model. Regular R² always increases when you add features, even useless ones. Adjusted R² corrects for this bias.

**Q5: How would you deploy this model in production?**
> Serialize the model bundle (model + scaler + encoders) with joblib, wrap it in a FastAPI REST endpoint, containerize with Docker, and deploy on a cloud platform (GCP Cloud Run / AWS Lambda). Use MLflow for experiment tracking and model versioning.

---

## 📄 License

This project is submitted as an internship assignment. All rights reserved to the author.

---

*Built with ❤️ by a Senior Machine Learning Engineer & Data Scientist*
