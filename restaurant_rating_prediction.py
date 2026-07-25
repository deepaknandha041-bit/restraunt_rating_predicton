"""
Restaurant Rating Prediction Project
=====================================
Complete, production-ready, industry-level Machine Learning project.
Predicts the Aggregate Rating of restaurants using Zomato dataset.

Technologies: Python, scikit-learn, XGBoost, TensorFlow/Keras, PyTorch
Author     : Senior Machine Learning Engineer & Data Scientist
Date       : 2026-07-25
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import sys
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import (train_test_split, KFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               ExtraTreesRegressor)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================
# GLOBAL SETUP
# ============================================================
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


# ============================================================
# PART 2 — DATA LOADING & EXPLORATION
# ============================================================
def load_data(filepath="dataset/restaurant_data.csv"):
    """Load Zomato dataset, clean BOM from column names."""
    print(f"[*] Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(filepath, encoding="latin-1")
        # Strip BOM and whitespace from column names
        df.columns = (df.columns
                      .str.replace('\ufeff', '', regex=False)
                      .str.replace('ï»¿', '', regex=False)
                      .str.strip())
        print(f"[+] Loaded successfully. Shape: {df.shape}")
        return df
    except Exception as exc:
        print(f"[-] Error loading data: {exc}")
        raise


def explore_dataset(df):
    """Print profiling information about the dataset."""
    print("\n" + "="*50)
    print(" PART 2: DATA EXPLORATION ")
    print("="*50)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nData Types:\n", df.dtypes)
    missing = df.isnull().sum()
    print("\nMissing Values:\n",
          missing[missing > 0] if missing.sum() > 0 else "  None")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nSummary Statistics:\n", df.describe().T.to_string())


# ============================================================
# PART 2 & 4 — PREPROCESSING & FEATURE ENGINEERING
# ============================================================
def preprocess_and_engineer(df):
    """
    Full preprocessing pipeline:
    - Drop duplicates
    - Impute missing Cuisines
    - Engineer new features
    - Drop leakage / redundant columns
    - Label-encode categoricals
    """
    print("\n" + "="*50)
    print(" PART 2 & 4: PREPROCESSING & FEATURE ENGINEERING ")
    print("="*50)

    df_c = df.copy()

    # 1. Duplicates
    before = len(df_c)
    df_c.drop_duplicates(inplace=True)
    print(f"[+] Dropped {before - len(df_c)} duplicate rows.")

    # 2. Missing values
    df_c['Cuisines'] = df_c['Cuisines'].fillna('Unknown Cuisines')
    print("[+] Filled 'Cuisines' NaN with 'Unknown Cuisines'.")

    # 3. Feature Engineering
    df_c['Cuisine Count'] = df_c['Cuisines'].apply(
        lambda x: len(str(x).split(',')))

    df_c['Online Delivery Flag'] = (
        df_c['Has Online delivery'].map({'Yes': 1, 'No': 0}))
    df_c['Table Booking Flag'] = (
        df_c['Has Table booking'].map({'Yes': 1, 'No': 0}))

    # Simulated restaurant age from Restaurant ID
    df_c['Restaurant Age'] = 2026 - (df_c['Restaurant ID'] % 15 + 2010)

    def _cost_bucket(cost):
        if cost <= 300:   return 'Low'
        if cost <= 800:   return 'Medium'
        if cost <= 2000:  return 'High'
        return 'Premium'

    df_c['Cost Category'] = df_c['Average Cost for two'].apply(_cost_bucket)
    df_c['Price Bucket']  = df_c['Price range'].astype(float)
    df_c['Log Votes']     = np.log1p(df_c['Votes'])
    df_c['Log Cost']      = np.log1p(df_c['Average Cost for two'])

    print("[+] Engineered: Cuisine Count, Delivery/Booking Flags, "
          "Restaurant Age, Cost Category, Log Votes, Log Cost.")

    # 4. Drop target-leakage and identifier columns
    drop_cols = [
        'Restaurant ID', 'Restaurant Name', 'Address',
        'Locality Verbose', 'Currency', 'Switch to order menu',
        'Rating color', 'Rating text'
    ]
    df_c.drop(columns=drop_cols, inplace=True, errors='ignore')
    print(f"[+] Dropped columns: {drop_cols}")

    # 5. Label-encode categoricals
    cat_cols = ['City', 'Locality', 'Cuisines', 'Cost Category',
                'Has Online delivery', 'Has Table booking',
                'Is delivering now']
    label_encoders = {}
    for col in cat_cols:
        if col in df_c.columns:
            le = LabelEncoder()
            df_c[col] = le.fit_transform(df_c[col].astype(str))
            label_encoders[col] = le
    print("[+] Label-encoded categorical features.")

    return df_c, label_encoders


# ============================================================
# PART 3 — EDA VISUALIZATIONS
# ============================================================
def perform_eda(df):
    """Generate and save 6 professional EDA plots."""
    print("\n" + "="*50)
    print(" PART 3: EXPLORATORY DATA ANALYSIS ")
    print("="*50)

    # 1. Rating Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Aggregate rating'], kde=True, bins=30,
                 color='#1A365D', ax=ax)
    ax.set_title('Distribution of Restaurant Aggregate Ratings',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Aggregate Rating (0 - 5)'); ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig('images/rating_distribution.png', dpi=150)
    plt.close(fig)
    print("[+] Saved rating_distribution.png")

    # 2. Votes vs Rating
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df['Votes'], df['Aggregate rating'],
               alpha=0.35, color='#0D9488', s=15)
    ax.set_title('Votes vs Aggregate Rating',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Votes'); ax.set_ylabel('Aggregate Rating')
    fig.tight_layout()
    fig.savefig('images/votes_vs_rating.png', dpi=150)
    plt.close(fig)
    print("[+] Saved votes_vs_rating.png")

    # 3. Price Range vs Rating (boxplot)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='Price range', y='Aggregate rating', data=df,
                palette='viridis', ax=ax)
    ax.set_title('Aggregate Rating by Price Range',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Price Range (1=Low ... 4=Premium)')
    ax.set_ylabel('Aggregate Rating')
    fig.tight_layout()
    fig.savefig('images/price_range_vs_rating.png', dpi=150)
    plt.close(fig)
    print("[+] Saved price_range_vs_rating.png")

    # 4. Correlation Heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Restaurant Features',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig('images/correlation_heatmap.png', dpi=150)
    plt.close(fig)
    print("[+] Saved correlation_heatmap.png")

    # 5. Table Booking vs Rating
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Has Table booking', y='Aggregate rating', data=df,
                palette='Set2', ax=ax)
    ax.set_title('Table Booking vs Rating',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Has Table Booking (encoded)')
    ax.set_ylabel('Aggregate Rating')
    fig.tight_layout()
    fig.savefig('images/table_booking_vs_rating.png', dpi=150)
    plt.close(fig)
    print("[+] Saved table_booking_vs_rating.png")

    # 6. Pair Plot
    pplot_cols = ['Aggregate rating', 'Log Votes', 'Log Cost',
                  'Cuisine Count']
    g = sns.pairplot(df[pplot_cols], diag_kind='kde',
                     plot_kws={'alpha': 0.5, 'color': '#7C3AED', 's': 10})
    g.fig.suptitle('Pairwise Feature Relationships',
                   y=1.02, fontsize=14, fontweight='bold')
    g.fig.savefig('images/pair_plot.png', dpi=150)
    plt.close(g.fig)
    print("[+] Saved pair_plot.png")


# ============================================================
# PART 5 & 6 — ML MODELS & EVALUATION
# ============================================================
def _metrics(y_true, y_pred, n_features):
    """Compute MAE, MSE, RMSE, R², Adjusted R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    n    = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return mae, mse, rmse, r2, adj_r2


def train_ml_models(X_train, y_train, X_test, y_test):
    """Train 6 traditional regression models and evaluate."""
    print("\n" + "="*50)
    print(" PART 5 & 6: ML MODELS & EVALUATION ")
    print("="*50)

    models = {
        'Linear Regression':
            LinearRegression(),
        'Decision Tree':
            DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest':
            RandomForestRegressor(n_estimators=100, max_depth=12,
                                  random_state=42, n_jobs=1),
        'Gradient Boosting':
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=5, random_state=42),
        'Extra Trees':
            ExtraTreesRegressor(n_estimators=100, max_depth=12,
                                random_state=42, n_jobs=1),
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=1, verbosity=0)
        print("[+] XGBoost included.")

    results, trained = {}, {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"[*] Training {name} ...")
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae, mse, rmse, r2, adj_r2 = _metrics(
                y_test, preds, X_train.shape[1])
            cv = cross_val_score(model, X_train, y_train,
                                 cv=kf, scoring='r2').mean()
            results[name] = dict(MAE=mae, MSE=mse, RMSE=rmse,
                                 R2=r2, Adj_R2=adj_r2, CV_R2=cv)
            trained[name] = model
            print(f"    R2={r2:.4f}  RMSE={rmse:.4f}  CV={cv:.4f}")
        except Exception as exc:
            print(f"    [Error] {exc}")

    return trained, results


def train_tensorflow_models(X_train, y_train, X_test, y_test):
    """Train TF DNN + Keras sequential regression models."""
    print("\n[*] Training TensorFlow / Keras models ...")
    if not TF_AVAILABLE:
        print("[-] TensorFlow unavailable.")
        return None, {}, {}

    n = len(y_test); p = X_train.shape[1]

    # Scale target for neural net stability
    ts = StandardScaler()
    yt_s = ts.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # --- Model 7: TF Deep Neural Network ---
    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(p,)),
        BatchNormalization(), Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(), Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    dnn.compile(optimizer='adam', loss='mse')
    dnn.fit(X_train, yt_s, epochs=100, batch_size=64,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=10,
                                     restore_best_weights=True)],
            verbose=0)
    p7 = ts.inverse_transform(
        dnn.predict(X_test, verbose=0)).ravel()
    mae, mse, rmse, r2, adj_r2 = _metrics(y_test, p7, p)
    tf_metrics = dict(MAE=mae, MSE=mse, RMSE=rmse,
                      R2=r2, Adj_R2=adj_r2, CV_R2=np.nan)
    print(f"    [TF DNN]        R2={r2:.4f}  RMSE={rmse:.4f}")

    # --- Model 8: Keras Sequential (alt architecture) ---
    k2 = Sequential([
        Dense(64, activation='relu', input_shape=(p,)),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    k2.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3), loss='mse')
    k2.fit(X_train, yt_s, epochs=50, batch_size=64,
           validation_split=0.1, verbose=0)
    p8 = ts.inverse_transform(k2.predict(X_test, verbose=0)).ravel()
    mae2, mse2, rmse2, r2_2, adj2 = _metrics(y_test, p8, p)
    keras_metrics = dict(MAE=mae2, MSE=mse2, RMSE=rmse2,
                         R2=r2_2, Adj_R2=adj2, CV_R2=np.nan)
    print(f"    [Keras Alt]     R2={r2_2:.4f}  RMSE={rmse2:.4f}")

    # Return wrapped DNN for feature-importance / plots
    class _TFWrap:
        def __init__(self, model, scaler):
            self.model, self.scaler = model, scaler
        def predict(self, X):
            return self.scaler.inverse_transform(
                self.model.predict(X, verbose=0)).ravel()

    return _TFWrap(dnn, ts), tf_metrics, keras_metrics


def train_pytorch_model(X_train, y_train, X_test, y_test):
    """Train a PyTorch MLP regressor."""
    print("\n[*] Training PyTorch MLP ...")
    if not TORCH_AVAILABLE:
        print("[-] PyTorch unavailable.")
        return None, {}

    n = len(y_test); p = X_train.shape[1]
    ys = StandardScaler()
    yt_s = ys.fit_transform(y_train.values.reshape(-1, 1))

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(yt_s,    dtype=torch.float32)
    Xte = torch.tensor(X_test,  dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=64, shuffle=True)

    class MLP(nn.Module):
        def __init__(self, inp):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inp, 64), nn.ReLU(),
                nn.Linear(64, 32),  nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    model  = MLP(p)
    optim_ = optim.Adam(model.parameters(), lr=5e-3)
    crit   = nn.MSELoss()

    model.train()
    for _ in range(100):
        for bx, by in loader:
            optim_.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            optim_.step()

    model.eval()
    with torch.no_grad():
        p9 = ys.inverse_transform(
            model(Xte).numpy()).ravel()

    mae, mse, rmse, r2, adj_r2 = _metrics(y_test, p9, p)
    pt_metrics = dict(MAE=mae, MSE=mse, RMSE=rmse,
                      R2=r2, Adj_R2=adj_r2, CV_R2=np.nan)
    print(f"    [PyTorch MLP]   R2={r2:.4f}  RMSE={rmse:.4f}")
    return model, pt_metrics


# ============================================================
# PART 7 — FEATURE IMPORTANCE
# ============================================================
def analyse_feature_importance(model, X_train, y_train, feature_names):
    """Tree & permutation importance + plots."""
    print("\n" + "="*50)
    print(" PART 7: FEATURE IMPORTANCE ")
    print("="*50)

    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        print("\nTree Feature Importances (top 10):")
        for i in idx[:10]:
            print(f"  {feature_names[i]:<25}: {imp[i]:.4f}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=imp[idx], y=[feature_names[i] for i in idx],
                    palette='viridis', ax=ax)
        ax.set_title('Tree-Based Feature Importances',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        fig.tight_layout()
        fig.savefig('images/feature_importance_tree.png', dpi=150)
        plt.close(fig)
        print("[+] Saved feature_importance_tree.png")

    print("\nPermutation Importance (on training set) ...")
    perm = permutation_importance(model, X_train, y_train,
                                  n_repeats=5, random_state=42, n_jobs=1)
    pidx = perm.importances_mean.argsort()[::-1]
    print("  Top 10:")
    for i in pidx[:10]:
        print(f"  {feature_names[i]:<25}: "
              f"{perm.importances_mean[i]:.4f} ± {perm.importances_std[i]:.4f}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=perm.importances_mean[pidx],
                y=[feature_names[i] for i in pidx],
                palette='magma', ax=ax)
    ax.set_title('Permutation Feature Importances',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Mean Importance Decrease')
    ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig('images/feature_importance_permutation.png', dpi=150)
    plt.close(fig)
    print("[+] Saved feature_importance_permutation.png")


# ============================================================
# PART 8 — HYPERPARAMETER TUNING
# ============================================================
def tune_model(best_name, X_train, y_train):
    """GridSearchCV on the best traditional ML model."""
    print("\n" + "="*50)
    print(" PART 8: HYPERPARAMETER TUNING ")
    print("="*50)

    if best_name == 'XGBoost' and XGB_AVAILABLE:
        base = xgb.XGBRegressor(random_state=42, n_jobs=1, verbosity=0)
        grid = {'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1]}
    elif best_name in ('Random Forest', 'Extra Trees'):
        Cls  = (RandomForestRegressor if best_name == 'Random Forest'
                else ExtraTreesRegressor)
        base = Cls(random_state=42, n_jobs=1)
        grid = {'n_estimators': [100, 200],
                'max_depth': [10, 15]}
    elif best_name == 'Gradient Boosting':
        base = GradientBoostingRegressor(random_state=42)
        grid = {'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1]}
    else:
        print(f"[*] No tuning grid for {best_name}; returning as-is.")
        base = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=42)
        base.fit(X_train, y_train)
        return base

    gs = GridSearchCV(base, grid, cv=3, scoring='r2',
                      n_jobs=1, verbose=1)
    print(f"[*] GridSearchCV on {best_name} ...")
    gs.fit(X_train, y_train)
    print(f"[+] Best params : {gs.best_params_}")
    print(f"[+] Best CV R2  : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ============================================================
# PART 6 — PREDICTION PLOTS
# ============================================================
def plot_predictions(model, X_test, y_test):
    """Actual vs Predicted, Residual, and Residual Distribution plots."""
    preds = model.predict(X_test)
    resid = y_test - preds

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, preds, alpha=0.4, color='#1A365D', s=15)
    lims = [min(y_test.min(), preds.min()),
            max(y_test.max(), preds.max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_title('Actual vs Predicted Ratings',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    fig.tight_layout()
    fig.savefig('images/actual_vs_predicted.png', dpi=150)
    plt.close(fig)
    print("[+] Saved actual_vs_predicted.png")

    # 2. Residual Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(preds, resid, alpha=0.4, color='#7C3AED', s=15)
    ax.axhline(0, color='red', lw=2, linestyle='--')
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Residual')
    fig.tight_layout()
    fig.savefig('images/residual_plot.png', dpi=150)
    plt.close(fig)
    print("[+] Saved residual_plot.png")

    # 3. Residual Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(resid, kde=True, color='#DC2626', bins=30, ax=ax)
    ax.set_title('Distribution of Residuals',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Residual Error')
    fig.tight_layout()
    fig.savefig('images/residual_distribution.png', dpi=150)
    plt.close(fig)
    print("[+] Saved residual_distribution.png")


# ============================================================
# PART 9 — MODEL SAVING
# ============================================================
def save_model(model, scaler, encoders,
               pkl_path='models/best_model.pkl',
               jbl_path='models/best_model.joblib'):
    """Serialise model + scaler + encoders with pickle and joblib."""
    print("\n" + "="*50)
    print(" PART 9: MODEL SAVING ")
    print("="*50)

    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    bundle = {'model': model, 'scaler': scaler, 'encoders': encoders}

    with open(pkl_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[+] Saved -> {pkl_path}  (pickle)")

    joblib.dump(bundle, jbl_path)
    print(f"[+] Saved -> {jbl_path}  (joblib)")

    print("\n--- How to Load & Predict ---")
    print("  import pickle")
    print("  with open('models/best_model.pkl', 'rb') as f:")
    print("      bundle = pickle.load(f)")
    print("  prediction = bundle['model'].predict(scaled_features)[0]")


# ============================================================
# PART 10 — INFERENCE ON NEW SAMPLE
# ============================================================
def predict_sample(pkl_path='models/best_model.pkl'):
    """Load saved model and predict rating for a new restaurant."""
    print("\n" + "="*50)
    print(" PART 10: PREDICTION ON NEW SAMPLE ")
    print("="*50)

    if not os.path.exists(pkl_path):
        print(f"[-] {pkl_path} not found. Skipping.")
        return

    with open(pkl_path, 'rb') as f:
        bundle = pickle.load(f)
    model, scaler, encoders = (bundle['model'],
                                bundle['scaler'],
                                bundle['encoders'])

    raw = {
        'City': 'New Delhi',
        'Locality': 'Connaught Place',
        'Cuisines': 'North Indian, Chinese',
        'Average Cost for two': 1200,
        'Has Table booking': 'Yes',
        'Has Online delivery': 'Yes',
        'Is delivering now': 'No',
        'Price range': 3,
        'Votes': 450,
        'Country Code': 1,
        'Longitude': 77.2189,
        'Latitude': 28.6304,
    }

    rec = {
        'Cuisine Count': len(raw['Cuisines'].split(',')),
        'Online Delivery Flag': 1,
        'Table Booking Flag': 1,
        'Restaurant Age': 5,
        'Cost Category': 'High',
        'Price Bucket': float(raw['Price range']),
        'Log Votes': np.log1p(raw['Votes']),
        'Log Cost': np.log1p(raw['Average Cost for two']),
        'City': raw['City'],
        'Locality': raw['Locality'],
        'Cuisines': raw['Cuisines'],
        'Has Online delivery': raw['Has Online delivery'],
        'Has Table booking': raw['Has Table booking'],
        'Is delivering now': raw['Is delivering now'],
        'Price range': raw['Price range'],
        'Average Cost for two': raw['Average Cost for two'],
        'Votes': raw['Votes'],
        'Country Code': raw['Country Code'],
        'Longitude': raw['Longitude'],
        'Latitude': raw['Latitude'],
    }

    sdf = pd.DataFrame([rec])
    for col, enc in encoders.items():
        if col in sdf.columns:
            val = str(sdf.at[0, col])
            if val in enc.classes_:
                sdf[col] = enc.transform([val])[0]
            else:
                sdf[col] = enc.transform([enc.classes_[0]])[0]

    feat_order = scaler.feature_names_in_
    sample_scaled = scaler.transform(sdf[feat_order])
    rating = float(model.predict(sample_scaled)[0])
    rating = max(0.0, min(5.0, rating))

    print("\nNew Restaurant Details:")
    for k, v in raw.items():
        print(f"  {k:<24}: {v}")
    print("-" * 40)
    print(f"  -> Predicted Aggregate Rating : {rating:.2f} / 5.0")
    print("=" * 50)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  RESTAURANT RATING PREDICTION - FULL ML PIPELINE")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Explore
    explore_dataset(df)

    # 3. Preprocess + Feature Engineering
    df_proc, label_encoders = preprocess_and_engineer(df)

    # 4. EDA
    perform_eda(df_proc)

    # 5. Prepare features & target
    TARGET = 'Aggregate rating'
    X = df_proc.drop(columns=[TARGET])
    y = df_proc[TARGET]
    feature_names = list(X.columns)

    # 6. Split
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42)
    print(f"\n[+] Split -> Train: {len(X_tr_raw)}, Test: {len(X_te_raw)}")

    # 7. Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    scaler.feature_names_in_ = np.array(feature_names)

    # 8. Train traditional ML
    trained, results = train_ml_models(X_tr, y_tr, X_te, y_te)

    # 9. TensorFlow / Keras
    tf_wrap, tf_m, keras_m = train_tensorflow_models(
        X_tr, y_tr, X_te, y_te)
    if tf_wrap:
        results['TensorFlow DNN']   = tf_m
        results['Keras Sequential'] = keras_m

    # 10. PyTorch
    _, pt_m = train_pytorch_model(X_tr, y_tr, X_te, y_te)
    if pt_m:
        results['PyTorch MLP'] = pt_m

    # 11. Results table
    res_df = pd.DataFrame(results).T
    res_df.index.name = 'Model'
    print("\n" + "="*50)
    print(" MODEL PERFORMANCE SUMMARY ")
    print("="*50)
    print(res_df.to_string())

    # Save CSV
    os.makedirs('models', exist_ok=True)
    res_df.to_csv('models/model_comparison_results.csv')
    print("\n[+] Saved models/model_comparison_results.csv")

    # 12. Best traditional ML model
    traditional = {k: v for k, v in results.items()
                   if not np.isnan(v['CV_R2'])}
    best_name = max(traditional, key=lambda k: traditional[k]['R2'])
    print(f"\n[+] Best model: {best_name}  "
          f"(R2={traditional[best_name]['R2']:.4f})")

    # 13. Hyperparameter tuning
    best_tuned = tune_model(best_name, X_tr, y_tr)

    # Re-evaluate tuned model
    tp = best_tuned.predict(X_te)
    tr2 = r2_score(y_te, tp)
    trmse = np.sqrt(mean_squared_error(y_te, tp))
    print(f"\n[+] Tuned -> R2={tr2:.4f}  RMSE={trmse:.4f}")

    # 14. Feature importance
    analyse_feature_importance(best_tuned, X_tr, y_tr, feature_names)

    # 15. Prediction plots
    plot_predictions(best_tuned, X_te, y_te)

    # 16. Save model
    save_model(best_tuned, scaler, label_encoders)

    # 17. Inference
    predict_sample()

    print("\n[+] Pipeline complete. All artefacts saved.")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"\n[-] FATAL: {exc}")
        traceback.print_exc()
        sys.exit(1)
