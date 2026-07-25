"""
app.py — Streamlit Front-End for Restaurant Rating Prediction
=============================================================
A stunning, production-ready Streamlit application that provides:
  • Live rating predictions using the trained ML model
  • Interactive EDA visualizations
  • Model comparison leaderboard
  • Feature importance analysis
  • About / Project overview page

Run with:
    streamlit run app.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark premium theme with glassmorphism & animations
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base & Background ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.10);
}

section[data-testid="stSidebar"] .stRadio > label {
    color: #e2e8f0 !important;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #cbd5e0 !important;
    padding: 0.5rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: background 0.2s;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(139,92,246,0.25);
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(102,126,234,0.4);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
    animation: float 6s ease-in-out infinite;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 250px;
    height: 250px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
    animation: float 8s ease-in-out infinite reverse;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-20px); }
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0 0 0.5rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-weight: 400;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(102,126,234,0.3);
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.25) 0%, rgba(118,75,162,0.25) 100%);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(139,92,246,0.7);
    box-shadow: 0 12px 30px rgba(102,126,234,0.35);
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}

.metric-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.65);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
    font-weight: 500;
}

.metric-sublabel {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    margin-top: 0.2rem;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.5);
    margin-bottom: 1.5rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.5), transparent);
    margin: 1.5rem 0;
}

/* ── Prediction Result ── */
.prediction-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid transparent;
    background-clip: padding-box;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
}

.prediction-box::before {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 22px;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
    z-index: -1;
}

.prediction-rating {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

.prediction-label {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.5rem;
}

/* ── Star Rating ── */
.star-display {
    font-size: 1.8rem;
    letter-spacing: 0.1rem;
    margin: 0.5rem 0;
}

/* ── Rating Badge ── */
.rating-badge {
    display: inline-block;
    padding: 0.35rem 1.2rem;
    border-radius: 50px;
    font-size: 0.9rem;
    font-weight: 700;
    margin-top: 0.6rem;
}

/* ── Info Alert ── */
.info-alert {
    background: rgba(102,126,234,0.15);
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #e2e8f0;
    font-size: 0.9rem;
}

/* ── Table Styling ── */
.styled-table {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Button Overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    letter-spacing: 0.02em;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.55);
}

/* ── Form / Input Overrides ── */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Sidebar Nav Title ── */
.nav-title {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.nav-subtitle {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45);
    margin-bottom: 1.5rem;
    letter-spacing: 0.05em;
}

/* ── Feature Table ── */
.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    color: #cbd5e0;
    font-size: 0.88rem;
}

.feature-name { color: rgba(255,255,255,0.55); }
.feature-val  { font-weight: 600; color: #a78bfa; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PKL   = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
IMG_DIR     = os.path.join(os.path.dirname(__file__), "images")
CSV_RESULTS = os.path.join(os.path.dirname(__file__), "models", "model_comparison_results.csv")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "dataset", "restaurant_data.csv")

STAR_MAP = {
    (0.0, 1.5): ("☆☆☆☆☆", "#ef4444", "Not Rated"),
    (1.5, 2.5): ("★☆☆☆☆", "#f97316", "Poor"),
    (2.5, 3.0): ("★★☆☆☆", "#eab308", "Average"),
    (3.0, 3.5): ("★★★☆☆", "#84cc16", "Good"),
    (3.5, 4.0): ("★★★★☆", "#22c55e", "Very Good"),
    (4.0, 4.5): ("★★★★☆", "#10b981", "Excellent"),
    (4.5, 5.1): ("★★★★★", "#6ee7b7", "Outstanding"),
}

def rating_to_stars(r):
    for (lo, hi), (stars, color, label) in STAR_MAP.items():
        if lo <= r < hi:
            return stars, color, label
    return "★★★★★", "#6ee7b7", "Outstanding"

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bundle(path: str):
    """Load model bundle once and cache it across sessions."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_results_csv(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df.index.name = "Model"
    return df

@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = (df.columns
                  .str.replace('\ufeff', '', regex=False)
                  .str.replace('ï»¿', '', regex=False)
                  .str.strip())
    return df

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def img_path(name: str) -> str:
    return os.path.join(IMG_DIR, name)

def show_image(name: str, caption: str = "", use_container_width: bool = True):
    p = img_path(name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=use_container_width)
    else:
        st.warning(f"Image not found: {name}")

def section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="section-header">{icon} {title}</div>
    <div class="section-sub">{subtitle}</div>
    """, unsafe_allow_html=True)

def glass_metric(label: str, value: str, sublabel: str = ""):
    sub = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {sub}
    </div>
    """

def preprocess_input(raw: dict, encoders: dict, scaler) -> np.ndarray:
    """Mirror the preprocessing from restaurant_rating_prediction.py."""
    cuisines_str = raw["Cuisines"]
    cuisine_count = len(cuisines_str.split(","))

    cost = raw["Average Cost for two"]
    if cost <= 300:
        cost_cat = "Low"
    elif cost <= 800:
        cost_cat = "Medium"
    elif cost <= 2000:
        cost_cat = "High"
    else:
        cost_cat = "Premium"

    rec = {
        "Cuisine Count":           cuisine_count,
        "Online Delivery Flag":    1 if raw["Has Online delivery"] == "Yes" else 0,
        "Table Booking Flag":      1 if raw["Has Table booking"] == "Yes" else 0,
        "Restaurant Age":          raw["Restaurant Age"],
        "Cost Category":           cost_cat,
        "Price Bucket":            float(raw["Price range"]),
        "Log Votes":               np.log1p(raw["Votes"]),
        "Log Cost":                np.log1p(cost),
        "City":                    raw["City"],
        "Locality":                raw["Locality"],
        "Cuisines":                cuisines_str,
        "Has Online delivery":     raw["Has Online delivery"],
        "Has Table booking":       raw["Has Table booking"],
        "Is delivering now":       raw["Is delivering now"],
        "Price range":             raw["Price range"],
        "Average Cost for two":    cost,
        "Votes":                   raw["Votes"],
        "Country Code":            raw["Country Code"],
        "Longitude":               raw["Longitude"],
        "Latitude":                raw["Latitude"],
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
    return scaler.transform(sdf[feat_order])


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div class="nav-title">🍽️ RatePro AI</div>
        <div class="nav-subtitle">RESTAURANT INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate to",
        options=[
            "🏠  Overview",
            "📊  EDA & Insights",
            "🤖  Model Leaderboard",
            "🔮  Live Predictor",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Sidebar stats
    bundle = load_bundle(MODEL_PKL)
    res_df = load_results_csv(CSV_RESULTS)

    if res_df is not None:
        best_r2   = res_df["R2"].max()
        best_rmse = res_df["RMSE"].min()
        best_name = res_df["R2"].idxmax()
        st.markdown(f"""
        <div style='padding:1rem 0;'>
            <div style='color:rgba(255,255,255,0.4);font-size:0.7rem;letter-spacing:0.1em;margin-bottom:0.8rem;'>MODEL STATUS</div>
            <div style='color:#22c55e;font-size:0.82rem;font-weight:700;'>✓ Model Loaded</div>
            <div style='color:rgba(255,255,255,0.5);font-size:0.75rem;margin-top:0.3rem;'>{best_name}</div>
            <div style='margin-top:0.8rem;'>
                <span style='color:rgba(255,255,255,0.4);font-size:0.72rem;'>Best R²</span>
                <span style='color:#a78bfa;font-weight:700;font-size:0.88rem;margin-left:0.5rem;'>{best_r2:.4f}</span>
            </div>
            <div style='margin-top:0.3rem;'>
                <span style='color:rgba(255,255,255,0.4);font-size:0.72rem;'>Best RMSE</span>
                <span style='color:#67e8f9;font-weight:700;font-size:0.88rem;margin-left:0.5rem;'>{best_rmse:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='position:absolute;bottom:1.5rem;left:1.5rem;right:1.5rem;'>
        <div style='color:rgba(255,255,255,0.25);font-size:0.7rem;text-align:center;'>
            Built with Streamlit · ML Pipeline v1.0<br/>Zomato Restaurant Dataset
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🍽️ Restaurant Rating Predictor</div>
        <div class="hero-subtitle">
            A production-ready AI system that predicts restaurant aggregate ratings<br>
            using ensemble machine learning — powered by the Zomato dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    if res_df is not None:
        n_models  = len(res_df)
        best_r2   = res_df["R2"].max()
        best_rmse = res_df["RMSE"].min()
        best_mae  = res_df["MAE"].min()
        best_cv   = res_df["CV_R2"].dropna().max()

        cols = st.columns(5)
        metrics = [
            ("Models Trained", str(n_models),     "Algorithms"),
            ("Best R² Score",  f"{best_r2:.4f}",  "Coefficient of Determination"),
            ("Best RMSE",      f"{best_rmse:.4f}", "Root Mean Squared Error"),
            ("Best MAE",       f"{best_mae:.4f}",  "Mean Absolute Error"),
            ("Best CV R²",     f"{best_cv:.4f}",   "5-Fold Cross Validation"),
        ]
        for col, (label, val, sub) in zip(cols, metrics):
            with col:
                st.markdown(glass_metric(label, val, sub), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # About cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:2rem;margin-bottom:0.6rem;">🎯</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">Project Goal</div>
            <div style="font-size:0.88rem;color:rgba(255,255,255,0.6);line-height:1.6;">
                Predict the <strong style="color:#a78bfa">Aggregate Rating</strong> (0–5) of restaurants
                on the Zomato platform using 9 different ML algorithms including
                tree ensembles, gradient boosting, and deep learning models.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:2rem;margin-bottom:0.6rem;">🛠️</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">Tech Stack</div>
            <div style="font-size:0.88rem;color:rgba(255,255,255,0.6);line-height:1.8;">
                🐍 Python 3.10+ &nbsp;|&nbsp; scikit-learn<br>
                ⚡ XGBoost &nbsp;|&nbsp; TensorFlow/Keras<br>
                🔦 PyTorch &nbsp;|&nbsp; Pandas / NumPy<br>
                📊 Matplotlib / Seaborn<br>
                🚀 Streamlit (this app)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:2rem;margin-bottom:0.6rem;">🔬</div>
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">Pipeline Steps</div>
            <div style="font-size:0.88rem;color:rgba(255,255,255,0.6);line-height:1.8;">
                1. Data Ingestion & Cleaning<br>
                2. Feature Engineering (8 new features)<br>
                3. EDA & Visualization<br>
                4. Multi-Model Training<br>
                5. GridSearch Tuning<br>
                6. Feature Importance Analysis<br>
                7. Model Serialization & Inference
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Model leaderboard preview
    section_header("🏆", "Model Leaderboard Preview",
                   "Top-line performance comparison across all trained models")

    if res_df is not None:
        display_df = res_df.copy()
        display_df = display_df.sort_values("R2", ascending=False)
        display_df = display_df.rename(columns={
            "MAE": "MAE ↓", "MSE": "MSE ↓", "RMSE": "RMSE ↓",
            "R2": "R² ↑", "Adj_R2": "Adj R² ↑", "CV_R2": "CV R² ↑"
        })
        st.dataframe(
            display_df.style
                .format("{:.4f}", na_rep="N/A")
                .background_gradient(subset=["R² ↑", "Adj R² ↑"], cmap="Greens")
                .background_gradient(subset=["RMSE ↓", "MAE ↓"], cmap="Reds_r"),
            use_container_width=True,
            height=370,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Engineered features
    section_header("⚙️", "Engineered Features",
                   "New features derived from the raw dataset to boost model performance")

    feat_data = [
        ("Cuisine Count",        "Number of different cuisines offered (from Cuisines column)"),
        ("Online Delivery Flag", "Binary: 1 = Has Online Delivery, 0 = No"),
        ("Table Booking Flag",   "Binary: 1 = Has Table Booking, 0 = No"),
        ("Restaurant Age",       "Simulated age from Restaurant ID (2026 − ID%15 − 2010)"),
        ("Cost Category",        "Bucketed cost: Low / Medium / High / Premium"),
        ("Price Bucket",         "Numeric price range (1–4)"),
        ("Log Votes",            "Natural log of Votes (log1p)"),
        ("Log Cost",             "Natural log of Average Cost for two (log1p)"),
    ]

    html = "<div class='glass-card'>"
    for name, desc in feat_data:
        html += f"""
        <div class="feature-row">
            <span class="feature-name">🔹 {name}</span>
            <span style="color:rgba(255,255,255,0.55);font-size:0.85rem;">{desc}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📊 EDA & Insights</div>
        <div class="hero-subtitle">Exploratory Data Analysis — understanding patterns in the Zomato dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats from dataset
    raw_df = load_dataset(DATA_PATH)
    if raw_df is not None:
        cols = st.columns(4)
        stats = [
            ("Total Records",   f"{len(raw_df):,}",        "Restaurants"),
            ("Unique Cities",   f"{raw_df['City'].nunique():,}" if 'City' in raw_df.columns else "N/A", "Locations"),
            ("Avg Rating",      f"{raw_df['Aggregate rating'].mean():.2f}" if 'Aggregate rating' in raw_df.columns else "N/A", "Out of 5.0"),
            ("Cuisines Types",  f"{raw_df['Cuisines'].nunique():,}" if 'Cuisines' in raw_df.columns else "N/A", "Unique"),
        ]
        for col, (label, val, sub) in zip(cols, stats):
            with col:
                st.markdown(glass_metric(label, val, sub), unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Row 1
    section_header("📈", "Rating Distribution",
                   "How restaurant ratings are distributed across the dataset")
    show_image("rating_distribution.png")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Row 2 — 2 columns
    section_header("🔍", "Feature Relationships",
                   "Exploring how individual features correlate with the rating")
    c1, c2 = st.columns(2)
    with c1:
        show_image("votes_vs_rating.png", "Votes vs Aggregate Rating")
    with c2:
        show_image("price_range_vs_rating.png", "Price Range vs Aggregate Rating")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        show_image("table_booking_vs_rating.png", "Table Booking vs Rating")
    with c2:
        # Live dynamic chart — top cities by average rating
        if raw_df is not None and "City" in raw_df.columns and "Aggregate rating" in raw_df.columns:
            top_cities = (
                raw_df[raw_df["Aggregate rating"] > 0]
                .groupby("City")["Aggregate rating"]
                .agg(["mean", "count"])
                .query("count >= 20")
                .sort_values("mean", ascending=False)
                .head(15)
                .reset_index()
            )
            if not top_cities.empty:
                fig, ax = plt.subplots(figsize=(8, 5),
                                       facecolor="none")
                ax.set_facecolor("none")
                bars = ax.barh(
                    top_cities["City"][::-1],
                    top_cities["mean"][::-1],
                    color=plt.cm.plasma(
                        np.linspace(0.3, 0.9, len(top_cities))),
                )
                ax.set_xlabel("Average Rating", color="white")
                ax.set_title("Top 15 Cities by Avg Rating (≥20 restaurants)",
                             color="white", fontweight="bold")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("rgba(255,255,255,0.1)")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    section_header("🌡️", "Correlation Heatmap",
                   "Pearson correlation between all numeric features and the target")
    show_image("correlation_heatmap.png")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    section_header("🔗", "Pairwise Relationships",
                   "Pair-plot revealing non-linear feature interactions")
    show_image("pair_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Leaderboard" in page:

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🤖 Model Leaderboard</div>
        <div class="hero-subtitle">Comprehensive comparison of 9 trained ML models — from linear regression to deep learning.</div>
    </div>
    """, unsafe_allow_html=True)

    res_df = load_results_csv(CSV_RESULTS)
    if res_df is None:
        st.error("Model results CSV not found. Please run the main pipeline first.")
    else:
        # KPIs
        best_name = res_df["R2"].idxmax()
        best_r2   = res_df["R2"].max()
        best_rmse = res_df["RMSE"].min()
        best_mae  = res_df["MAE"].min()

        cols = st.columns(3)
        with cols[0]:
            st.markdown(glass_metric("Best Model",  best_name, "by R² Score"), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(glass_metric("R² Score",    f"{best_r2:.4f}", "Coefficient of Determination"), unsafe_allow_html=True)
        with cols[2]:
            st.markdown(glass_metric("Best RMSE",   f"{best_rmse:.4f}", "Root Mean Squared Error"), unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Full sortable table
        section_header("📋", "Full Results Table", "Click column headers to sort")
        sorted_df = res_df.sort_values("R2", ascending=False)
        st.dataframe(
            sorted_df.style
                .format("{:.4f}", na_rep="N/A")
                .highlight_max(subset=["R2", "Adj_R2", "CV_R2"], color="#1a472a")
                .highlight_min(subset=["MAE", "RMSE"],            color="#1a472a"),
            use_container_width=True,
            height=370,
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Dynamic bar chart — R2 comparison
        section_header("📊", "R² Score Comparison", "Higher is better")

        fig, ax = plt.subplots(figsize=(11, 5), facecolor="none")
        ax.set_facecolor("none")
        sorted_models = sorted_df.index.tolist()
        r2_vals = sorted_df["R2"].values
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sorted_models)))
        bars = ax.barh(sorted_models[::-1], r2_vals[::-1],
                       color=colors, height=0.6)
        for bar, val in zip(bars, r2_vals[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", ha="left",
                    color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("R² Score", color="white")
        ax.set_title("Model R² Score Comparison", color="white",
                     fontweight="bold", fontsize=13)
        ax.tick_params(colors="white")
        ax.set_xlim(0, max(r2_vals) * 1.08)
        for spine in ax.spines.values():
            spine.set_edgecolor("rgba(255,255,255,0.1)")
        ax.axvline(best_r2, color="#f6d365", linewidth=1.5,
                   linestyle="--", alpha=0.7, label=f"Best: {best_r2:.4f}")
        ax.legend(facecolor="none", labelcolor="white", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # RMSE comparison
        c1, c2 = st.columns(2)
        with c1:
            section_header("📉", "RMSE Comparison", "Lower is better")
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="none")
            ax.set_facecolor("none")
            rmse_s = res_df["RMSE"].sort_values()
            clrs = plt.cm.viridis(np.linspace(0.3, 0.9, len(rmse_s)))
            ax.barh(rmse_s.index, rmse_s.values, color=clrs, height=0.6)
            ax.set_xlabel("RMSE", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("rgba(255,255,255,0.1)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            section_header("📉", "MAE Comparison", "Lower is better")
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="none")
            ax.set_facecolor("none")
            mae_s = res_df["MAE"].sort_values()
            clrs = plt.cm.magma(np.linspace(0.3, 0.9, len(mae_s)))
            ax.barh(mae_s.index, mae_s.values, color=clrs, height=0.6)
            ax.set_xlabel("MAE", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("rgba(255,255,255,0.1)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # Feature importance
        section_header("🌟", "Feature Importance",
                        "What drives the model's predictions?")
        c1, c2 = st.columns(2)
        with c1:
            show_image("feature_importance_tree.png", "Tree-Based Importances")
        with c2:
            show_image("feature_importance_permutation.png", "Permutation Importances")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        section_header("🎯", "Prediction Quality",
                        "Visual diagnostic plots for the best model")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_image("actual_vs_predicted.png", "Actual vs Predicted")
        with c2:
            show_image("residual_plot.png", "Residual Plot")
        with c3:
            show_image("residual_distribution.png", "Residual Distribution")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Predictor" in page:

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔮 Live Rating Predictor</div>
        <div class="hero-subtitle">Enter restaurant details below and get an instant AI-powered rating prediction.</div>
    </div>
    """, unsafe_allow_html=True)

    bundle = load_bundle(MODEL_PKL)
    if bundle is None:
        st.error("❌ Trained model not found at `models/best_model.pkl`."
                 " Please run the main pipeline (`restaurant_rating_prediction.py`) first.")
        st.stop()

    model    = bundle["model"]
    scaler   = bundle["scaler"]
    encoders = bundle["encoders"]

    # ── Form layout ──────────────────────────────────────────────────────────
    c_form, c_result = st.columns([1.1, 0.9], gap="large")

    with c_form:
        section_header("📝", "Restaurant Details",
                       "Fill in all fields for an accurate prediction")

        # City / Locality — use known encoder classes if available
        city_options    = sorted(encoders["City"].classes_.tolist())     if "City"     in encoders else ["New Delhi", "Mumbai", "Bangalore"]
        loc_options     = sorted(encoders["Locality"].classes_.tolist()) if "Locality" in encoders else ["Connaught Place", "Koramangala"]
        cuisine_opts    = sorted(encoders["Cuisines"].classes_.tolist()) if "Cuisines" in encoders else ["North Indian", "Chinese"]

        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("🏙️ City", city_options,
                                index=city_options.index("New Delhi") if "New Delhi" in city_options else 0)
        with col2:
            locality = st.selectbox("📍 Locality", loc_options)

        cuisines_sel = st.multiselect(
            "🍜 Cuisines (select all that apply)",
            options=cuisine_opts,
            default=[cuisine_opts[0]] if cuisine_opts else [],
            help="Select one or more cuisines",
        )
        cuisines_str = ", ".join(cuisines_sel) if cuisines_sel else cuisine_opts[0]

        col3, col4 = st.columns(2)
        with col3:
            avg_cost = st.number_input(
                "💰 Average Cost for Two (₹)",
                min_value=50, max_value=10000, value=800, step=50,
                help="Average cost for two people in local currency",
            )
        with col4:
            votes = st.number_input(
                "🗳️ Votes", min_value=0, max_value=100000, value=300, step=10,
                help="Number of user votes the restaurant has received",
            )

        col5, col6, col7 = st.columns(3)
        with col5:
            price_range = st.select_slider(
                "💲 Price Range",
                options=[1, 2, 3, 4],
                value=2,
                help="1=Cheap, 2=Moderate, 3=Expensive, 4=Very Expensive",
            )
        with col6:
            has_online = st.selectbox("🛵 Online Delivery",   ["Yes", "No"])
        with col7:
            has_table  = st.selectbox("🪑 Table Booking",     ["Yes", "No"])

        col8, col9, col10 = st.columns(3)
        with col8:
            is_delivering = st.selectbox("🟢 Delivering Now", ["No", "Yes"])
        with col9:
            rest_age = st.number_input(
                "📅 Restaurant Age (yrs)", min_value=0, max_value=50, value=5)
        with col10:
            country_code = st.number_input(
                "🌍 Country Code", min_value=1, max_value=999, value=1)

        col11, col12 = st.columns(2)
        with col11:
            latitude  = st.number_input("🗺️ Latitude",  value=28.6304, format="%.4f")
        with col12:
            longitude = st.number_input("🗺️ Longitude", value=77.2189, format="%.4f")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict Rating", use_container_width=True)

    # ── Prediction Result ────────────────────────────────────────────────────
    with c_result:
        section_header("🎯", "Prediction Result",
                       "AI-powered aggregate rating prediction")

        if predict_btn:
            with st.spinner("Running inference…"):
                raw_input = {
                    "City":                  city,
                    "Locality":              locality,
                    "Cuisines":              cuisines_str,
                    "Average Cost for two":  avg_cost,
                    "Has Table booking":     has_table,
                    "Has Online delivery":   has_online,
                    "Is delivering now":     is_delivering,
                    "Price range":           price_range,
                    "Votes":                 votes,
                    "Country Code":          country_code,
                    "Longitude":             longitude,
                    "Latitude":              latitude,
                    "Restaurant Age":        rest_age,
                }

                try:
                    X_scaled = preprocess_input(raw_input, encoders, scaler)
                    rating   = float(model.predict(X_scaled)[0])
                    rating   = max(0.0, min(5.0, rating))

                    stars, color, label = rating_to_stars(rating)

                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size:0.85rem;color:rgba(255,255,255,0.5);
                                    margin-bottom:0.5rem;letter-spacing:0.1em;">
                            PREDICTED AGGREGATE RATING
                        </div>
                        <div class="prediction-rating">{rating:.2f}</div>
                        <div class="star-display" style="color:{color};">{stars}</div>
                        <div class="prediction-label">{label}</div>
                        <div style="margin-top:1rem;">
                            <span style="background:rgba(255,255,255,0.08);
                                         border:1px solid rgba(255,255,255,0.15);
                                         border-radius:50px;padding:0.3rem 1rem;
                                         color:rgba(255,255,255,0.7);font-size:0.8rem;">
                                Scale: 0.0 – 5.0
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Rating gauge bar
                    pct = rating / 5.0
                    gauge_clr = color
                    st.markdown(f"""
                    <div style="margin-top:1.2rem;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:0.72rem;color:rgba(255,255,255,0.4);
                                    margin-bottom:0.3rem;">
                            <span>0.0</span><span>2.5</span><span>5.0</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.08);border-radius:50px;
                                    height:10px;overflow:hidden;">
                            <div style="width:{pct*100:.1f}%;height:100%;
                                        background:linear-gradient(90deg,#667eea,{gauge_clr});
                                        border-radius:50px;
                                        transition:width 1s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Input summary
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="font-size:0.82rem;font-weight:700;
                                color:rgba(255,255,255,0.55);margin-bottom:0.7rem;
                                letter-spacing:0.06em;">INPUT SUMMARY</div>
                    """, unsafe_allow_html=True)

                    summary_items = [
                        ("City",         city),
                        ("Locality",     locality),
                        ("Cuisines",     cuisines_str[:40] + ("…" if len(cuisines_str) > 40 else "")),
                        ("Avg Cost",     f"₹{avg_cost:,}"),
                        ("Price Range",  f"{price_range}/4"),
                        ("Votes",        f"{votes:,}"),
                        ("Online Del.",  has_online),
                        ("Table Book.",  has_table),
                    ]
                    html = "<div class='glass-card' style='padding:1rem;'>"
                    for k, v in summary_items:
                        html += f"""
                        <div class="feature-row">
                            <span class="feature-name">{k}</span>
                            <span class="feature-val">{v}</span>
                        </div>"""
                    html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

        else:
            # Placeholder state
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:3rem 2rem;">
                <div style="font-size:3.5rem;margin-bottom:1rem;">🔮</div>
                <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">
                    Ready to Predict
                </div>
                <div style="font-size:0.88rem;color:rgba(255,255,255,0.45);">
                    Fill in the restaurant details on the left<br>
                    and click <strong style="color:#a78bfa">⚡ Predict Rating</strong> to get<br>
                    an AI-powered prediction instantly.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick tips
            st.markdown("""
            <div class="info-alert">
                💡 <strong>Tips for better accuracy:</strong><br>
                • Choose cuisines that match the restaurant's menu<br>
                • Set price range accurately (1=Budget, 4=Fine Dining)<br>
                • Higher votes generally indicates better reliability
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">ℹ️ About This Project</div>
        <div class="hero-subtitle">
            A production-ready ML internship project — built for learning, understanding, and deployment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">
                📖 Project Overview
            </div>
            <div style="font-size:0.9rem;color:rgba(255,255,255,0.65);line-height:1.8;">
                This project demonstrates an end-to-end Machine Learning pipeline
                for predicting restaurant aggregate ratings from the Zomato dataset.
                It was built as an internship demonstration project showcasing
                industry-level best practices.<br><br>
                The pipeline covers every stage from raw data ingestion through
                feature engineering, model training, evaluation, hyperparameter tuning,
                serialization, and finally this Streamlit web deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">
                🗂️ Project Structure
            </div>
            <pre style="color:#a78bfa;font-size:0.82rem;background:rgba(0,0,0,0.3);
                        padding:1rem;border-radius:8px;overflow-x:auto;line-height:1.7;">
restaurant_rating_prediction/
├── app.py                        ← This Streamlit app
├── restaurant_rating_prediction.py  ← ML pipeline
├── generate_report.py            ← PDF report generator
├── requirements.txt
├── README.md
├── report.pdf                    ← Generated report
├── dataset/
│   └── restaurant_data.csv       ← Zomato dataset
├── models/
│   ├── best_model.pkl
│   ├── best_model.joblib
│   └── model_comparison_results.csv
└── images/
    ├── rating_distribution.png
    ├── correlation_heatmap.png
    └── ...                       ← 11 EDA/eval plots
            </pre>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">
                🤖 Models Trained
            </div>
            <div style="font-size:0.88rem;color:rgba(255,255,255,0.65);line-height:2;">
                1️⃣ &nbsp;<strong style="color:#a78bfa">Linear Regression</strong> — Baseline model<br>
                2️⃣ &nbsp;<strong style="color:#a78bfa">Decision Tree</strong> — Non-linear baseline<br>
                3️⃣ &nbsp;<strong style="color:#a78bfa">Random Forest</strong> — Bagging ensemble<br>
                4️⃣ &nbsp;<strong style="color:#a78bfa">Gradient Boosting</strong> — Sequential ensemble<br>
                5️⃣ &nbsp;<strong style="color:#a78bfa">Extra Trees</strong> — Randomized forest<br>
                6️⃣ &nbsp;<strong style="color:#a78bfa">XGBoost</strong> — Optimized boosting<br>
                7️⃣ &nbsp;<strong style="color:#a78bfa">TensorFlow DNN</strong> — Deep neural network<br>
                8️⃣ &nbsp;<strong style="color:#a78bfa">Keras Sequential</strong> — Alt DNN architecture<br>
                9️⃣ &nbsp;<strong style="color:#a78bfa">PyTorch MLP</strong> — Custom MLP regressor
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:1rem;">
                🚀 How to Run
            </div>
            <div style="font-size:0.88rem;color:rgba(255,255,255,0.65);line-height:1.8;margin-bottom:0.8rem;">
                Step 1: Install dependencies
            </div>
            <pre style="color:#67e8f9;font-size:0.82rem;background:rgba(0,0,0,0.3);
                        padding:0.8rem;border-radius:8px;">pip install -r requirements.txt</pre>
            <div style="font-size:0.88rem;color:rgba(255,255,255,0.65);line-height:1.8;margin:0.8rem 0;">
                Step 2: Run the ML pipeline (trains & saves model)
            </div>
            <pre style="color:#67e8f9;font-size:0.82rem;background:rgba(0,0,0,0.3);
                        padding:0.8rem;border-radius:8px;">python restaurant_rating_prediction.py</pre>
            <div style="font-size:0.88rem;color:rgba(255,255,255,0.65);line-height:1.8;margin:0.8rem 0;">
                Step 3: Launch this Streamlit app
            </div>
            <pre style="color:#67e8f9;font-size:0.82rem;background:rgba(0,0,0,0.3);
                        padding:0.8rem;border-radius:8px;">streamlit run app.py</pre>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Features checklist
    section_header("✅", "Features Implemented",
                   "What this project covers")

    feats = [
        ("Data Ingestion",           "Reads CSV with BOM-stripping and encoding handling"),
        ("Data Cleaning",            "Deduplication, missing value imputation"),
        ("Feature Engineering",      "8 derived features: log transforms, buckets, flags"),
        ("EDA Visualizations",       "11 professionally styled matplotlib/seaborn plots"),
        ("9 ML Models",              "Linear to DNN — sklearn, XGBoost, TF, PyTorch"),
        ("Cross-Validation",         "5-Fold KFold CV R² for all sklearn models"),
        ("GridSearch Tuning",        "Hyperparameter optimization for the best model"),
        ("Feature Importance",       "Tree-based + permutation importance analysis"),
        ("Model Serialization",      "Pickle and Joblib save/load bundles"),
        ("Inference Pipeline",       "Full preprocessing → scaling → prediction chain"),
        ("PDF Report",               "Multi-page professional report with ReportLab"),
        ("Jupyter Notebook",         "Matching .ipynb for interactive analysis"),
        ("Streamlit Deployment",     "This interactive web application"),
    ]

    cols = st.columns(2)
    half = len(feats) // 2
    for i, (feat, desc) in enumerate(feats):
        with cols[0 if i < half + 1 else 1]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:0.7rem;
                        padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.06);">
                <span style="color:#22c55e;font-size:1rem;min-width:20px;">✓</span>
                <div>
                    <div style="color:#e2e8f0;font-size:0.88rem;font-weight:600;">{feat}</div>
                    <div style="color:rgba(255,255,255,0.45);font-size:0.78rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:2rem;color:rgba(255,255,255,0.3);font-size:0.8rem;">
        🍽️ Restaurant Rating Predictor &nbsp;|&nbsp;
        Built with ❤️ using Python, scikit-learn & Streamlit &nbsp;|&nbsp;
        Zomato Dataset &nbsp;|&nbsp; Internship Project 2026
    </div>
    """, unsafe_allow_html=True)
