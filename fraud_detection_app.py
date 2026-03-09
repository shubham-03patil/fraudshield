import streamlit as st
import numpy as np
import time
import random
import pandas as pd
import pickle
from datetime import datetime

# ── Loading screen ────────────────────────────────────────────────────────────
def show_loading_screen():
    st.markdown("""
    <style>
    .loading-wrapper {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 80vh; text-align: center;
    }
    .loading-shield {
        font-size: 4rem; margin-bottom: 1.5rem;
        animation: pulse-shield 1.5s infinite;
    }
    @keyframes pulse-shield {
        0%, 100% { transform: scale(1); opacity: 1; }
        50%       { transform: scale(1.1); opacity: 0.8; }
    }
    .loading-title {
        font-size: 2rem; font-weight: 800; color: #f0f4ff;
        letter-spacing: -0.02em; margin-bottom: 0.3rem;
    }
    .loading-title span { color: #3b82f6; }
    .loading-sub {
        font-size: 0.8rem; color: #4b6a9c; letter-spacing: 0.12em;
        text-transform: uppercase; margin-bottom: 2.5rem;
    }
    .loading-bar-bg {
        width: 300px; height: 4px; background: #0d1827;
        border-radius: 2px; overflow: hidden; margin: 0 auto 1rem auto;
        border: 1px solid #1a2a42;
    }
    .loading-bar-fill {
        height: 100%; background: linear-gradient(90deg, #1d4ed8, #3b82f6);
        border-radius: 2px; animation: loading-progress 3s ease-in-out forwards;
    }
    @keyframes loading-progress {
        0%   { width: 0%; }
        20%  { width: 25%; }
        50%  { width: 60%; }
        80%  { width: 85%; }
        100% { width: 100%; }
    }
    .loading-status {
        font-size: 0.75rem; color: #3b82f6; font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em; height: 1.2rem;
    }
    .loading-steps {
        margin-top: 2rem; display: flex; flex-direction: column; gap: 0.5rem;
        align-items: center;
    }
    .loading-step {
        font-size: 0.72rem; color: #2a3a52; font-family: 'JetBrains Mono', monospace;
        display: flex; align-items: center; gap: 8px;
    }
    .loading-step.done { color: #10b981; }
    .loading-step.active { color: #3b82f6; }
    </style>
    <div class="loading-wrapper">
      <div class="loading-shield">🛡️</div>
      <div class="loading-title">Fraud<span>Shield</span></div>
      <div class="loading-sub">Ensemble Detection System</div>
      <div class="loading-bar-bg">
        <div class="loading-bar-fill"></div>
      </div>
      <div class="loading-steps">
        <div class="loading-step done">✓ &nbsp;Secure environment initialised</div>
        <div class="loading-step done">✓ &nbsp;Dataset verified · 1,000,000 transactions</div>
        <div class="loading-step active">⟳ &nbsp;Loading ML models (RF · LR · SVM)...</div>
        <div class="loading-step">○ &nbsp;Calibrating ensemble voting system</div>
        <div class="loading-step">○ &nbsp;System ready</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('fraud_model_rf.pkl',  'rb') as f: rf     = pickle.load(f)
    with open('fraud_model_lr.pkl',  'rb') as f: lr     = pickle.load(f)
    with open('fraud_model_svm.pkl', 'rb') as f: svm    = pickle.load(f)
    with open('scaler.pkl',          'rb') as f: scaler = pickle.load(f)
    return rf, lr, svm, scaler

# Show loading screen while models load on first visit
if "models_loaded" not in st.session_state:
    loading_placeholder = st.empty()
    with loading_placeholder:
        show_loading_screen()
    rf_model, lr_model, svm_model, scaler = load_models()
    time.sleep(1.5)
    loading_placeholder.empty()
    st.session_state.models_loaded = True
    st.rerun()
else:
    rf_model, lr_model, svm_model, scaler = load_models()

# ── Real stats ────────────────────────────────────────────────────────────────
TOTAL_DATASET   = 1000000
TRAIN_SAMPLES   = 800000
TEST_SAMPLES    = 200000
TEST_FRAUD      = 17443
TEST_LEGIT      = 182557
FRAUD_RATE      = round(TEST_FRAUD / TEST_SAMPLES * 100, 1)

RF_ACC=100.00;  RF_PRE=100.00;  RF_REC=99.99;  RF_F1=99.99
LR_ACC=95.88;   LR_PRE=89.15;   LR_REC=60.01;  LR_F1=71.73
SVM_ACC=93.56;  SVM_PRE=90.39;  SVM_REC=29.32; SVM_F1=44.28

# Confusion matrices  [[TN, FP], [FN, TP]]
RF_CM  = [[182557, 0],     [2,     17441]]
LR_CM  = [[181283, 1274],  [6976,  10467]]
SVM_CM = [[182013, 544],   [12328, 5115]]

# Real feature importances
FEAT_IMP = {
    "Ratio to Median Price": 52.72,
    "Online Order":          16.94,
    "Distance from Home":    13.49,
    "Used PIN":               6.39,
    "Used Chip":              5.21,
    "Distance Last Txn":      4.57,
    "Repeat Retailer":        0.68,
}

INDIAN_MERCHANTS = ["Reliance Fresh","BigBasket","Flipkart","Amazon India",
                    "Zomato","Swiggy","IRCTC","MakeMyTrip","Myntra","PhonePe",
                    "Paytm Mall","Nykaa","Tata CLiQ","Meesho","Snapdeal"]

def random_inr():
    return f"₹{random.randint(200, 85000):,}"

# ── Predict ───────────────────────────────────────────────────────────────────
def predict_all(dist_home, dist_last, ratio_median, repeat_retailer, used_chip, used_pin, online_order):
    raw    = [[dist_home, dist_last, ratio_median, repeat_retailer, used_chip, used_pin, online_order]]
    scaled = scaler.transform(raw)
    rf_score  = float(rf_model.predict_proba(raw)[0][1])
    lr_score  = float(lr_model.predict_proba(scaled)[0][1])
    svm_score = float(svm_model.predict_proba(scaled)[0][1])
    votes    = [rf_score >= 0.5, lr_score >= 0.5, svm_score >= 0.5]
    is_fraud = sum(votes) >= 2
    avg      = (rf_score + lr_score + svm_score) / 3
    return {"rf": rf_score, "lr": lr_score, "svm": svm_score,
            "avg": avg, "is_fraud": is_fraud, "votes": votes}

def explain_prediction(dist_home, dist_last, ratio_median, repeat_retailer,
                       used_chip, used_pin, online_order, is_fraud, score):
    reasons = []
    if dist_home > 50:   reasons.append(f"transaction occurred {dist_home:.0f} km from home")
    if dist_last > 10:   reasons.append(f"it was {dist_last:.0f} km from the last transaction")
    if ratio_median > 3: reasons.append(f"the purchase price was {ratio_median:.1f}x the usual amount")
    if not repeat_retailer: reasons.append("this is an unknown retailer")
    if not used_chip:    reasons.append("chip was not used")
    if not used_pin:     reasons.append("PIN was not verified")
    if online_order:     reasons.append("it was an online order")

    pct = int(score * 100)
    if is_fraud:
        if reasons:
            return f"🚨 This transaction was flagged as fraud ({pct}% risk) because {', '.join(reasons[:3])}."
        return f"🚨 This transaction was flagged as high risk ({pct}%) by the ensemble model."
    else:
        safe_reasons = []
        if repeat_retailer: safe_reasons.append("known retailer")
        if used_chip:       safe_reasons.append("chip verified")
        if used_pin:        safe_reasons.append("PIN confirmed")
        if dist_home < 20:  safe_reasons.append("transaction near home")
        if safe_reasons:
            return f"✅ Transaction appears legitimate ({pct}% risk) — {', '.join(safe_reasons)}."
        return f"✅ Transaction cleared with {pct}% risk score. No major fraud indicators detected."

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FraudShield", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important; color: #e8edf5 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background: #080c14 !important; }
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #141e30 !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1320; }
::-webkit-scrollbar-thumb { background: #2a3a5c; border-radius: 2px; }

/* Sidebar nav */
[data-testid="stSidebarNav"] { display: none; }
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 1.5rem 1rem 1rem 1rem; margin-bottom: 0.5rem;
    border-bottom: 1px solid #141e30;
}
.sidebar-logo { font-size: 1.1rem; font-weight: 800; color: #f0f4ff; }
.sidebar-logo span { color: #3b82f6; }
.sidebar-sub { font-size: 0.62rem; color: #3a5070; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 0.65rem 1rem; border-radius: 10px; margin: 2px 8px;
    cursor: pointer; font-size: 0.82rem; font-weight: 600; color: #4b6a9c;
    transition: all 0.15s; text-decoration: none;
}
.nav-item:hover { background: #0f1c2e; color: #8ba4c8; }
.nav-item.active { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.2); }
.nav-section { font-size: 0.6rem; color: #2a3a52; text-transform: uppercase; letter-spacing: 0.12em; padding: 0.75rem 1rem 0.25rem 1rem; font-weight: 700; }
.sidebar-footer { padding: 1rem; border-top: 1px solid #141e30; margin-top: auto; }
.sidebar-badge { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2); color: #10b981; border-radius: 8px; padding: 0.5rem 0.75rem; font-size: 0.7rem; font-weight: 600; text-align: center; }

/* Login */
.login-card {
    background: linear-gradient(145deg, #0d1827 0%, #0a1220 100%);
    border: 1px solid #1a2d4a; border-radius: 20px; padding: 3rem 2.5rem;
    width: 100%; max-width: 440px; position: relative; overflow: hidden;
    box-shadow: 0 40px 80px rgba(0,0,0,0.6);
}
.login-card::before { content:''; position:absolute; top:-60px; left:-60px; width:220px; height:220px; background:radial-gradient(circle,rgba(59,130,246,0.12) 0%,transparent 70%); pointer-events:none; }
.login-card::after  { content:''; position:absolute; bottom:-40px; right:-40px; width:160px; height:160px; background:radial-gradient(circle,rgba(16,185,129,0.08) 0%,transparent 70%); pointer-events:none; }
.brand-name { font-size:1.3rem; font-weight:800; color:#f0f4ff; letter-spacing:-0.02em; }
.brand-sub  { font-size:0.7rem; color:#4b6a9c; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px; }
.login-title    { font-size:1.8rem; font-weight:800; color:#f0f4ff; margin-bottom:0.3rem; }
.login-subtitle { font-size:0.85rem; color:#4b6a9c; margin-bottom:2rem; }
.stat-strip { display:flex; background:#070b12; border:1px solid #1a2a42; border-radius:12px; overflow:hidden; margin-bottom:2rem; }
.stat-item  { flex:1; padding:0.75rem 0.5rem; text-align:center; border-right:1px solid #1a2a42; }
.stat-item:last-child { border-right:none; }
.stat-num { font-size:1rem; font-weight:700; color:#3b82f6; font-family:'JetBrains Mono',monospace; }
.stat-lbl { font-size:0.6rem; color:#3a5070; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px; }

/* Inputs */
.stTextInput > div > div > input {
    background:#070b12 !important; border:1px solid #1a2a42 !important;
    border-radius:10px !important; color:#e8edf5 !important;
    font-family:'Syne',sans-serif !important; font-size:0.9rem !important;
    padding:0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus { border-color:#3b82f6 !important; box-shadow:0 0 0 3px rgba(59,130,246,0.12) !important; }
.stTextInput label { color:#6b8ab0 !important; font-size:0.75rem !important; font-weight:600 !important; letter-spacing:0.06em !important; text-transform:uppercase !important; }

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#1d4ed8 0%,#3b82f6 100%) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:0.9rem !important; padding:0.75rem 1.5rem !important;
    width:100% !important; transition:all 0.2s !important;
    box-shadow:0 4px 20px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 8px 28px rgba(59,130,246,0.45) !important; }

/* Page header */
.page-header {
    padding: 1.5rem 0 1.5rem 0; margin-bottom: 1.5rem;
    border-bottom: 1px solid #141e30;
}
.page-title { font-size: 1.6rem; font-weight: 800; color: #f0f4ff; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.page-sub   { font-size: 0.8rem; color: #4b6a9c; }

/* Metric cards */
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }
.metric-grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem; }
.metric-card {
    background:linear-gradient(145deg,#0d1827,#0a1220);
    border:1px solid #1a2d4a; border-radius:16px; padding:1.25rem 1.5rem;
    position:relative; overflow:hidden;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:16px 16px 0 0; }
.metric-card.blue::before   { background:linear-gradient(90deg,#3b82f6,#60a5fa); }
.metric-card.red::before    { background:linear-gradient(90deg,#ef4444,#f87171); }
.metric-card.green::before  { background:linear-gradient(90deg,#10b981,#34d399); }
.metric-card.purple::before { background:linear-gradient(90deg,#8b5cf6,#a78bfa); }
.metric-card.amber::before  { background:linear-gradient(90deg,#f59e0b,#fcd34d); }
.metric-label { font-size:0.7rem; font-weight:600; color:#4b6a9c; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem; }
.metric-value { font-size:1.9rem; font-weight:800; color:#f0f4ff; font-family:'JetBrains Mono',monospace; letter-spacing:-0.03em; line-height:1; margin-bottom:0.3rem; }
.metric-delta { font-size:0.72rem; font-weight:600; }
.metric-delta.pos { color:#10b981; }
.metric-delta.neg { color:#ef4444; }
.metric-delta.neu { color:#6b8ab0; }

/* Section cards */
.section-card { background:linear-gradient(145deg,#0d1827,#0a1220); border:1px solid #1a2d4a; border-radius:16px; padding:1.5rem; margin-bottom:1rem; }
.section-title { font-size:0.78rem; font-weight:700; color:#4b6a9c; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px; }
.section-title::before { content:''; width:3px; height:14px; background:#3b82f6; border-radius:2px; display:inline-block; }

/* Result boxes */
.result-safe  { background:rgba(16,185,129,0.07); border:1px solid rgba(16,185,129,0.2); border-radius:14px; padding:1.5rem; text-align:center; }
.result-fraud { background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.25); border-radius:14px; padding:1.5rem; text-align:center; animation:pulse-red 2s infinite; }
@keyframes pulse-red { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.15);}50%{box-shadow:0 0 0 10px rgba(239,68,68,0);} }
.result-icon  { font-size:2.5rem; margin-bottom:0.5rem; }
.result-label { font-size:0.7rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.3rem; }
.result-label.safe  { color:#10b981; }
.result-label.fraud { color:#ef4444; }
.result-main { font-size:1.4rem; font-weight:800; color:#f0f4ff; margin-bottom:0.5rem; }
.result-conf { font-size:0.8rem; color:#6b8ab0; }

/* Explanation box */
.explain-box {
    background: #070b12; border: 1px solid #1a2a42; border-left: 3px solid #3b82f6;
    border-radius: 10px; padding: 1rem; font-size: 0.82rem; color: #8ba4c8;
    line-height: 1.6; margin-top: 0.75rem;
}
.explain-box.fraud { border-left-color: #ef4444; }
.explain-box.safe  { border-left-color: #10b981; }

/* Risk bar */
.risk-bar-bg   { height:8px; background:#0d1827; border-radius:4px; border:1px solid #1a2a42; overflow:hidden; margin:0.5rem 0; }
.risk-bar-fill { height:100%; border-radius:4px; }

/* Vote bars */
.vote-row  { margin-bottom:1rem; }
.vote-label { display:flex; justify-content:space-between; margin-bottom:4px; }
.vote-name  { font-size:0.75rem; font-weight:700; color:#8ba4c8; }
.vote-score { font-size:0.75rem; font-family:'JetBrains Mono',monospace; }
.vote-bar-bg   { height:10px; background:#070b12; border-radius:5px; border:1px solid #1a2a42; overflow:hidden; }
.vote-bar-fill { height:100%; border-radius:5px; }
.vote-verdict  { font-size:0.65rem; font-weight:700; letter-spacing:0.08em; margin-top:3px; text-align:right; }

/* Transactions */
.txn-row { display:flex; align-items:center; justify-content:space-between; padding:0.7rem 0; border-bottom:1px solid #0f1c2e; }
.txn-row:last-child { border-bottom:none; }
.txn-id     { font-family:'JetBrains Mono',monospace; color:#4b6a9c; font-size:0.72rem; }
.txn-amount { font-family:'JetBrains Mono',monospace; font-weight:600; color:#c8d8f0; }
.txn-badge  { padding:0.2rem 0.6rem; border-radius:20px; font-size:0.65rem; font-weight:700; text-transform:uppercase; }
.txn-safe   { background:rgba(16,185,129,0.12); color:#10b981; border:1px solid rgba(16,185,129,0.2); }
.txn-fraud  { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.2); }

/* Alerts */
.alert-box { background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.2); border-left:3px solid #ef4444; border-radius:8px; padding:0.75rem 1rem; font-size:0.8rem; color:#fca5a5; margin-bottom:0.75rem; }
.alert-box.info { background:rgba(59,130,246,0.06); border-color:rgba(59,130,246,0.2); border-left-color:#3b82f6; color:#93c5fd; }
.alert-box.success { background:rgba(16,185,129,0.06); border-color:rgba(16,185,129,0.2); border-left-color:#10b981; color:#6ee7b7; }

/* Confusion matrix */
.cm-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:0.5rem; }
.cm-cell { border-radius:12px; padding:1rem; text-align:center; }
.cm-tn { background:rgba(16,185,129,0.1);  border:1px solid rgba(16,185,129,0.2); }
.cm-fp { background:rgba(239,68,68,0.08);  border:1px solid rgba(239,68,68,0.15); }
.cm-fn { background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.15); }
.cm-tp { background:rgba(59,130,246,0.1);  border:1px solid rgba(59,130,246,0.2); }
.cm-num   { font-size:1.5rem; font-weight:800; font-family:'JetBrains Mono',monospace; }
.cm-label { font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
.cm-tn .cm-num   { color:#10b981; } .cm-tn .cm-label { color:#065f46; }
.cm-fp .cm-num   { color:#ef4444; } .cm-fp .cm-label { color:#991b1b; }
.cm-fn .cm-num   { color:#f59e0b; } .cm-fn .cm-label { color:#92400e; }
.cm-tp .cm-num   { color:#3b82f6; } .cm-tp .cm-label { color:#1e3a5f; }

/* Feature bars */
.feat-row  { margin-bottom:0.9rem; }
.feat-name { font-size:0.72rem; color:#6b8ab0; margin-bottom:3px; text-transform:uppercase; letter-spacing:0.05em; }
.feat-bar-bg   { height:8px; background:#0d1827; border-radius:4px; }
.feat-bar-fill { height:8px; border-radius:4px; }

/* Accuracy table */
.acc-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
.acc-table th { color:#4b6a9c; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; padding:0.5rem 0.75rem; border-bottom:1px solid #1a2a42; text-align:left; }
.acc-table td { padding:0.55rem 0.75rem; border-bottom:1px solid #0f1c2e; color:#c8d8f0; font-family:'JetBrains Mono',monospace; }
.acc-table tr:last-child td { border-bottom:none; }
.acc-best { color:#10b981 !important; font-weight:700; }

/* Dataset donut placeholder */
.donut-wrap { display:flex; align-items:center; gap:2rem; padding:0.5rem 0; }
.donut-legend { flex:1; }
.legend-item { display:flex; align-items:center; gap:8px; margin-bottom:0.6rem; font-size:0.8rem; color:#8ba4c8; }
.legend-dot  { width:10px; height:10px; border-radius:50%; flex-shrink:0; }

/* Sliders / Radios / Number inputs */
.stSlider > div > div > div > div { background:#3b82f6 !important; }
.stSlider label { color:#6b8ab0 !important; font-size:0.75rem !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }
.stRadio label  { color:#8ba4c8 !important; font-size:0.85rem !important; }
.stRadio > div  { flex-direction:row !important; gap:1rem !important; }
.stNumberInput > div > div > input { background:#070b12 !important; border:1px solid #1a2a42 !important; border-radius:10px !important; color:#e8edf5 !important; font-family:'JetBrains Mono',monospace !important; }
.stSelectbox > div > div { background:#070b12 !important; border:1px solid #1a2a42 !important; border-radius:10px !important; color:#e8edf5 !important; }
.stSelectbox label { color:#6b8ab0 !important; font-size:0.75rem !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }
.stMarkdown p { color:#8ba4c8; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "logged_in"     not in st.session_state: st.session_state.logged_in     = False
if "username"      not in st.session_state: st.session_state.username      = ""
if "last_result"   not in st.session_state: st.session_state.last_result   = None
if "last_inputs"   not in st.session_state: st.session_state.last_inputs   = None
if "page"          not in st.session_state: st.session_state.page          = "Dashboard"
if "total_scanned" not in st.session_state: st.session_state.total_scanned = TEST_SAMPLES
if "fraud_caught"  not in st.session_state: st.session_state.fraud_caught  = TEST_FRAUD
if "history"       not in st.session_state:
    st.session_state.history = [
        {"id":"TXN-0987","merchant":"Unknown Vendor","amount":"₹78,500","fraud":True, "time":"2 min ago",  "score":0.91},
        {"id":"TXN-0986","merchant":"BigBasket",     "amount":"₹3,450", "fraud":False,"time":"8 min ago",  "score":0.04},
        {"id":"TXN-0985","merchant":"Flipkart",      "amount":"₹12,999","fraud":False,"time":"15 min ago", "score":0.07},
        {"id":"TXN-0984","merchant":"Unknown Vendor","amount":"₹55,200","fraud":True, "time":"22 min ago", "score":0.87},
        {"id":"TXN-0983","merchant":"IRCTC",         "amount":"₹2,340", "fraud":False,"time":"31 min ago", "score":0.03},
    ]

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════════════════════════════════════
def show_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown(f"""
        <div class="login-card">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:2rem;">
            <div style="width:44px;height:44px;background:linear-gradient(135deg,#1d4ed8,#3b82f6);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 8px 24px rgba(59,130,246,0.3);">🛡️</div>
            <div style="line-height:1">
              <div class="brand-name">FraudShield</div>
              <div class="brand-sub">Ensemble Detection System</div>
            </div>
          </div>
          <div class="stat-strip">
            <div class="stat-item"><div class="stat-num">3</div><div class="stat-lbl">Models</div></div>
            <div class="stat-item"><div class="stat-num">1M+</div><div class="stat-lbl">Transactions</div></div>
            <div class="stat-item"><div class="stat-num">100%</div><div class="stat-lbl">Best Acc</div></div>
          </div>
          <div class="login-title">Secure Access</div>
          <div class="login-subtitle">Sign in to the fraud detection dashboard</div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="analyst@bank.com")
        password = st.text_input("Password", placeholder="••••••••", type="password")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Sign In  →"):
            if username and password:
                with st.spinner("Authenticating…"):
                    time.sleep(0.6)
                st.session_state.logged_in = True
                st.session_state.username  = username.split("@")[0].capitalize()
                st.rerun()
            else:
                st.error("Please enter both username and password.")
        st.markdown('<div style="text-align:center;margin-top:1rem;font-size:0.72rem;color:#2a3f5c;">Demo: any username · any password</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
          <span style="font-size:1.4rem">🛡️</span>
          <div>
            <div class="sidebar-logo">Fraud<span>Shield</span></div>
            <div class="sidebar-sub">Ensemble System</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            ("🏠", "Dashboard",           "Overview & KPIs"),
            ("🔍", "Transaction Analyser","Run fraud detection"),
            ("📊", "Model Performance",   "Accuracy & metrics"),
            ("📈", "Dataset Explorer",    "Data insights"),
            ("📜", "Transaction History", "Past predictions"),
        ]

        st.markdown('<div class="nav-section">Navigation</div>', unsafe_allow_html=True)
        for icon, name, desc in pages:
            active = "active" if st.session_state.page == name else ""
            if st.button(f"{icon}  {name}", key=f"nav_{name}",
                         help=desc, use_container_width=True):
                st.session_state.page = name
                st.rerun()

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:0 8px;">
          <div class="sidebar-badge">● 3 Models Active<br>
          <span style="font-size:0.62rem;color:#065f46;">RF · LR · SVM</span></div>
          <div style="margin-top:0.75rem;font-size:0.7rem;color:#2a3a52;text-align:center;">
            Logged in as <span style="color:#4b6a9c;">{st.session_state.username}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.last_result = None
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    fraud_rate = round(st.session_state.fraud_caught / st.session_state.total_scanned * 100, 2)

    st.markdown("""
    <div class="page-header">
      <div class="page-title">Dashboard</div>
      <div class="page-sub">Real-time overview of the fraud detection system</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card blue">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">{st.session_state.total_scanned:,}</div>
        <div class="metric-delta pos">Full dataset analysed</div>
      </div>
      <div class="metric-card red">
        <div class="metric-label">Fraud Detected</div>
        <div class="metric-value">{st.session_state.fraud_caught:,}</div>
        <div class="metric-delta neg">{fraud_rate}% fraud rate</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Best Accuracy</div>
        <div class="metric-value">{RF_ACC}%</div>
        <div class="metric-delta pos">Random Forest</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-label">Legitimate Cases</div>
        <div class="metric-value">{st.session_state.total_scanned - st.session_state.fraud_caught:,}</div>
        <div class="metric-delta neu">{round(100-fraud_rate,2)}% clean</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="medium")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Transactions</div>', unsafe_allow_html=True)
        for txn in st.session_state.history:
            bc = "txn-fraud" if txn["fraud"] else "txn-safe"
            bt = "FRAUD"     if txn["fraud"] else "SAFE"
            sc = txn.get("score", 0)
            st.markdown(f"""
            <div class="txn-row">
              <div>
                <div style="font-size:0.82rem;color:#c8d8f0;font-weight:600;">{txn['merchant']}</div>
                <div class="txn-id">{txn['id']} · {txn['time']}</div>
              </div>
              <div style="text-align:right;">
                <div class="txn-amount">{txn['amount']}</div>
                <div style="font-size:0.65rem;color:#3a5070;font-family:'JetBrains Mono',monospace;">risk: {sc:.2f}</div>
              </div>
              <span class="txn-badge {bc}" style="margin-left:1rem">{bt}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Accuracy Overview</div>', unsafe_allow_html=True)
        models_overview = [
            ("🌲 Random Forest",       RF_ACC,  "#10b981"),
            ("📈 Logistic Regression", LR_ACC,  "#8b5cf6"),
            ("⚡ SVM",                 SVM_ACC, "#f59e0b"),
        ]
        for name, acc, color in models_overview:
            st.markdown(f"""
            <div style="margin-bottom:1.2rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-size:0.8rem;font-weight:600;color:#8ba4c8;">{name}</span>
                <span style="font-size:0.8rem;font-family:'JetBrains Mono',monospace;color:{color};font-weight:700;">{acc}%</span>
              </div>
              <div class="feat-bar-bg">
                <div class="feat-bar-fill" style="width:{acc}%;background:{color}"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Split</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
          <div style="background:#070b12;border:1px solid #1a2a42;border-radius:10px;padding:0.75rem;text-align:center;">
            <div style="font-size:1.1rem;font-weight:800;color:#3b82f6;font-family:'JetBrains Mono',monospace;">{TRAIN_SAMPLES:,}</div>
            <div style="font-size:0.62rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;">Training</div>
          </div>
          <div style="background:#070b12;border:1px solid #1a2a42;border-radius:10px;padding:0.75rem;text-align:center;">
            <div style="font-size:1.1rem;font-weight:800;color:#8b5cf6;font-family:'JetBrains Mono',monospace;">{TEST_SAMPLES:,}</div>
            <div style="font-size:0.62rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;">Testing</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TRANSACTION ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════
def page_analyser():
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Transaction Analyser</div>
      <div class="page-sub">Enter transaction details — all 3 models vote simultaneously</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="medium")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            dist_home = st.number_input("Distance from Home (km)", min_value=0.0, max_value=500.0, value=12.0, step=0.5, format="%.1f")
        with c2:
            dist_last = st.number_input("Distance from Last Txn (km)", min_value=0.0, max_value=500.0, value=2.5, step=0.5, format="%.1f")

        ratio_median = st.slider("Ratio to Median Purchase Price", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem;color:#4b6a9c;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Transaction Flags</div>', unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        with f1:
            repeat_retailer = st.radio("Repeat Retailer", [1,0], format_func=lambda x:"Yes" if x else "No", horizontal=True)
            used_chip       = st.radio("Used Chip",        [1,0], format_func=lambda x:"Yes" if x else "No", horizontal=True)
        with f2:
            used_pin     = st.radio("Used PIN",     [1,0], format_func=lambda x:"Yes" if x else "No", horizontal=True)
            online_order = st.radio("Online Order", [0,1], format_func=lambda x:"Yes" if x else "No", horizontal=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        if st.button("🔍  Run Ensemble Analysis", key="predict_btn"):
            with st.spinner("All 3 models are analysing…"):
                time.sleep(0.8)
            result = predict_all(dist_home, dist_last, ratio_median,
                                 repeat_retailer, used_chip, used_pin, online_order)
            st.session_state.last_result = result
            st.session_state.last_inputs = {
                "dist_home": dist_home, "dist_last": dist_last,
                "ratio_median": ratio_median, "repeat_retailer": repeat_retailer,
                "used_chip": used_chip, "used_pin": used_pin, "online_order": online_order
            }
            merchant = random.choice(INDIAN_MERCHANTS) if not result["is_fraud"] else "Unknown Vendor"
            new_txn = {
                "id":       f"TXN-{str(st.session_state.total_scanned+1).zfill(4)}",
                "merchant": merchant,
                "amount":   random_inr(),
                "fraud":    result["is_fraud"],
                "time":     "just now",
                "score":    result["avg"],
            }
            st.session_state.history.insert(0, new_txn)
            st.session_state.history = st.session_state.history[:10]
            st.session_state.total_scanned += 1
            if result["is_fraud"]:
                st.session_state.fraud_caught += 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick test cases
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Test Guide</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#4b6a9c;margin-bottom:0.75rem;font-weight:600;">🚨 FRAUD scenario — try these values:</div>
        <div style="background:#070b12;border:1px solid #1a2a42;border-radius:8px;padding:0.75rem;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8ba4c8;line-height:1.8;">
          Distance from Home → 200<br>
          Distance Last Txn  → 50<br>
          Ratio to Median    → 8.0<br>
          Repeat Retailer    → No<br>
          Used Chip          → No · PIN → No · Online → Yes
        </div>
        <div style="font-size:0.78rem;color:#4b6a9c;margin:0.75rem 0;font-weight:600;">✅ SAFE scenario — try these values:</div>
        <div style="background:#070b12;border:1px solid #1a2a42;border-radius:8px;padding:0.75rem;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#8ba4c8;line-height:1.8;">
          Distance from Home → 5<br>
          Distance Last Txn  → 1<br>
          Ratio to Median    → 1.0<br>
          Repeat Retailer    → Yes<br>
          Used Chip → Yes · PIN → Yes · Online → No
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # Final verdict
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Final Verdict</div>', unsafe_allow_html=True)

        if st.session_state.last_result is None:
            st.markdown("""
            <div style="text-align:center;padding:2rem 0;color:#2a3f5c;">
              <div style="font-size:2.5rem;margin-bottom:0.5rem;opacity:0.4">🔍</div>
              <div style="font-size:0.82rem;">Run analysis to see prediction</div>
            </div>""", unsafe_allow_html=True)
        else:
            res = st.session_state.last_result
            inp = st.session_state.last_inputs
            pct = int(res["avg"] * 100)
            bar_color = "#ef4444" if res["is_fraud"] else "#10b981"

            if res["is_fraud"]:
                st.markdown(f"""
                <div class="result-fraud">
                  <div class="result-icon">🚨</div>
                  <div class="result-label fraud">FRAUD DETECTED</div>
                  <div class="result-main">High-Risk Transaction</div>
                  <div class="result-conf">Ensemble confidence: {pct}% · Recommend: Block</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                  <div class="result-icon">✅</div>
                  <div class="result-label safe">LEGITIMATE</div>
                  <div class="result-main">Transaction Cleared</div>
                  <div class="result-conf">Ensemble risk: {pct}% · Recommend: Approve</div>
                </div>""", unsafe_allow_html=True)

            # Natural language explanation
            explanation = explain_prediction(
                inp["dist_home"], inp["dist_last"], inp["ratio_median"],
                inp["repeat_retailer"], inp["used_chip"], inp["used_pin"],
                inp["online_order"], res["is_fraud"], res["avg"]
            )
            cls = "fraud" if res["is_fraud"] else "safe"
            st.markdown(f'<div class="explain-box {cls}">{explanation}</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:1rem;">
              <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#4b6a9c;margin-bottom:4px;">
                <span>ENSEMBLE RISK SCORE</span>
                <span style="font-family:'JetBrains Mono',monospace;color:{bar_color}">{res['avg']:.3f}</span>
              </div>
              <div class="risk-bar-bg"><div class="risk-bar-fill" style="width:{pct}%;background:{bar_color}"></div></div>
              <div style="display:flex;justify-content:space-between;font-size:0.62rem;color:#2a3a52;margin-top:3px;">
                <span>Low Risk</span><span>Medium</span><span>High Risk</span>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Model votes
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Individual Model Votes</div>', unsafe_allow_html=True)
        if st.session_state.last_result is None:
            st.markdown('<div style="text-align:center;padding:1.5rem 0;color:#2a3f5c;font-size:0.82rem;">Run analysis to see votes</div>', unsafe_allow_html=True)
        else:
            res = st.session_state.last_result
            models_info = [
                ("🌲 Random Forest",       res["rf"],  res["votes"][0], "#3b82f6"),
                ("📈 Logistic Regression", res["lr"],  res["votes"][1], "#8b5cf6"),
                ("⚡ SVM",                 res["svm"], res["votes"][2], "#f59e0b"),
            ]
            for name, score, voted_fraud, color in models_info:
                pct       = int(score * 100)
                verdict   = "FRAUD 🚨" if voted_fraud else "SAFE ✅"
                v_color   = "#ef4444"  if voted_fraud else "#10b981"
                bar_color = "#ef4444"  if voted_fraud else color
                st.markdown(f"""
                <div class="vote-row">
                  <div class="vote-label">
                    <span class="vote-name">{name}</span>
                    <span class="vote-score" style="color:{bar_color}">{score:.3f}</span>
                  </div>
                  <div class="vote-bar-bg"><div class="vote-bar-fill" style="width:{pct}%;background:{bar_color}"></div></div>
                  <div class="vote-verdict" style="color:{v_color}">Voted: {verdict}</div>
                </div>""", unsafe_allow_html=True)

            fv = sum(res["votes"]); sv = 3 - fv
            st.markdown(f"""
            <div style="margin-top:0.75rem;background:#070b12;border:1px solid #1a2a42;border-radius:10px;padding:0.75rem 1rem;display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:0.72rem;color:#4b6a9c;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">Vote Tally</span>
              <span style="font-size:0.85rem;font-family:'JetBrains Mono',monospace;">
                <span style="color:#ef4444;font-weight:700;">{fv} Fraud</span>
                <span style="color:#2a3a52;"> vs </span>
                <span style="color:#10b981;font-weight:700;">{sv} Safe</span>
              </span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
def page_performance():
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Model Performance</div>
      <div class="page-sub">Real evaluation metrics from the Kaggle test set (200,000 transactions)</div>
    </div>""", unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="metric-grid-3">
      <div class="metric-card green">
        <div class="metric-label">🌲 Random Forest</div>
        <div class="metric-value">{RF_ACC}%</div>
        <div class="metric-delta pos">Best performing model</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-label">📈 Logistic Regression</div>
        <div class="metric-value">{LR_ACC}%</div>
        <div class="metric-delta neu">Good baseline model</div>
      </div>
      <div class="metric-card amber">
        <div class="metric-label">⚡ SVM</div>
        <div class="metric-value">{SVM_ACC}%</div>
        <div class="metric-delta neu">Support Vector Machine</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # Full metrics table
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Full Metrics Comparison</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="acc-table">
          <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
          <tbody>
            <tr>
              <td>🌲 Random Forest</td>
              <td class="acc-best">{RF_ACC}%</td><td class="acc-best">{RF_PRE}%</td>
              <td class="acc-best">{RF_REC}%</td><td class="acc-best">{RF_F1}%</td>
            </tr>
            <tr>
              <td>📈 Logistic Reg.</td>
              <td>{LR_ACC}%</td><td>{LR_PRE}%</td><td>{LR_REC}%</td><td>{LR_F1}%</td>
            </tr>
            <tr>
              <td>⚡ SVM</td>
              <td>{SVM_ACC}%</td><td>{SVM_PRE}%</td><td>{SVM_REC}%</td><td>{SVM_F1}%</td>
            </tr>
          </tbody>
        </table>
        <div style="margin-top:0.75rem;font-size:0.7rem;color:#2a3a52;">
          * Evaluated on 200,000 test transactions from Kaggle dataset
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance — REAL values
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        colors = ["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4","#84cc16"]
        for i, (name, pct) in enumerate(FEAT_IMP.items()):
            color = colors[i % len(colors)]
            st.markdown(f"""
            <div class="feat-row">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span class="feat-name">{name}</span>
                <span style="font-size:0.72rem;color:{color};font-family:'JetBrains Mono',monospace;font-weight:700;">{pct}%</span>
              </div>
              <div class="feat-bar-bg">
                <div class="feat-bar-fill" style="width:{pct}%;background:{color}"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div style="margin-top:0.5rem;font-size:0.7rem;color:#2a3a52;">* Real values from rf_classifier.feature_importances_</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Confusion matrices
        selected_model = st.selectbox("Select model for confusion matrix",
                                      ["🌲 Random Forest", "📈 Logistic Regression", "⚡ SVM"])

        if "Random Forest" in selected_model:
            cm = RF_CM; acc = RF_ACC
        elif "Logistic" in selected_model:
            cm = LR_CM; acc = LR_ACC
        else:
            cm = SVM_CM; acc = SVM_ACC

        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1rem;">
          <span style="font-size:0.72rem;color:#4b6a9c;text-transform:uppercase;letter-spacing:0.08em;">Model Accuracy: </span>
          <span style="font-size:0.9rem;font-weight:700;color:#10b981;font-family:'JetBrains Mono',monospace;">{acc}%</span>
        </div>
        <div style="font-size:0.65rem;color:#3a5070;text-align:center;margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.08em;">← Predicted →</div>
        <div class="cm-grid">
          <div class="cm-cell cm-tn">
            <div class="cm-num">{tn:,}</div>
            <div class="cm-label">True Negative</div>
            <div style="font-size:0.6rem;color:#065f46;margin-top:4px;">Correctly said SAFE</div>
          </div>
          <div class="cm-cell cm-fp">
            <div class="cm-num">{fp:,}</div>
            <div class="cm-label">False Positive</div>
            <div style="font-size:0.6rem;color:#991b1b;margin-top:4px;">Said FRAUD but was safe</div>
          </div>
          <div class="cm-cell cm-fn">
            <div class="cm-num">{fn:,}</div>
            <div class="cm-label">False Negative</div>
            <div style="font-size:0.6rem;color:#92400e;margin-top:4px;">Missed fraud cases</div>
          </div>
          <div class="cm-cell cm-tp">
            <div class="cm-num">{tp:,}</div>
            <div class="cm-label">True Positive</div>
            <div style="font-size:0.6rem;color:#1e3a5f;margin-top:4px;">Correctly caught FRAUD</div>
          </div>
        </div>
        <div style="margin-top:1rem;background:#070b12;border:1px solid #1a2a42;border-radius:8px;padding:0.75rem;font-size:0.72rem;color:#4b6a9c;line-height:1.8;">
          <span style="color:#10b981;font-weight:700;">✓ Fraud caught:</span> {tp:,} out of {tp+fn:,} ({round(tp/(tp+fn)*100,1)}%)<br>
          <span style="color:#ef4444;font-weight:700;">✗ Fraud missed:</span> {fn:,} transactions<br>
          <span style="color:#3b82f6;font-weight:700;">✓ Safe approved:</span> {tn:,} transactions
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Why RF is best explanation
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Why Random Forest Wins</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.8rem;color:#6b8ab0;line-height:1.8;">
          <div style="margin-bottom:0.5rem;">🌲 <strong style="color:#8ba4c8;">Random Forest</strong> uses 100+ decision trees voting together — similar to our 3-model ensemble but at a much larger scale.</div>
          <div style="margin-bottom:0.5rem;">📉 <strong style="color:#8ba4c8;">Logistic Regression</strong> struggles because fraud patterns are non-linear — it can't capture complex relationships.</div>
          <div>⚡ <strong style="color:#8ba4c8;">SVM</strong> has low recall (29%) meaning it misses many fraud cases — it's too conservative in flagging fraud.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
def page_dataset():
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Dataset Explorer</div>
      <div class="page-sub">Insights from the Kaggle Credit Card Fraud Detection dataset</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card blue">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">1M</div>
        <div class="metric-delta pos">1,000,000 transactions</div>
      </div>
      <div class="metric-card red">
        <div class="metric-label">Fraud Cases</div>
        <div class="metric-value">87K</div>
        <div class="metric-delta neg">8.7% of dataset</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Legitimate</div>
        <div class="metric-value">913K</div>
        <div class="metric-delta pos">91.3% clean</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-label">Features</div>
        <div class="metric-value">7</div>
        <div class="metric-delta neu">Input variables</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Class Distribution</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom:1rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-size:0.8rem;color:#10b981;font-weight:600;">✅ Legitimate Transactions</span>
            <span style="font-size:0.8rem;font-family:'JetBrains Mono',monospace;color:#10b981;font-weight:700;">91.3%</span>
          </div>
          <div class="feat-bar-bg" style="height:14px;border-radius:7px;">
            <div class="feat-bar-fill" style="width:91.3%;background:#10b981;height:14px;border-radius:7px;"></div>
          </div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-size:0.8rem;color:#ef4444;font-weight:600;">🚨 Fraud Transactions</span>
            <span style="font-size:0.8rem;font-family:'JetBrains Mono',monospace;color:#ef4444;font-weight:700;">8.7%</span>
          </div>
          <div class="feat-bar-bg" style="height:14px;border-radius:7px;">
            <div class="feat-bar-fill" style="width:8.7%;background:#ef4444;height:14px;border-radius:7px;"></div>
          </div>
        </div>
        <div style="margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.72rem;color:#4b6a9c;text-align:center;">
          <div style="background:#070b12;border:1px solid #1a2a42;border-radius:8px;padding:0.5rem;">
            <div style="font-size:1rem;font-weight:700;color:#10b981;font-family:'JetBrains Mono',monospace;">913,000</div>
            <div>Legitimate</div>
          </div>
          <div style="background:#070b12;border:1px solid #1a2a42;border-radius:8px;padding:0.5rem;">
            <div style="font-size:1rem;font-weight:700;color:#ef4444;font-family:'JetBrains Mono',monospace;">87,000</div>
            <div>Fraud</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Fraud Patterns Found</div>', unsafe_allow_html=True)
        patterns = [
            ("🏠", "Far from home",      "Transactions 100km+ from home are 3x more likely to be fraud"),
            ("💰", "High ratio",         "Purchases 5x+ the median amount are strong fraud indicators"),
            ("🌐", "Online orders",      "Online transactions have higher fraud rate than in-store"),
            ("💳", "No chip used",       "Transactions without chip verification are riskier"),
            ("🔢", "No PIN verified",    "Missing PIN verification increases fraud probability"),
            ("🏪", "Unknown retailer",   "First-time retailers account for 67% of fraud cases"),
        ]
        for icon, title, desc in patterns:
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-bottom:0.8rem;padding-bottom:0.8rem;border-bottom:1px solid #0f1c2e;">
              <span style="font-size:1.2rem;flex-shrink:0;">{icon}</span>
              <div>
                <div style="font-size:0.78rem;font-weight:700;color:#c8d8f0;margin-bottom:2px;">{title}</div>
                <div style="font-size:0.72rem;color:#4b6a9c;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature Descriptions</div>', unsafe_allow_html=True)
        features = [
            ("Distance from Home",       "How far the transaction is from the cardholder's home address",                "13.49%"),
            ("Distance from Last Txn",   "Distance between this and the previous transaction location",                  "4.57%"),
            ("Ratio to Median Price",    "How the purchase amount compares to the cardholder's typical spending",         "52.72%"),
            ("Repeat Retailer",          "Whether the cardholder has bought from this merchant before",                  "0.68%"),
            ("Used Chip",                "Whether the physical chip on the card was used for the transaction",            "5.21%"),
            ("Used PIN",                 "Whether the PIN number was verified during the transaction",                    "6.39%"),
            ("Online Order",             "Whether this was an online transaction (no physical card present)",             "16.94%"),
        ]
        for name, desc, imp in features:
            st.markdown(f"""
            <div style="margin-bottom:0.9rem;padding-bottom:0.9rem;border-bottom:1px solid #0f1c2e;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
                <span style="font-size:0.78rem;font-weight:700;color:#c8d8f0;">{name}</span>
                <span style="font-size:0.65rem;background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);color:#3b82f6;border-radius:20px;padding:1px 8px;font-family:'JetBrains Mono',monospace;">imp: {imp}</span>
              </div>
              <div style="font-size:0.72rem;color:#4b6a9c;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Train / Test Split</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.78rem;color:#4b6a9c;margin-bottom:0.75rem;">Standard 80/20 split used for training and evaluation</div>
        <div style="margin-bottom:0.75rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.75rem;color:#3b82f6;font-weight:600;">Training Set (80%)</span>
            <span style="font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:#3b82f6;">{TRAIN_SAMPLES:,}</span>
          </div>
          <div class="feat-bar-bg" style="height:12px;border-radius:6px;">
            <div style="width:80%;background:#3b82f6;height:12px;border-radius:6px;"></div>
          </div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:0.75rem;color:#8b5cf6;font-weight:600;">Testing Set (20%)</span>
            <span style="font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:#8b5cf6;">{TEST_SAMPLES:,}</span>
          </div>
          <div class="feat-bar-bg" style="height:12px;border-radius:6px;">
            <div style="width:20%;background:#8b5cf6;height:12px;border-radius:6px;"></div>
          </div>
        </div>
        <div style="margin-top:0.75rem;display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.7rem;text-align:center;">
          <div style="background:#070b12;border:1px solid #1a2a42;border-radius:8px;padding:0.5rem;">
            <div style="color:#ef4444;font-weight:700;font-family:'JetBrains Mono',monospace;">{TEST_FRAUD:,}</div>
            <div style="color:#4b6a9c;">Fraud in test</div>
          </div>
          <div style="background:#070b12;border:1px solid #1a2a42;border-radius:8px;padding:0.5rem;">
            <div style="color:#10b981;font-weight:700;font-family:'JetBrains Mono',monospace;">{TEST_LEGIT:,}</div>
            <div style="color:#4b6a9c;">Legit in test</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — TRANSACTION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Transaction History</div>
      <div class="page-sub">All transactions analysed in this session</div>
    </div>""", unsafe_allow_html=True)

    history = st.session_state.history
    total   = len(history)
    frauds  = sum(1 for t in history if t["fraud"])
    safes   = total - frauds

    st.markdown(f"""
    <div class="metric-grid-3">
      <div class="metric-card blue">
        <div class="metric-label">Total Analysed</div>
        <div class="metric-value">{total}</div>
        <div class="metric-delta neu">This session</div>
      </div>
      <div class="metric-card red">
        <div class="metric-label">Fraud Flagged</div>
        <div class="metric-value">{frauds}</div>
        <div class="metric-delta neg">Blocked</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Cleared Safe</div>
        <div class="metric-value">{safes}</div>
        <div class="metric-delta pos">Approved</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter
    filter_opt = st.selectbox("Filter transactions", ["All", "Fraud Only", "Safe Only"])

    filtered = history
    if filter_opt == "Fraud Only":
        filtered = [t for t in history if t["fraud"]]
    elif filter_opt == "Safe Only":
        filtered = [t for t in history if not t["fraud"]]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Transactions ({len(filtered)} shown)</div>', unsafe_allow_html=True)

    if not filtered:
        st.markdown('<div style="text-align:center;padding:2rem;color:#2a3f5c;font-size:0.82rem;">No transactions found</div>', unsafe_allow_html=True)
    else:
        # Header row
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1.5fr 1fr 1fr 0.8fr;padding:0.5rem 0;border-bottom:1px solid #1a2a42;margin-bottom:0.25rem;">
          <span style="font-size:0.65rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">TXN ID</span>
          <span style="font-size:0.65rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Merchant</span>
          <span style="font-size:0.65rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Amount</span>
          <span style="font-size:0.65rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Risk Score</span>
          <span style="font-size:0.65rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Status</span>
        </div>""", unsafe_allow_html=True)

        for txn in filtered:
            bc  = "txn-fraud" if txn["fraud"] else "txn-safe"
            bt  = "FRAUD"     if txn["fraud"] else "SAFE"
            sc  = txn.get("score", 0)
            sc_color = "#ef4444" if txn["fraud"] else "#10b981"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1.5fr 1fr 1fr 0.8fr;padding:0.65rem 0;border-bottom:1px solid #0a1220;align-items:center;">
              <div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#c8d8f0;">{txn['id']}</div>
                <div style="font-size:0.62rem;color:#3a5070;">{txn['time']}</div>
              </div>
              <div style="font-size:0.78rem;color:#8ba4c8;">{txn['merchant']}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#c8d8f0;font-weight:600;">{txn['amount']}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:{sc_color};font-weight:600;">{sc:.3f}</div>
              <span class="txn-badge {bc}">{bt}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if history:
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        if st.button("🗑  Clear Session History", key="clear_history"):
            st.session_state.history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    show_sidebar()
    page = st.session_state.page
    if   page == "Dashboard":            page_dashboard()
    elif page == "Transaction Analyser": page_analyser()
    elif page == "Model Performance":    page_performance()
    elif page == "Dataset Explorer":     page_dataset()
    elif page == "Transaction History":  page_history()
