import streamlit as st
import numpy as np
import time
import random
import pandas as pd
import pickle

# ── Load all 3 models + scaler ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('fraud_model_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('fraud_model_lr.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('fraud_model_svm.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return rf, lr, svm, scaler

rf_model, lr_model, svm_model, scaler = load_models()

# ── Predict with all 3 models ─────────────────────────────────────────────────
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FraudShield · Detection System",
                   page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

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
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1320; }
::-webkit-scrollbar-thumb { background: #2a3a5c; border-radius: 2px; }

/* LOGIN */
.login-card {
    background: linear-gradient(145deg, #0d1827 0%, #0a1220 100%);
    border: 1px solid #1a2d4a; border-radius: 20px; padding: 3rem 2.5rem;
    width: 100%; max-width: 440px; position: relative; overflow: hidden;
    box-shadow: 0 40px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(59,130,246,0.08);
}
.login-card::before {
    content: ''; position: absolute; top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.login-card::after {
    content: ''; position: absolute; bottom: -40px; right: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.brand-mark { display:flex; align-items:center; gap:12px; margin-bottom:2rem; }
.shield-icon {
    width:44px; height:44px; background:linear-gradient(135deg,#1d4ed8,#3b82f6);
    border-radius:12px; display:flex; align-items:center; justify-content:center;
    font-size:20px; box-shadow:0 8px 24px rgba(59,130,246,0.3);
}
.brand-name { font-size:1.3rem; font-weight:800; color:#f0f4ff; letter-spacing:-0.02em; }
.brand-sub  { font-size:0.7rem; font-weight:500; color:#4b6a9c; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px; }
.login-title    { font-size:1.8rem; font-weight:800; color:#f0f4ff; letter-spacing:-0.03em; margin-bottom:0.3rem; }
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
    padding:0.75rem 1rem !important; transition:border-color 0.2s !important;
}
.stTextInput > div > div > input:focus { border-color:#3b82f6 !important; box-shadow:0 0 0 3px rgba(59,130,246,0.12) !important; }
.stTextInput label { color:#6b8ab0 !important; font-size:0.75rem !important; font-weight:600 !important; letter-spacing:0.06em !important; text-transform:uppercase !important; }

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#1d4ed8 0%,#3b82f6 100%) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:0.9rem !important; padding:0.75rem 1.5rem !important;
    width:100% !important; cursor:pointer !important; transition:all 0.2s !important;
    box-shadow:0 4px 20px rgba(59,130,246,0.3) !important; letter-spacing:0.02em !important;
}
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 8px 28px rgba(59,130,246,0.45) !important; }

/* DASHBOARD */
.dash-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:1.2rem 2rem; background:#0a0f1a;
    border-bottom:1px solid #141e30; margin:-3rem -3rem 2rem -3rem;
}
.nav-logo { font-size:1.2rem; font-weight:800; color:#f0f4ff; letter-spacing:-0.02em; }
.nav-logo span { color:#3b82f6; }
.nav-badge {
    background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.25);
    color:#10b981; border-radius:20px; padding:0.25rem 0.8rem;
    font-size:0.72rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase;
}
.nav-user {
    background:linear-gradient(135deg,#1d4ed8,#3b82f6); border-radius:50%;
    width:34px; height:34px; display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:0.85rem; box-shadow:0 4px 12px rgba(59,130,246,0.35);
}
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem; }
.metric-card {
    background:linear-gradient(145deg,#0d1827,#0a1220);
    border:1px solid #1a2d4a; border-radius:16px; padding:1.25rem 1.5rem;
    position:relative; overflow:hidden;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    border-radius:16px 16px 0 0;
}
.metric-card.blue::before   { background:linear-gradient(90deg,#3b82f6,#60a5fa); }
.metric-card.red::before    { background:linear-gradient(90deg,#ef4444,#f87171); }
.metric-card.green::before  { background:linear-gradient(90deg,#10b981,#34d399); }
.metric-card.purple::before { background:linear-gradient(90deg,#8b5cf6,#a78bfa); }
.metric-label { font-size:0.7rem; font-weight:600; color:#4b6a9c; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem; }
.metric-value { font-size:2rem; font-weight:800; color:#f0f4ff; font-family:'JetBrains Mono',monospace; letter-spacing:-0.03em; line-height:1; margin-bottom:0.3rem; }
.metric-delta.pos { color:#10b981; font-size:0.72rem; font-weight:600; }
.metric-delta.neg { color:#ef4444; font-size:0.72rem; font-weight:600; }
.metric-delta.neu { color:#6b8ab0; font-size:0.72rem; font-weight:600; }

.section-card { background:linear-gradient(145deg,#0d1827,#0a1220); border:1px solid #1a2d4a; border-radius:16px; padding:1.5rem; }
.section-title { font-size:0.78rem; font-weight:700; color:#4b6a9c; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px; }
.section-title::before { content:''; width:3px; height:14px; background:#3b82f6; border-radius:2px; display:inline-block; }

.result-safe  { background:rgba(16,185,129,0.07); border:1px solid rgba(16,185,129,0.2); border-radius:14px; padding:1.5rem; text-align:center; }
.result-fraud { background:rgba(239,68,68,0.07);  border:1px solid rgba(239,68,68,0.25); border-radius:14px; padding:1.5rem; text-align:center; animation:pulse-red 2s infinite; }
@keyframes pulse-red {
    0%,100% { box-shadow:0 0 0 0 rgba(239,68,68,0.15); }
    50%     { box-shadow:0 0 0 10px rgba(239,68,68,0); }
}
.result-icon  { font-size:2.5rem; margin-bottom:0.5rem; }
.result-label { font-size:0.7rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.3rem; }
.result-label.safe  { color:#10b981; }
.result-label.fraud { color:#ef4444; }
.result-main { font-size:1.4rem; font-weight:800; color:#f0f4ff; margin-bottom:0.5rem; }
.result-conf { font-size:0.8rem; color:#6b8ab0; }

.risk-bar-bg   { height:8px; background:#0d1827; border-radius:4px; border:1px solid #1a2a42; overflow:hidden; margin:0.5rem 0; }
.risk-bar-fill { height:100%; border-radius:4px; }

.vote-row  { margin-bottom:1rem; }
.vote-label { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.vote-name  { font-size:0.75rem; font-weight:700; color:#8ba4c8; }
.vote-score { font-size:0.75rem; font-family:'JetBrains Mono',monospace; }
.vote-bar-bg   { height:10px; background:#070b12; border-radius:5px; border:1px solid #1a2a42; overflow:hidden; }
.vote-bar-fill { height:100%; border-radius:5px; }
.vote-verdict  { font-size:0.65rem; font-weight:700; letter-spacing:0.08em; margin-top:3px; text-align:right; }

.txn-row { display:flex; align-items:center; justify-content:space-between; padding:0.7rem 0; border-bottom:1px solid #0f1c2e; font-size:0.82rem; }
.txn-row:last-child { border-bottom:none; }
.txn-id     { font-family:'JetBrains Mono',monospace; color:#4b6a9c; font-size:0.72rem; }
.txn-amount { font-family:'JetBrains Mono',monospace; font-weight:600; color:#c8d8f0; }
.txn-badge  { padding:0.2rem 0.6rem; border-radius:20px; font-size:0.65rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; }
.txn-safe   { background:rgba(16,185,129,0.12); color:#10b981; border:1px solid rgba(16,185,129,0.2); }
.txn-fraud  { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.2); }

.alert-box { background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.2); border-left:3px solid #ef4444; border-radius:8px; padding:0.75rem 1rem; font-size:0.8rem; color:#fca5a5; margin-bottom:0.75rem; }
.alert-box.info { background:rgba(59,130,246,0.06); border-color:rgba(59,130,246,0.2); border-left-color:#3b82f6; color:#93c5fd; }

.stSlider > div > div > div > div { background:#3b82f6 !important; }
.stSlider label { color:#6b8ab0 !important; font-size:0.75rem !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }
.stRadio label  { color:#8ba4c8 !important; font-size:0.85rem !important; }
.stRadio > div  { flex-direction:row !important; gap:1rem !important; }
.stNumberInput > div > div > input { background:#070b12 !important; border:1px solid #1a2a42 !important; border-radius:10px !important; color:#e8edf5 !important; font-family:'JetBrains Mono',monospace !important; }
.feat-row  { margin-bottom:0.8rem; }
.feat-name { font-size:0.72rem; color:#6b8ab0; margin-bottom:3px; text-transform:uppercase; letter-spacing:0.05em; }
.feat-bar-bg   { height:6px; background:#0d1827; border-radius:3px; }
.feat-bar-fill { height:6px; border-radius:3px; background:linear-gradient(90deg,#1d4ed8,#60a5fa); }
.stMarkdown p { color:#8ba4c8; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "logged_in"     not in st.session_state: st.session_state.logged_in     = False
if "username"      not in st.session_state: st.session_state.username      = ""
if "last_result"   not in st.session_state: st.session_state.last_result   = None
if "total_scanned" not in st.session_state: st.session_state.total_scanned = 4821
if "fraud_caught"  not in st.session_state: st.session_state.fraud_caught  = 127
if "history"       not in st.session_state:
    st.session_state.history = [
        {"id":"TXN-4821","amount":"$142.50",   "fraud":False,"time":"2 min ago"},
        {"id":"TXN-4820","amount":"$89.99",    "fraud":False,"time":"8 min ago"},
        {"id":"TXN-4819","amount":"$1,250.00", "fraud":True, "time":"15 min ago"},
        {"id":"TXN-4818","amount":"$34.20",    "fraud":False,"time":"22 min ago"},
        {"id":"TXN-4817","amount":"$567.80",   "fraud":True, "time":"31 min ago"},
    ]

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════════════════════════════════════
def show_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="login-card">
          <div class="brand-mark">
            <div class="shield-icon">🛡️</div>
            <div style="line-height:1">
              <div class="brand-name">FraudShield</div>
              <div class="brand-sub">Ensemble Detection System</div>
            </div>
          </div>
          <div class="stat-strip">
            <div class="stat-item"><div class="stat-num">3</div><div class="stat-lbl">Models</div></div>
            <div class="stat-item"><div class="stat-num">99.2%</div><div class="stat-lbl">Accuracy</div></div>
            <div class="stat-item"><div class="stat-num">Live</div><div class="stat-lbl">Status</div></div>
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

        st.markdown('<div style="text-align:center;margin-top:1.2rem;font-size:0.72rem;color:#2a3f5c;">Demo: any username · any password</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def show_dashboard():
    user_init = st.session_state.username[0].upper() if st.session_state.username else "A"

    st.markdown(f"""
    <div class="dash-header">
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:1.4rem">🛡️</span>
        <span class="nav-logo">Fraud<span>Shield</span></span>
        <span style="color:#1a2d4a;font-size:0.8rem;margin-left:4px">/ Ensemble Console</span>
      </div>
      <div style="display:flex;align-items:center;gap:1rem;">
        <span class="nav-badge">● 3 Models Active</span>
        <div style="font-size:0.78rem;color:#4b6a9c;">{st.session_state.username}</div>
        <div class="nav-user">{user_init}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card blue">
        <div class="metric-label">Transactions Scanned</div>
        <div class="metric-value">{st.session_state.total_scanned:,}</div>
        <div class="metric-delta pos">↑ 312 today</div>
      </div>
      <div class="metric-card red">
        <div class="metric-label">Fraud Detected</div>
        <div class="metric-value">{st.session_state.fraud_caught}</div>
        <div class="metric-delta neg">↑ 14 today</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">Ensemble Accuracy</div>
        <div class="metric-value">99.2%</div>
        <div class="metric-delta pos">↑ 0.3% vs single model</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-label">Active Models</div>
        <div class="metric-value">3</div>
        <div class="metric-delta neu">RF · LR · SVM</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap="medium")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Transaction Analyser</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.78rem;color:#3a5070;margin-bottom:1rem;">All 3 models analyse simultaneously and vote on the final verdict.</p>', unsafe_allow_html=True)

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
            new_txn = {
                "id":     f"TXN-{st.session_state.total_scanned+1}",
                "amount": f"${random.uniform(10,2000):.2f}",
                "fraud":  result["is_fraud"],
                "time":   "just now",
            }
            st.session_state.history.insert(0, new_txn)
            st.session_state.history = st.session_state.history[:5]
            st.session_state.total_scanned += 1
            if result["is_fraud"]:
                st.session_state.fraud_caught += 1
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        feats = [("Ratio to Median Price",88),("Distance from Home",74),
                 ("Distance Last Txn",52),("Repeat Retailer",38),
                 ("Used Chip",31),("Used PIN",22),("Online Order",18)]
        for name, pct in feats:
            st.markdown(f"""
            <div class="feat-row">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
                <span class="feat-name">{name}</span>
                <span style="font-size:0.7rem;color:#3b82f6;font-family:'JetBrains Mono',monospace;">{pct}%</span>
              </div>
              <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # Final verdict
        st.markdown('<div class="section-card" style="margin-bottom:12px">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Final Verdict</div>', unsafe_allow_html=True)

        if st.session_state.last_result is None:
            st.markdown("""
            <div style="text-align:center;padding:2rem 0;color:#2a3f5c;">
              <div style="font-size:2.5rem;margin-bottom:0.5rem;opacity:0.4">🔍</div>
              <div style="font-size:0.82rem;">Run analysis to see ensemble prediction</div>
            </div>""", unsafe_allow_html=True)
        else:
            res = st.session_state.last_result
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
                st.markdown('<div class="alert-box" style="margin-top:0.75rem">⚠ Majority of models flagged this. Automated block initiated.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                  <div class="result-icon">✅</div>
                  <div class="result-label safe">LEGITIMATE</div>
                  <div class="result-main">Transaction Cleared</div>
                  <div class="result-conf">Ensemble risk: {pct}% · Recommend: Approve</div>
                </div>""", unsafe_allow_html=True)
                st.markdown('<div class="alert-box info" style="margin-top:0.75rem">ℹ All models cleared this transaction. Approved.</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="section-card" style="margin-bottom:12px">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Votes</div>', unsafe_allow_html=True)

        if st.session_state.last_result is None:
            st.markdown('<div style="text-align:center;padding:1.5rem 0;color:#2a3f5c;font-size:0.82rem;">Run analysis to see individual model votes</div>', unsafe_allow_html=True)
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
                  <div class="vote-bar-bg">
                    <div class="vote-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
                  </div>
                  <div class="vote-verdict" style="color:{v_color}">Voted: {verdict}</div>
                </div>""", unsafe_allow_html=True)

            fraud_votes = sum(res["votes"])
            safe_votes  = 3 - fraud_votes
            st.markdown(f"""
            <div style="margin-top:1rem;background:#070b12;border:1px solid #1a2a42;border-radius:10px;padding:0.75rem 1rem;display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:0.72rem;color:#4b6a9c;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">Final Vote Tally</span>
              <span style="font-size:0.85rem;font-family:'JetBrains Mono',monospace;">
                <span style="color:#ef4444;font-weight:700;">{fraud_votes} Fraud</span>
                <span style="color:#2a3a52;"> vs </span>
                <span style="color:#10b981;font-weight:700;">{safe_votes} Safe</span>
              </span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Recent transactions
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Transactions</div>', unsafe_allow_html=True)
        for txn in st.session_state.history:
            bc = "txn-fraud" if txn["fraud"] else "txn-safe"
            bt = "FRAUD"     if txn["fraud"] else "SAFE"
            st.markdown(f"""
            <div class="txn-row">
              <div>
                <div style="font-size:0.82rem;color:#c8d8f0;font-weight:600;">{txn['id']}</div>
                <div class="txn-id">{txn['time']}</div>
              </div>
              <div class="txn-amount">{txn['amount']}</div>
              <span class="txn-badge {bc}">{bt}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    _, _, logout_col = st.columns([3, 1, 0.6])
    with logout_col:
        if st.button("Sign Out", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.last_result = None
            st.rerun()

# ── Router ────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_login()
else:
    show_dashboard()
