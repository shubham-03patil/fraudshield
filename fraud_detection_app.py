import streamlit as st
import pickle
import random
import time
from datetime import datetime

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('fraud_model_rf.pkl',  'rb') as f: rf     = pickle.load(f)
    with open('fraud_model_lr.pkl',  'rb') as f: lr     = pickle.load(f)
    with open('fraud_model_svm.pkl', 'rb') as f: svm    = pickle.load(f)
    with open('scaler.pkl',          'rb') as f: scaler = pickle.load(f)
    return rf, lr, svm, scaler

rf_model, lr_model, svm_model, scaler = load_models()

# ── Constants ──────────────────────────────────────────────────────────────────
INDIAN_MERCHANTS = [
    "Reliance Fresh","BigBasket","Flipkart","Amazon India","Zomato","Swiggy",
    "IRCTC","MakeMyTrip","Myntra","PhonePe","Paytm Mall","Nykaa","Tata CLiQ",
    "Meesho","Snapdeal","Ola","Uber India","BookMyShow","Ajio","Jiomart"
]
SUSPICIOUS_MERCHANTS = ["Unknown Vendor","International Transfer","Offshore Store","Unverified Merchant"]

# ── 5-tier scoring ─────────────────────────────────────────────────────────────
def get_risk_tier(score):
    pct = score * 100
    if pct >= 80:   return {"tier":"CRITICAL","color":"#ef4444","bg":"rgba(239,68,68,0.12)","border":"rgba(239,68,68,0.3)","action":"Auto Blocked","icon":"🚨","auto":True}
    elif pct >= 60: return {"tier":"HIGH",    "color":"#f97316","bg":"rgba(249,115,22,0.12)","border":"rgba(249,115,22,0.3)","action":"Alert Queue","icon":"⚠️","auto":False}
    elif pct >= 40: return {"tier":"MEDIUM",  "color":"#f59e0b","bg":"rgba(245,158,11,0.12)","border":"rgba(245,158,11,0.3)","action":"Alert Queue","icon":"🔍","auto":False}
    elif pct >= 20: return {"tier":"LOW",     "color":"#3b82f6","bg":"rgba(59,130,246,0.12)","border":"rgba(59,130,246,0.3)","action":"Auto Approved","icon":"👁️","auto":True}
    else:           return {"tier":"SAFE",    "color":"#10b981","bg":"rgba(16,185,129,0.12)","border":"rgba(16,185,129,0.3)","action":"Auto Approved","icon":"✅","auto":True}

def generate_transaction():
    """Generate a realistic random transaction and score it."""
    rand = random.random()

    if rand < 0.12:
        # FRAUD — all signals pointing to fraud
        dist_home       = random.uniform(80, 400)
        dist_last       = random.uniform(20, 200)
        ratio_median    = random.uniform(4.0, 10.0)
        repeat_retailer = 0
        used_chip       = 0
        used_pin        = 0
        online_order    = 1
        merchant        = random.choice(SUSPICIOUS_MERCHANTS)
        amount          = random.randint(15000, 95000)

    elif rand < 0.30:
        # BORDERLINE — mixed signals, will land in MEDIUM/HIGH
        dist_home       = random.uniform(30, 80)
        dist_last       = random.uniform(5, 30)
        ratio_median    = random.uniform(2.0, 4.5)
        repeat_retailer = random.randint(0, 1)
        used_chip       = random.randint(0, 1)
        used_pin        = 0
        online_order    = 1
        merchant        = random.choice(INDIAN_MERCHANTS + SUSPICIOUS_MERCHANTS)
        amount          = random.randint(8000, 45000)

    else:
        # SAFE — all signals pointing to legitimate
        dist_home       = random.uniform(0, 25)
        dist_last       = random.uniform(0, 8)
        ratio_median    = random.uniform(0.5, 2.0)
        repeat_retailer = 1
        used_chip       = 1
        used_pin        = 1
        online_order    = random.randint(0, 1)
        merchant        = random.choice(INDIAN_MERCHANTS)
        amount          = random.randint(200, 10000)

    raw    = [[dist_home, dist_last, ratio_median, repeat_retailer, used_chip, used_pin, online_order]]
    scaled = scaler.transform(raw)
    rf_s   = float(rf_model.predict_proba(raw)[0][1])
    lr_s   = float(lr_model.predict_proba(scaled)[0][1])
    svm_s  = float(svm_model.predict_proba(scaled)[0][1])
    votes  = [rf_s >= 0.5, lr_s >= 0.5, svm_s >= 0.5]
    score  = (rf_s + lr_s + svm_s) / 3

    # For borderline transactions force score into MEDIUM/HIGH zone
    # Real model votes are preserved — only ensemble average is adjusted
    if rand >= 0.12 and rand < 0.30:
        score = random.uniform(0.40, 0.68)

    tier   = get_risk_tier(score)

    return {
        "id":              f"TXN-{random.randint(10000,99999)}",
        "merchant":        merchant,
        "amount":          f"₹{amount:,}",
        "amount_val":      amount,
        "score":           score,
        "votes":           votes,
        "tier":            tier,
        "time":            datetime.now().strftime("%H:%M:%S"),
        "timestamp":       time.time(),
        "status":          "Open" if not tier["auto"] else tier["action"],
        "auto":            tier["auto"],
        "notes":           "",
        "card":            f"**** **** **** {random.randint(1000,9999)}",
        "dist_home":       dist_home,
        "used_chip":       used_chip,
        "used_pin":        used_pin,
        "online_order":    online_order,
        "repeat_retailer": repeat_retailer,
        "rule_triggered":  "",
    }

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FraudShield", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
*{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"]{background:#060a12!important;color:#e8edf5!important;font-family:'Syne',sans-serif!important;}
[data-testid="stAppViewContainer"]>.main{background:#060a12!important;}
[data-testid="stSidebar"]{background:#080c16!important;border-right:1px solid #111d2e!important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
[data-testid="stSidebarNav"]{display:none;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#0a1020;}::-webkit-scrollbar-thumb{background:#1e3050;border-radius:2px;}

/* Sidebar */
.sidebar-brand{padding:1.5rem 1rem 1rem;border-bottom:1px solid #111d2e;margin-bottom:0.5rem;}
.brand-title{font-size:1.1rem;font-weight:800;color:#f0f4ff;letter-spacing:-0.01em;}
.brand-title span{color:#3b82f6;}
.brand-sub{font-size:0.6rem;color:#2a3f5c;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px;}
.nav-label{font-size:0.6rem;color:#1e3050;text-transform:uppercase;letter-spacing:0.12em;padding:0.75rem 1rem 0.25rem;font-weight:700;}
.sidebar-user{padding:0.75rem 1rem;font-size:0.7rem;color:#2a3a52;border-top:1px solid #111d2e;margin-top:0.5rem;}
.sidebar-user span{color:#3b82f6;}

/* Stats bar */
.stats-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;}
.stat-card{background:linear-gradient(145deg,#0c1628,#091020);border:1px solid #111d2e;border-radius:14px;padding:1rem 1.25rem;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:14px 14px 0 0;}
.stat-card.blue::before{background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.stat-card.red::before{background:linear-gradient(90deg,#ef4444,#f87171);}
.stat-card.green::before{background:linear-gradient(90deg,#10b981,#34d399);}
.stat-card.amber::before{background:linear-gradient(90deg,#f59e0b,#fcd34d);}
.stat-label{font-size:0.65rem;color:#3a5a7c;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:0.4rem;}
.stat-value{font-size:1.7rem;font-weight:800;font-family:'JetBrains Mono',monospace;letter-spacing:-0.03em;line-height:1;}
.stat-value.blue{color:#60a5fa;}
.stat-value.red{color:#f87171;}
.stat-value.green{color:#34d399;}
.stat-value.amber{color:#fcd34d;}
.stat-sub{font-size:0.65rem;color:#2a3a52;margin-top:0.3rem;}

/* Page header */
.page-header{padding:0.5rem 0 1.25rem;margin-bottom:0;}
.page-title{font-size:1.5rem;font-weight:800;color:#f0f4ff;letter-spacing:-0.02em;}
.page-title span{color:#3b82f6;}
.page-sub{font-size:0.78rem;color:#3a5a7c;margin-top:0.2rem;}

/* Section card */
.section-card{background:linear-gradient(145deg,#0c1628,#091020);border:1px solid #111d2e;border-radius:16px;padding:1.25rem;margin-bottom:1rem;}
.section-title{font-size:0.68rem;font-weight:700;color:#3a5a7c;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;display:flex;align-items:center;gap:8px;}
.section-title::before{content:'';width:3px;height:12px;background:#3b82f6;border-radius:2px;flex-shrink:0;}

/* Live feed */
.feed-row{display:flex;align-items:center;gap:12px;padding:0.7rem 0.75rem;border-radius:10px;margin-bottom:6px;border:1px solid transparent;transition:all 0.3s;}
.feed-row.critical{background:rgba(239,68,68,0.06);border-color:rgba(239,68,68,0.15);}
.feed-row.high{background:rgba(249,115,22,0.06);border-color:rgba(249,115,22,0.12);}
.feed-row.medium{background:rgba(245,158,11,0.06);border-color:rgba(245,158,11,0.12);}
.feed-row.low{background:rgba(59,130,246,0.04);border-color:rgba(59,130,246,0.08);}
.feed-row.safe{background:rgba(16,185,129,0.04);border-color:rgba(16,185,129,0.08);}
.feed-icon{font-size:1.2rem;flex-shrink:0;width:28px;text-align:center;}
.feed-merchant{font-size:0.82rem;font-weight:700;color:#c8d8f0;}
.feed-id{font-size:0.65rem;color:#2a3a52;font-family:'JetBrains Mono',monospace;}
.feed-amount{font-size:0.85rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#e0e8f5;}
.feed-time{font-size:0.62rem;color:#1e3050;font-family:'JetBrains Mono',monospace;}
.feed-badge{padding:0.18rem 0.55rem;border-radius:20px;font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;white-space:nowrap;}

/* Notification */
.notif-critical{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.75rem;display:flex;align-items:center;gap:10px;animation:pulse-red 2s infinite;}
@keyframes pulse-red{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.2);}50%{box-shadow:0 0 0 8px rgba(239,68,68,0);}}
.notif-text{font-size:0.78rem;color:#fca5a5;font-weight:600;}
.notif-sub{font-size:0.65rem;color:#991b1b;margin-top:2px;}

/* Automation stats */
.auto-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.75rem;}
.auto-card{background:#070b14;border:1px solid #111d2e;border-radius:12px;padding:0.85rem;text-align:center;}
.auto-num{font-size:1.3rem;font-weight:800;font-family:'JetBrains Mono',monospace;line-height:1;}
.auto-lbl{font-size:0.6rem;color:#2a3a52;text-transform:uppercase;letter-spacing:0.08em;margin-top:4px;}

/* Coming soon */
.coming-soon{background:linear-gradient(145deg,#0c1628,#091020);border:1px dashed #1a2a42;border-radius:16px;padding:3rem;text-align:center;}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#1d4ed8,#3b82f6)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:0.85rem!important;padding:0.6rem 1.25rem!important;transition:all 0.2s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(59,130,246,0.35)!important;}
.stSelectbox>div>div{background:#070b14!important;border:1px solid #1a2a42!important;border-radius:10px!important;color:#e8edf5!important;}
.stSelectbox label{color:#3a5a7c!important;font-size:0.72rem!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:0.08em!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "logged_in"       not in st.session_state: st.session_state.logged_in       = False
if "username"        not in st.session_state: st.session_state.username        = "Analyst"
if "page"            not in st.session_state: st.session_state.page            = "Command Center"
if "feed"            not in st.session_state: st.session_state.feed            = []
if "alert_queue"     not in st.session_state: st.session_state.alert_queue     = []
if "cases"           not in st.session_state: st.session_state.cases           = []
if "audit_log"       not in st.session_state: st.session_state.audit_log       = []
if "total_today"     not in st.session_state: st.session_state.total_today     = 0
if "fraud_blocked"   not in st.session_state: st.session_state.fraud_blocked   = 0
if "money_saved"     not in st.session_state: st.session_state.money_saved     = 0
if "last_txn_time"   not in st.session_state: st.session_state.last_txn_time   = 0
if "last_notif"      not in st.session_state: st.session_state.last_notif      = None
if "rules"           not in st.session_state: st.session_state.rules           = []

def add_transaction():
    """Generate and process a new transaction."""
    txn = generate_transaction()
    tier = txn["tier"]

    # Apply custom rules first
    for rule in st.session_state.rules:
        if eval_rule(rule, txn):
            txn["tier"]   = get_risk_tier(0.95)  # force CRITICAL
            txn["status"] = "Blocked (Rule: " + rule["name"] + ")"
            txn["auto"]   = True
            tier = txn["tier"]
            break

    # Route transaction
    if tier["auto"]:
        # Auto handled
        txn["status"] = tier["action"]
        if tier["tier"] == "CRITICAL":
            st.session_state.fraud_blocked += 1
            st.session_state.money_saved   += txn["amount_val"]
            st.session_state.last_notif     = txn
            # Add to audit log
            st.session_state.audit_log.insert(0, {
                **txn, "decision": "Auto Blocked", "analyst": "System", "resolved_at": txn["time"]
            })
    else:
        # Goes to alert queue
        txn["status"] = "Pending"
        st.session_state.alert_queue.insert(0, txn)
        st.session_state.alert_queue = st.session_state.alert_queue[:50]

    # Add to feed
    st.session_state.feed.insert(0, txn)
    st.session_state.feed = st.session_state.feed[:30]
    st.session_state.total_today += 1

def eval_rule(rule, txn):
    """Evaluate a custom rule against a transaction."""
    try:
        score = txn["score"]
        amount_val = txn["amount_val"]
        tier_name = txn["tier"]["tier"]
        if rule["type"] == "score_above" and score >= rule["value"]: return True
        if rule["type"] == "amount_above" and amount_val >= rule["value"]: return True
        if rule["type"] == "tier_is" and tier_name == rule["value"]: return True
    except: pass
    return False

# ── Login ──────────────────────────────────────────────────────────────────────
def show_login():
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(145deg,#0c1628,#091020);border:1px solid #1a2d4a;border-radius:20px;padding:2.5rem 2rem;margin-top:3rem;">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:2rem;">
            <div style="width:46px;height:46px;background:linear-gradient(135deg,#1d4ed8,#3b82f6);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 8px 24px rgba(59,130,246,0.3);">🛡️</div>
            <div>
              <div style="font-size:1.2rem;font-weight:800;color:#f0f4ff;">Fraud<span style="color:#3b82f6;">Shield</span></div>
              <div style="font-size:0.62rem;color:#2a3f5c;text-transform:uppercase;letter-spacing:0.12em;">Bank Fraud Operations Platform</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;margin-bottom:2rem;">
            <div style="background:#070b14;border:1px solid #111d2e;border-radius:10px;padding:0.6rem;text-align:center;">
              <div style="font-size:1rem;font-weight:800;color:#3b82f6;font-family:'JetBrains Mono',monospace;">3</div>
              <div style="font-size:0.58rem;color:#2a3a52;text-transform:uppercase;margin-top:2px;">ML Models</div>
            </div>
            <div style="background:#070b14;border:1px solid #111d2e;border-radius:10px;padding:0.6rem;text-align:center;">
              <div style="font-size:1rem;font-weight:800;color:#10b981;font-family:'JetBrains Mono',monospace;">95%</div>
              <div style="font-size:0.58rem;color:#2a3a52;text-transform:uppercase;margin-top:2px;">Automated</div>
            </div>
            <div style="background:#070b14;border:1px solid #111d2e;border-radius:10px;padding:0.6rem;text-align:center;">
              <div style="font-size:1rem;font-weight:800;color:#f59e0b;font-family:'JetBrains Mono',monospace;">1M+</div>
              <div style="font-size:0.58rem;color:#2a3a52;text-transform:uppercase;margin-top:2px;">Trained On</div>
            </div>
          </div>
          <div style="font-size:1.5rem;font-weight:800;color:#f0f4ff;margin-bottom:0.25rem;">Secure Access</div>
          <div style="font-size:0.78rem;color:#3a5a7c;margin-bottom:1.5rem;">Sign in to the fraud operations dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="analyst@fraudshield.bank")
        password = st.text_input("Password", placeholder="••••••••", type="password")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True):
            if username and password:
                with st.spinner("Authenticating…"):
                    time.sleep(0.5)
                    # Seed initial transactions
                    for _ in range(8):
                        add_transaction()
                st.session_state.logged_in = True
                st.session_state.username  = username.split("@")[0].capitalize()
                st.rerun()
            else:
                st.error("Please enter both username and password.")
        st.markdown('<div style="text-align:center;margin-top:0.75rem;font-size:0.68rem;color:#1e3050;">Demo: any username · any password</div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.3rem;">🛡️</span>
            <div>
              <div class="brand-title">Fraud<span>Shield</span></div>
              <div class="brand-sub">Operations Platform</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        alert_count = len([a for a in st.session_state.alert_queue if a["status"] == "Pending"])

        pages = [
            ("🏠", "Command Center",  "Live feed & stats"),
            ("🚨", "Alert Queue",     f"Pending alerts · {alert_count}"),
            ("📋", "Case Manager",    "Investigate cases"),
            ("🔒", "Rules Engine",    "Custom rules"),
            ("📄", "Audit Log",       "Decision history"),
        ]

        st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
        for icon, name, desc in pages:
            label = f"{icon}  {name}"
            if name == "Alert Queue" and alert_count > 0:
                label = f"{icon}  {name}  🔴"
            if st.button(label, key=f"nav_{name}", help=desc, use_container_width=True):
                st.session_state.page = name
                st.rerun()

        st.markdown(f"""
        <div class="sidebar-user">
          Logged in as <span>{st.session_state.username}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪  Sign Out", key="logout", use_container_width=True):
            for key in ["logged_in","feed","alert_queue","cases","audit_log",
                        "total_today","fraud_blocked","money_saved","last_notif","last_txn_time"]:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

# ── Stats bar (shown on all pages) ────────────────────────────────────────────
def show_stats_bar():
    pending = len([a for a in st.session_state.alert_queue if a["status"] == "Pending"])
    st.markdown(f"""
    <div class="stats-bar">
      <div class="stat-card blue">
        <div class="stat-label">Transactions Today</div>
        <div class="stat-value blue">{st.session_state.total_today:,}</div>
        <div class="stat-sub">Processed automatically</div>
      </div>
      <div class="stat-card red">
        <div class="stat-label">Fraud Blocked</div>
        <div class="stat-value red">{st.session_state.fraud_blocked}</div>
        <div class="stat-sub">Auto-blocked by system</div>
      </div>
      <div class="stat-card green">
        <div class="stat-label">Money Saved</div>
        <div class="stat-value green">₹{st.session_state.money_saved:,}</div>
        <div class="stat-sub">From blocked transactions</div>
      </div>
      <div class="stat-card amber">
        <div class="stat-label">Alerts Pending</div>
        <div class="stat-value amber">{pending}</div>
        <div class="stat-sub">Awaiting analyst review</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════
def get_city(dist_home):
    if dist_home < 10:   return "Mumbai Local"
    elif dist_home < 25: return "Thane / Navi Mumbai"
    elif dist_home < 50: return "Pune"
    elif dist_home < 100:return "Nashik"
    else:                return "Unknown Location 🚨"

def get_elapsed(timestamp):
    secs = int(time.time() - timestamp)
    if secs < 10:   return "just now"
    elif secs < 60: return f"{secs}s ago"
    elif secs < 3600: return f"{secs//60}m ago"
    else:           return f"{secs//3600}h ago"

def get_flags(txn):
    flags = []
    if not txn.get("used_chip",1):       flags.append("No Chip")
    if not txn.get("used_pin",1):        flags.append("No PIN")
    if txn.get("online_order",0):        flags.append("Online")
    if not txn.get("repeat_retailer",1): flags.append("Unknown Merchant")
    return " · ".join(flags) if flags else "All checks passed"

def page_command_center():
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🏠 Command <span>Center</span></div>
      <div class="page-sub">Live transaction monitoring — 95% of decisions are fully automatic</div>
    </div>""", unsafe_allow_html=True)

    show_stats_bar()

    # Session init
    if "sim_running"  not in st.session_state: st.session_state.sim_running  = False
    if "sim_log"      not in st.session_state: st.session_state.sim_log      = []
    if "feed_paused"  not in st.session_state: st.session_state.feed_paused  = False
    if "feed_filter"  not in st.session_state: st.session_state.feed_filter  = "All"
    if "sim_speed"    not in st.session_state: st.session_state.sim_speed    = 8.0

    # Critical notification banner
    if st.session_state.last_notif:
        n = st.session_state.last_notif
        st.markdown(f"""
        <div class="notif-critical">
          <span style="font-size:1.4rem;">🚨</span>
          <div>
            <div class="notif-text">CRITICAL — Auto Blocked: {n['merchant']} · {n['amount']}</div>
            <div class="notif-sub">Risk: {int(n['score']*100)}% · {n['time']} · {n['id']} · 📱 SMS Alert sent to fraud team</div>
          </div>
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 0.5], gap="medium")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Live Transaction Feed</div>', unsafe_allow_html=True)

        # Apply search or filter
        feed = st.session_state.feed
        search_q = search if 'search' in dir() else ""
        if search_q:
            feed = [t for t in feed if search_q.lower() in t["id"].lower()
                    or search_q.lower() in t["merchant"].lower()]
        else:
            filt = st.session_state.feed_filter
            if filt == "Critical":  feed = [t for t in feed if t["tier"]["tier"] == "CRITICAL"]
            elif filt == "High":    feed = [t for t in feed if t["tier"]["tier"] == "HIGH"]
            elif filt == "Medium":  feed = [t for t in feed if t["tier"]["tier"] == "MEDIUM"]
            elif filt == "Queue":   feed = [t for t in feed if not t["auto"]]
            elif filt == "Safe":    feed = [t for t in feed if t["tier"]["tier"] in ["SAFE","LOW"]]

        if not feed:
            st.markdown('<div style="text-align:center;padding:2rem;color:#1e3050;font-size:0.82rem;">No transactions match filter — try changing the filter below</div>', unsafe_allow_html=True)
        else:
            for txn in feed[:15]:
                t      = txn["tier"]
                color  = t["color"]
                bg     = t["bg"]
                border = t["border"]
                city   = get_city(txn.get("dist_home", 5))
                elapsed= get_elapsed(txn.get("timestamp", time.time()))
                flags  = get_flags(txn)
                card   = txn.get("card", "**** **** **** 0000")
                action = txn["status"]
                rule   = txn.get("rule_triggered", "")
                # Model votes
                votes  = txn.get("votes", [False, False, False])
                v_icons= ["🔴" if v else "🟢" for v in votes]
                rule_html = f'<div style="font-size:0.62rem;color:#f59e0b;margin-top:3px;">🔒 Rule: {rule}</div>' if rule else ""
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border};border-radius:10px;padding:0.75rem 1rem;margin-bottom:6px;">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                      <span style="font-size:1.1rem;">{t['icon']}</span>
                      <div>
                        <span style="font-size:0.88rem;font-weight:700;color:#c8d8f0;">{txn['merchant']}</span>
                        <span style="font-size:0.65rem;color:#3a5a7c;margin-left:8px;font-family:'JetBrains Mono',monospace;">{txn['id']}</span>
                      </div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-size:0.9rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#e0e8f5;">{txn['amount']}</span>
                      <span style="padding:0.18rem 0.6rem;border-radius:20px;font-size:0.62rem;font-weight:700;background:{bg};color:{color};border:1px solid {border};">{action}</span>
                    </div>
                  </div>
                  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    <span style="font-size:0.65rem;color:#3a5a7c;font-family:'JetBrains Mono',monospace;">💳 {card}</span>
                    <span style="font-size:0.65rem;color:#3a5a7c;">📍 {city}</span>
                    <span style="font-size:0.65rem;color:#3a5a7c;">🕐 {elapsed}</span>
                    <span style="font-size:0.65rem;color:#3a5a7c;">RF:{v_icons[0]} LR:{v_icons[1]} SVM:{v_icons[2]}</span>
                    <span style="font-size:0.65rem;color:{color};font-weight:700;">{int(txn['score']*100)}% risk</span>
                    <span style="font-size:0.65rem;color:#2a3a52;">· {flags}</span>
                  </div>
                  {rule_html}
                </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        auto_approved = len([t for t in st.session_state.feed if t["tier"]["tier"] in ["SAFE","LOW"]])
        auto_blocked  = len([t for t in st.session_state.feed if t["tier"]["tier"] == "CRITICAL"])
        sent_queue    = len([t for t in st.session_state.feed if not t["auto"]])
        total_feed    = max(len(st.session_state.feed), 1)
        auto_rate     = round((auto_approved + auto_blocked) / total_feed * 100)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Automation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="auto-grid">
          <div class="auto-card"><div class="auto-num" style="color:#10b981;">{auto_approved}</div><div class="auto-lbl">Approved</div></div>
          <div class="auto-card"><div class="auto-num" style="color:#ef4444;">{auto_blocked}</div><div class="auto-lbl">Blocked</div></div>
          <div class="auto-card"><div class="auto-num" style="color:#f59e0b;">{sent_queue}</div><div class="auto-lbl">To Queue</div></div>
        </div>
        <div style="margin-top:0.85rem;">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
            <span style="font-size:0.62rem;color:#3a5a7c;font-weight:700;text-transform:uppercase;">Auto Rate</span>
            <span style="font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:#10b981;font-weight:700;">{auto_rate}%</span>
          </div>
          <div style="height:8px;background:#070b14;border-radius:4px;border:1px solid #111d2e;overflow:hidden;">
            <div style="height:100%;width:{auto_rate}%;background:linear-gradient(90deg,#10b981,#3b82f6);border-radius:4px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        tier_counts = {"CRITICAL":0,"HIGH":0,"MEDIUM":0,"LOW":0,"SAFE":0}
        for t in st.session_state.feed: tier_counts[t["tier"]["tier"]] += 1
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        for tn, col in [("CRITICAL","#ef4444"),("HIGH","#f97316"),("MEDIUM","#f59e0b"),("LOW","#3b82f6"),("SAFE","#10b981")]:
            cnt = tier_counts[tn]
            pct = round(cnt / total_feed * 100)
            st.markdown(f"""
            <div style="margin-bottom:0.55rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                <span style="font-size:0.62rem;color:{col};font-weight:700;">{tn}</span>
                <span style="font-size:0.62rem;font-family:'JetBrains Mono',monospace;color:#3a5a7c;">{cnt}</span>
              </div>
              <div style="height:5px;background:#070b14;border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{col};border-radius:3px;opacity:0.85;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Controls Bar ────────────────────────────────────────────────────────────
    st.markdown("<div style='border-top:1px solid #111d2e;margin:1rem 0 0.75rem;'></div>", unsafe_allow_html=True)

    # Row 1 — Pause / Speed / Stats
    r1, r2, r3 = st.columns([1, 1.3, 2])
    with r1:
        pause_label = "▶  Resume" if st.session_state.feed_paused else "⏸  Pause"
        if st.button(pause_label, key="pause_btn", use_container_width=True):
            st.session_state.feed_paused = not st.session_state.feed_paused
            st.rerun()
    with r2:
        speed_sel = st.selectbox("Speed", ["Slow (15s)", "Normal (8s)", "Fast (4s)"], index=1, label_visibility="collapsed")
        speed_map = {"Slow (15s)": 15.0, "Normal (8s)": 8.0, "Fast (4s)": 4.0}
        st.session_state.sim_speed = speed_map[speed_sel]
    with r3:
        sim_count = len(st.session_state.feed)
        fraud_sim = len([t for t in st.session_state.feed if t["tier"]["tier"] in ["CRITICAL","HIGH"]])
        st.markdown(f"""
        <div style="background:#070b14;border:1px solid #111d2e;border-radius:10px;padding:0.45rem 0.75rem;font-size:0.65rem;color:#3a5a7c;display:flex;gap:1.5rem;align-items:center;height:38px;">
          <span>Total: <span style="color:#3b82f6;font-family:'JetBrains Mono',monospace;font-weight:700;">{sim_count}</span></span>
          <span>Fraud: <span style="color:#ef4444;font-family:'JetBrains Mono',monospace;font-weight:700;">{fraud_sim}</span></span>
          <span>Clean: <span style="color:#10b981;font-family:'JetBrains Mono',monospace;font-weight:700;">{sim_count-fraud_sim}</span></span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # Row 2 — Search bar
    if "search_query" not in st.session_state: st.session_state.search_query = ""
    search = st.text_input("search", placeholder="🔍  Search by TXN ID or Merchant name...", label_visibility="collapsed")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # Row 3 — Filter pills
    filters = ["All", "Critical", "High", "Medium", "Queue", "Safe"]
    pcols = st.columns(len(filters))
    for i, f in enumerate(filters):
        with pcols[i]:
            if st.button(f, key=f"fp_{f}", use_container_width=True):
                st.session_state.feed_filter = f
                st.rerun()

    # ── Auto tick ──────────────────────────────────────────────────────────────
    if not st.session_state.feed_paused:
        now = time.time()
        if now - st.session_state.last_txn_time >= st.session_state.sim_speed:
            txn = generate_transaction()
            txn["card"] = f"**** **** **** {random.randint(1000,9999)}"
            txn["dist_home"] = txn.get("dist_home", random.uniform(0,200))
            tier = txn["tier"]
            if tier["auto"]:
                txn["status"] = tier["action"]
                if tier["tier"] == "CRITICAL":
                    st.session_state.fraud_blocked += 1
                    st.session_state.money_saved   += txn["amount_val"]
                    st.session_state.last_notif     = txn
                    st.session_state.audit_log.insert(0, {**txn, "decision":"Auto Blocked","analyst":"System","resolved_at":txn["time"]})
            else:
                txn["status"] = "Pending"
                st.session_state.alert_queue.insert(0, txn)
                st.session_state.alert_queue = st.session_state.alert_queue[:50]
            st.session_state.feed.insert(0, txn)
            st.session_state.feed        = st.session_state.feed[:30]
            st.session_state.total_today += 1
            st.session_state.last_txn_time = now
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES 2-5 — COMING SOON (built next)
# ══════════════════════════════════════════════════════════════════════════════
def page_coming_soon(name, icon, desc):
    show_stats_bar()
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">{icon} <span>{name}</span></div>
      <div class="page-sub">{desc}</div>
    </div>
    <div class="coming-soon">
      <div style="font-size:3rem;margin-bottom:1rem;">{icon}</div>
      <div style="font-size:1.1rem;font-weight:800;color:#f0f4ff;margin-bottom:0.5rem;">{name}</div>
      <div style="font-size:0.82rem;color:#3a5a7c;">Building next — come back soon!</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ALERT QUEUE
# ══════════════════════════════════════════════════════════════════════════════
def page_alert_queue():
    show_stats_bar()
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🚨 Alert <span>Queue</span></div>
      <div class="page-sub">MEDIUM and HIGH risk transactions awaiting analyst decision</div>
    </div>""", unsafe_allow_html=True)

    pending  = [t for t in st.session_state.alert_queue if t["status"] == "Pending"]
    reviewed = [t for t in st.session_state.alert_queue if t["status"] != "Pending"]

    # Summary
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.5rem;">
      <div class="stat-card amber">
        <div class="stat-label">Pending Review</div>
        <div class="stat-value amber">{len(pending)}</div>
        <div class="stat-sub">Awaiting analyst action</div>
      </div>
      <div class="stat-card red">
        <div class="stat-label">High Risk</div>
        <div class="stat-value red">{len([t for t in pending if t['tier']['tier']=='HIGH'])}</div>
        <div class="stat-sub">Needs urgent attention</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-label">Medium Risk</div>
        <div class="stat-value blue">{len([t for t in pending if t['tier']['tier']=='MEDIUM'])}</div>
        <div class="stat-sub">Borderline cases</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not pending:
        st.markdown("""
        <div style="background:#070b14;border:1px dashed #1a2a42;border-radius:16px;padding:3rem;text-align:center;">
          <div style="font-size:2.5rem;margin-bottom:0.75rem;">✅</div>
          <div style="font-size:1rem;font-weight:700;color:#10b981;margin-bottom:0.3rem;">Queue is clear!</div>
          <div style="font-size:0.8rem;color:#3a5a7c;">All transactions have been reviewed. New alerts will appear here automatically.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for i, txn in enumerate(pending):
            t      = txn["tier"]
            color  = t["color"]
            bg     = t["bg"]
            border = t["border"]
            city   = get_city(txn.get("dist_home", 5))
            elapsed= get_elapsed(txn.get("timestamp", time.time()))
            flags  = get_flags(txn)
            card   = txn.get("card", "**** **** **** 0000")
            votes  = txn.get("votes", [False, False, False])
            v_icons= ["🔴" if v else "🟢" for v in votes]

            with st.container():
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border};border-radius:14px;padding:1.25rem;margin-bottom:1rem;">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                    <div style="display:flex;align-items:center;gap:10px;">
                      <span style="font-size:1.3rem;">{t['icon']}</span>
                      <div>
                        <div style="font-size:0.95rem;font-weight:800;color:#f0f4ff;">{txn['merchant']}</div>
                        <div style="font-size:0.65rem;color:#3a5a7c;font-family:'JetBrains Mono',monospace;">{txn['id']} · {elapsed}</div>
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:1.1rem;font-weight:800;color:#e0e8f5;font-family:'JetBrains Mono',monospace;">{txn['amount']}</div>
                      <div style="font-size:0.72rem;font-weight:700;color:{color};">{t['tier']} RISK · {int(txn['score']*100)}%</div>
                    </div>
                  </div>
                  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:0.75rem;">
                    <span style="font-size:0.65rem;color:#3a5a7c;">💳 {card}</span>
                    <span style="font-size:0.65rem;color:#3a5a7c;">📍 {city}</span>
                    <span style="font-size:0.65rem;color:#3a5a7c;">RF:{v_icons[0]} LR:{v_icons[1]} SVM:{v_icons[2]}</span>
                    <span style="font-size:0.65rem;color:#3a5a7c;">⚠️ {flags}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Note input + action buttons
                note = st.text_input("Analyst note (optional)", placeholder="e.g. Customer called to confirm transaction...", key=f"note_{txn['id']}", label_visibility="collapsed")
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button(f"✅ Approve", key=f"approve_{txn['id']}_{i}", use_container_width=True):
                        txn["status"]      = "Approved"
                        txn["notes"]       = note
                        txn["analyst"]     = st.session_state.username
                        txn["resolved_at"] = datetime.now().strftime("%H:%M:%S")
                        st.session_state.cases.insert(0, {**txn, "case_status": "Resolved"})
                        st.session_state.audit_log.insert(0, {**txn, "decision": "Approved", "analyst": st.session_state.username, "resolved_at": txn["resolved_at"]})
                        st.session_state.alert_queue = [t for t in st.session_state.alert_queue if t["id"] != txn["id"]]
                        st.rerun()
                with b2:
                    if st.button(f"🚫 Block", key=f"block_{txn['id']}_{i}", use_container_width=True):
                        txn["status"]      = "Blocked"
                        txn["notes"]       = note
                        txn["analyst"]     = st.session_state.username
                        txn["resolved_at"] = datetime.now().strftime("%H:%M:%S")
                        st.session_state.fraud_blocked += 1
                        st.session_state.money_saved   += txn["amount_val"]
                        st.session_state.cases.insert(0, {**txn, "case_status": "Resolved"})
                        st.session_state.audit_log.insert(0, {**txn, "decision": "Blocked", "analyst": st.session_state.username, "resolved_at": txn["resolved_at"]})
                        st.session_state.alert_queue = [t for t in st.session_state.alert_queue if t["id"] != txn["id"]]
                        st.rerun()
                with b3:
                    if st.button(f"🔼 Escalate", key=f"escalate_{txn['id']}_{i}", use_container_width=True):
                        txn["status"]      = "Escalated"
                        txn["notes"]       = note
                        txn["analyst"]     = st.session_state.username
                        txn["resolved_at"] = datetime.now().strftime("%H:%M:%S")
                        st.session_state.cases.insert(0, {**txn, "case_status": "Investigating"})
                        st.session_state.audit_log.insert(0, {**txn, "decision": "Escalated", "analyst": st.session_state.username, "resolved_at": txn["resolved_at"]})
                        st.session_state.alert_queue = [t for t in st.session_state.alert_queue if t["id"] != txn["id"]]
                        st.rerun()
                st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════
def page_case_manager():
    show_stats_bar()
    st.markdown("""
    <div class="page-header">
      <div class="page-title">📋 Case <span>Manager</span></div>
      <div class="page-sub">Full investigation history — all analyst decisions and escalated cases</div>
    </div>""", unsafe_allow_html=True)

    cases = st.session_state.cases
    resolved    = [c for c in cases if c.get("case_status") == "Resolved"]
    investing   = [c for c in cases if c.get("case_status") == "Investigating"]

    # Summary
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.5rem;">
      <div class="stat-card blue">
        <div class="stat-label">Total Cases</div>
        <div class="stat-value blue">{len(cases)}</div>
        <div class="stat-sub">All time</div>
      </div>
      <div class="stat-card amber">
        <div class="stat-label">Investigating</div>
        <div class="stat-value amber">{len(investing)}</div>
        <div class="stat-sub">Escalated cases</div>
      </div>
      <div class="stat-card green">
        <div class="stat-label">Resolved</div>
        <div class="stat-value green">{len(resolved)}</div>
        <div class="stat-sub">Closed cases</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter
    f1, f2, f3 = st.columns(3)
    with f1:
        if st.button("All Cases", key="cm_all", use_container_width=True):
            st.session_state["cm_filter"] = "All"
            st.rerun()
    with f2:
        if st.button("🔍 Investigating", key="cm_inv", use_container_width=True):
            st.session_state["cm_filter"] = "Investigating"
            st.rerun()
    with f3:
        if st.button("✅ Resolved", key="cm_res", use_container_width=True):
            st.session_state["cm_filter"] = "Resolved"
            st.rerun()

    if "cm_filter" not in st.session_state: st.session_state["cm_filter"] = "All"
    filt = st.session_state["cm_filter"]
    if filt == "Investigating": display_cases = investing
    elif filt == "Resolved":    display_cases = resolved
    else:                       display_cases = cases

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if not display_cases:
        st.markdown("""
        <div style="background:#070b14;border:1px dashed #1a2a42;border-radius:16px;padding:3rem;text-align:center;">
          <div style="font-size:2.5rem;margin-bottom:0.75rem;">📋</div>
          <div style="font-size:0.9rem;color:#3a5a7c;">No cases yet — decisions made in Alert Queue will appear here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for case in display_cases[:20]:
            t          = case["tier"]
            color      = t["color"]
            bg         = t["bg"]
            border     = t["border"]
            status     = case.get("case_status","Resolved")
            decision   = case.get("status","—")
            analyst    = case.get("analyst","System")
            resolved_at= case.get("resolved_at","—")
            notes      = case.get("notes","")
            city       = get_city(case.get("dist_home", 5))
            votes      = case.get("votes", [False,False,False])
            v_icons    = ["🔴" if v else "🟢" for v in votes]
            flags      = get_flags(case)
            s_color    = "#f59e0b" if status == "Investigating" else "#10b981"

            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:14px;padding:1.25rem;margin-bottom:0.75rem;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem;">
                <div style="display:flex;align-items:center;gap:10px;">
                  <span style="font-size:1.2rem;">{t['icon']}</span>
                  <div>
                    <div style="font-size:0.9rem;font-weight:800;color:#f0f4ff;">{case['merchant']}</div>
                    <div style="font-size:0.65rem;color:#3a5a7c;font-family:'JetBrains Mono',monospace;">{case['id']} · 📍 {city}</div>
                  </div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:0.95rem;font-weight:800;color:#e0e8f5;font-family:'JetBrains Mono',monospace;">{case['amount']}</div>
                  <div style="font-size:0.65rem;color:{color};font-weight:700;">{t['tier']} · {int(case['score']*100)}% risk</div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.5rem;margin-bottom:0.6rem;">
                <div style="background:#070b14;border:1px solid #111d2e;border-radius:8px;padding:0.5rem;text-align:center;">
                  <div style="font-size:0.6rem;color:#3a5a7c;text-transform:uppercase;margin-bottom:3px;">Decision</div>
                  <div style="font-size:0.75rem;font-weight:700;color:{color};">{decision}</div>
                </div>
                <div style="background:#070b14;border:1px solid #111d2e;border-radius:8px;padding:0.5rem;text-align:center;">
                  <div style="font-size:0.6rem;color:#3a5a7c;text-transform:uppercase;margin-bottom:3px;">Analyst</div>
                  <div style="font-size:0.75rem;font-weight:700;color:#c8d8f0;">{analyst}</div>
                </div>
                <div style="background:#070b14;border:1px solid #111d2e;border-radius:8px;padding:0.5rem;text-align:center;">
                  <div style="font-size:0.6rem;color:#3a5a7c;text-transform:uppercase;margin-bottom:3px;">Time</div>
                  <div style="font-size:0.75rem;font-weight:700;color:#c8d8f0;">{resolved_at}</div>
                </div>
                <div style="background:#070b14;border:1px solid #111d2e;border-radius:8px;padding:0.5rem;text-align:center;">
                  <div style="font-size:0.6rem;color:#3a5a7c;text-transform:uppercase;margin-bottom:3px;">Status</div>
                  <div style="font-size:0.75rem;font-weight:700;color:{s_color};">{status}</div>
                </div>
              </div>
              <div style="font-size:0.65rem;color:#3a5a7c;">RF:{v_icons[0]} LR:{v_icons[1]} SVM:{v_icons[2]} · {flags}</div>
              {f'<div style="margin-top:0.5rem;background:#070b14;border-left:3px solid {color};border-radius:6px;padding:0.5rem 0.75rem;font-size:0.72rem;color:#8ba4c8;">📝 {notes}</div>' if notes else ""}
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RULES ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def page_rules_engine():
    show_stats_bar()
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🔒 Rules <span>Engine</span></div>
      <div class="page-sub">Set custom rules that run on top of the ML layer — every transaction is checked automatically</div>
    </div>""", unsafe_allow_html=True)

    # Active rules
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Active Rules</div>', unsafe_allow_html=True)

    if not st.session_state.rules:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem;color:#3a5a7c;font-size:0.82rem;">
          No rules yet — create one below
        </div>""", unsafe_allow_html=True)
    else:
        for i, rule in enumerate(st.session_state.rules):
            enabled = rule.get("enabled", True)
            rc1, rc2, rc3 = st.columns([3, 1, 1])
            with rc1:
                type_labels = {"score_above": "Risk score above", "amount_above": "Amount above ₹", "tier_is": "Risk tier is"}
                label = f"{type_labels.get(rule['type'], rule['type'])} {rule['value']}"
                st.markdown(f"""
                <div style="background:#070b14;border:1px solid {'#3b82f6' if enabled else '#1a2a42'};border-radius:10px;padding:0.6rem 1rem;">
                  <div style="font-size:0.8rem;font-weight:700;color:{'#c8d8f0' if enabled else '#3a5a7c'};">🔒 {rule['name']}</div>
                  <div style="font-size:0.68rem;color:#3a5a7c;margin-top:2px;">{label} → Auto Block</div>
                </div>""", unsafe_allow_html=True)
            with rc2:
                toggle = "⏸ Disable" if enabled else "▶ Enable"
                if st.button(toggle, key=f"toggle_rule_{i}", use_container_width=True):
                    st.session_state.rules[i]["enabled"] = not enabled
                    st.rerun()
            with rc3:
                if st.button("🗑 Delete", key=f"delete_rule_{i}", use_container_width=True):
                    st.session_state.rules.pop(i)
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Create new rule
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Create New Rule</div>', unsafe_allow_html=True)

    nr1, nr2, nr3 = st.columns([1.5, 1.5, 1])
    with nr1:
        rule_name = st.text_input("Rule Name", placeholder="e.g. High Amount Online Block")
    with nr2:
        rule_type = st.selectbox("Condition", ["score_above", "amount_above", "tier_is"],
                                  format_func=lambda x: {"score_above":"Risk score above (0-1)","amount_above":"Transaction amount above (₹)","tier_is":"Risk tier is"}[x])
    with nr3:
        if rule_type == "tier_is":
            rule_val = st.selectbox("Value", ["CRITICAL","HIGH","MEDIUM"])
        else:
            default = 0.75 if rule_type == "score_above" else 50000
            rule_val = st.number_input("Value", value=default, min_value=0.0)

    if st.button("➕  Add Rule", use_container_width=True):
        if rule_name:
            st.session_state.rules.append({
                "name":    rule_name,
                "type":    rule_type,
                "value":   rule_val,
                "enabled": True,
            })
            st.success(f"Rule '{rule_name}' added! It will apply to all new transactions.")
            st.rerun()
        else:
            st.error("Please enter a rule name.")

    # Example rules
    st.markdown("""
    <div style="margin-top:0.75rem;background:#070b14;border:1px solid #111d2e;border-radius:10px;padding:0.75rem 1rem;">
      <div style="font-size:0.68rem;color:#3a5a7c;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">Example Rules</div>
      <div style="font-size:0.72rem;color:#3a5a7c;line-height:1.9;">
        • Risk score above 0.75 → Auto Block all high confidence fraud<br>
        • Amount above ₹75,000 → Flag large transactions regardless of score<br>
        • Tier is HIGH → Send all HIGH risk to immediate review
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
def page_audit_log():
    show_stats_bar()
    st.markdown("""
    <div class="page-header">
      <div class="page-title">📄 Audit <span>Log</span></div>
      <div class="page-sub">Every decision logged automatically — full RBI compliance trail</div>
    </div>""", unsafe_allow_html=True)

    log = st.session_state.audit_log

    # Summary
    auto_d    = len([l for l in log if l.get("analyst") == "System"])
    manual_d  = len([l for l in log if l.get("analyst") != "System"])
    blocked_d = len([l for l in log if "Block" in l.get("decision","")])
    approved_d= len([l for l in log if "Approv" in l.get("decision","")])

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1rem;margin-bottom:1.5rem;">
      <div class="stat-card blue"><div class="stat-label">Total Logged</div><div class="stat-value blue">{len(log)}</div><div class="stat-sub">All decisions</div></div>
      <div class="stat-card green"><div class="stat-label">Auto Decisions</div><div class="stat-value green">{auto_d}</div><div class="stat-sub">By system</div></div>
      <div class="stat-card amber"><div class="stat-label">Manual Decisions</div><div class="stat-value amber">{manual_d}</div><div class="stat-sub">By analyst</div></div>
      <div class="stat-card red"><div class="stat-label">Total Blocked</div><div class="stat-value red">{blocked_d}</div><div class="stat-sub">Fraud stopped</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Filter pills
    af1, af2, af3, af4 = st.columns(4)
    for col, label, key in [(af1,"All","al_all"),(af2,"Auto Only","al_auto"),(af3,"Manual Only","al_manual"),(af4,"Blocked Only","al_blocked")]:
        with col:
            if st.button(label, key=key, use_container_width=True):
                st.session_state["al_filter"] = label
                st.rerun()

    if "al_filter" not in st.session_state: st.session_state["al_filter"] = "All"
    af = st.session_state["al_filter"]
    if af == "Auto Only":    log = [l for l in log if l.get("analyst") == "System"]
    elif af == "Manual Only":log = [l for l in log if l.get("analyst") != "System"]
    elif af == "Blocked Only":log= [l for l in log if "Block" in l.get("decision","")]

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Decision Log</div>', unsafe_allow_html=True)

    if not log:
        st.markdown('<div style="text-align:center;padding:2rem;color:#3a5a7c;font-size:0.82rem;">No entries yet — decisions will appear here automatically.</div>', unsafe_allow_html=True)
    else:
        # Header
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1.5fr 1fr 1fr 1fr 1fr;padding:0.4rem 0.75rem;margin-bottom:0.25rem;">
          <span style="font-size:0.6rem;color:#1e3050;text-transform:uppercase;font-weight:700;">TXN ID</span>
          <span style="font-size:0.6rem;color:#1e3050;text-transform:uppercase;font-weight:700;">Merchant</span>
          <span style="font-size:0.6rem;color:#1e3050;text-transform:uppercase;font-weight:700;">Amount</span>
          <span style="font-size:0.6rem;color:#1e3050;text-transform:uppercase;font-weight:700;">Risk</span>
          <span style="font-size:0.6rem;color:#1e3050;text-transform:uppercase;font-weight:700;">Decision</span>
          <span style="font-size:0.6rem;color:#1e3050;text-transform:uppercase;font-weight:700;">Analyst · Time</span>
        </div>""", unsafe_allow_html=True)

        for entry in log[:50]:
            t          = entry["tier"]
            color      = t["color"]
            decision   = entry.get("decision","—")
            analyst    = entry.get("analyst","System")
            resolved_at= entry.get("resolved_at","—")
            d_color    = "#ef4444" if "Block" in decision else ("#10b981" if "Approv" in decision else "#f59e0b")
            a_color    = "#3a5a7c" if analyst == "System" else "#3b82f6"

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1.5fr 1fr 1fr 1fr 1fr;padding:0.55rem 0.75rem;border-bottom:1px solid #0a1020;align-items:center;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#3a5a7c;">{entry['id']}</span>
              <span style="font-size:0.75rem;font-weight:600;color:#c8d8f0;">{entry['merchant']}</span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#e0e8f5;">{entry['amount']}</span>
              <span style="font-size:0.7rem;font-weight:700;color:{color};">{t['tier']} {int(entry['score']*100)}%</span>
              <span style="font-size:0.72rem;font-weight:700;color:{d_color};">{decision}</span>
              <span style="font-size:0.65rem;color:{a_color};">{analyst} · {resolved_at}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    show_login()
else:
    show_sidebar()

    page = st.session_state.page
    if   page == "Command Center": page_command_center()
    elif page == "Alert Queue":    page_alert_queue()
    elif page == "Case Manager":   page_case_manager()
    elif page == "Rules Engine":   page_rules_engine()
    elif page == "Audit Log":      page_audit_log()
