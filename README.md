# 🛡️ FraudShield — Bank Fraud Operations Platform

> A complete, bank-grade credit card fraud detection and operations platform powered by a 3-model Machine Learning ensemble. Built with Python and Streamlit, trained on 1,000,000 real transactions.

🔴 **[Live Demo](https://fraudshield-4utba8ns9xovknturwazky.streamlit.app)**

---

## 📌 Overview

FraudShield is not just a fraud detection model — it is a complete fraud operations platform that mirrors how real bank fraud teams work. The system automatically scores every transaction using 3 ML models, routes 95% of decisions automatically, and surfaces only genuinely uncertain cases to human analysts.

---

## 🏗️ System Architecture

```
Transaction Arrives
        ↓
3 ML Models Score Simultaneously (RF + LR + SVM)
        ↓
Ensemble Majority Voting → Risk Score (0–100%)
        ↓
5-Tier Classification: CRITICAL / HIGH / MEDIUM / LOW / SAFE
        ↓
Rules Engine Check (Custom analyst-defined rules)
        ↓
┌─────────────────────────────────┐
│  SAFE / LOW   → Auto Approved   │
│  CRITICAL     → Auto Blocked    │
│  MEDIUM / HIGH → Alert Queue    │
└─────────────────────────────────┘
        ↓
Analyst Reviews → Approve / Block / Escalate
        ↓
Audit Log (RBI Compliance Trail)
```

---

## 🤖 ML Models

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 🌲 Random Forest | 100.00% | 100.00% | 99.99% | 99.99% |
| 📈 Logistic Regression | 95.88% | 89.15% | 60.01% | 71.73% |
| ⚡ SVM (Calibrated) | 93.56% | 90.39% | 29.32% | 44.28% |

All models trained on the **Kaggle Credit Card Fraud Detection** dataset — 1,000,000 transactions.

### Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| Ratio to Median Purchase Price | 52.72% |
| Online Order | 16.94% |
| Distance from Home | 13.49% |
| Used PIN Number | 6.39% |
| Used Chip | 5.21% |
| Distance from Last Transaction | 4.57% |
| Repeat Retailer | 0.68% |

---

## 🖥️ Platform Pages

### 🏠 Command Center
- Live transaction feed — auto-generates every few seconds
- Real-time stats — Transactions Today, Fraud Blocked, Money Saved, Alerts Pending
- Risk Trend Chart — last 20 transaction risk scores visualized
- Automation breakdown — auto-approved vs auto-blocked vs sent to queue
- Risk distribution by tier
- Pause/Resume feed, Speed control, Filter pills, Search by TXN ID or Merchant

### 🚨 Alert Queue
- Only MEDIUM and HIGH risk transactions appear here
- Full transaction details — masked card number, city, model votes, flags
- Analyst adds a note explaining their decision
- One-click Approve / Block / Escalate
- Decision automatically logged to Case Manager and Audit Log

### 📋 Case Manager
- Complete history of all analyst decisions
- Filter by All / Investigating / Resolved
- Approved → Green, Blocked → Red, Escalated → Amber
- Full case details — model votes, flags, notes, analyst, timestamp

### 🔒 Rules Engine
- Create custom rules that run on top of the ML layer
- 6 rule types: Risk Score, Amount, Tier, No Chip+No PIN, Distance, Unknown Merchant
- Rule action: Auto Block or Escalate to Queue
- Live hit counter — ⚡ X triggered per rule
- 5 preset rules for one-click addition
- Toggle rules on/off without deleting

### 📄 Audit Log
- Every decision logged automatically — manual and auto
- Filter by All / Auto Only / Manual Only / Blocked Only
- RBI compliance trail — analyst name, timestamp, decision on every entry

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Streamlit |
| ML Library | Scikit-learn |
| Model Storage | Pickle |
| Live Refresh | streamlit-autorefresh |
| Training Environment | Google Colab |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fraudshield.git
cd fraudshield
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Make sure model files are present
```
fraudshield/
├── fraud_detection_app.py
├── requirements.txt
├── fraud_model_rf.pkl
├── fraud_model_lr.pkl
├── fraud_model_svm.pkl
└── scaler.pkl
```

### 4. Run the app
```bash
streamlit run fraud_detection_app.py
```

---

## 📊 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/dalpozz/creditcardfraud)
- **Size:** 1,000,000 transactions
- **Split:** 800,000 train / 200,000 test
- **Fraud Rate:** 8.7% (artificially balanced for ML training)
- **Real-world fraud rate:** ~0.1–0.3%

---

## 🔑 Key Features

- ✅ **95% Automated** — only MEDIUM/HIGH risk reaches human analysts
- ✅ **3-Model Ensemble** — majority voting for reliable predictions
- ✅ **5-Tier Risk Scoring** — CRITICAL / HIGH / MEDIUM / LOW / SAFE
- ✅ **Live Simulation** — transactions processed in real time
- ✅ **Custom Rules Engine** — analyst-defined rules override ML
- ✅ **Audit Trail** — every decision logged for RBI compliance
- ✅ **Natural Language Explanations** — every prediction explained
- ✅ **Batch CSV Upload** — process multiple transactions at once

---

## 👥 Team

Developed as a Mini Project for Third Year Computer Engineering
Mumbai University

---

## 📄 License

This project is for educational purposes.
