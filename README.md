# 🚚 DeliveryShield

> **AI-powered first-attempt delivery failure predictor — built on Amazon ALMRRC research data**

DeliveryShield predicts whether a specific delivery will fail on the first attempt **before the driver leaves the depot**, giving dispatch teams a 2-hour window to intervene and prevent the failure from occurring.

Most last-mile ML research focuses on route optimization *after* dispatch. DeliveryShield shifts the intervention point **upstream**, targeting the dispatch stage where cost can actually be prevented.

---

## 🎯 The Problem

- 5–10% of Amazon deliveries fail on the first attempt
- Each failure costs ~**$17.78** in re-routing, rescheduling, and customer service
- Existing tools are **reactive** — they fix failures after they happen
- DeliveryShield is **proactive** — it flags failures before the driver leaves

---

## 📊 Results

| Model | Accuracy | F1 Score | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 77.7% | 0.274 | 0.674 | 0.206 | 0.409 |
| LightGBM | 73.9% | 0.274 | 0.680 | 0.192 | 0.477 |
| **XGBoost (tuned + calibrated)** | **84.3%** | **0.348** | **0.751** | **0.304** | **0.406** |

> Best model selected automatically by AUC-ROC. Hyperparameters tuned via RandomizedSearchCV (40 iterations, 3-fold stratified CV). Probabilities calibrated using isotonic regression.

---

## 🏗️ Project Structure

```
deliveryshield/
├── src/
│   ├── generate_data.py     ← Synthetic ALMRRC-style dataset generator (50,000 rows)
│   └── train_model.py       ← Full training pipeline with tuning & calibration
├── app/
│   └── app.py               ← Streamlit web application
├── requirements.txt
└── README.md
```

> `data/` and `models/` are generated locally and excluded from version control.

---

## ⚙️ Features

**25 input features across 5 categories:**

| Category | Features |
|---|---|
| Address | Type (house/apt/gated/business), urban density, distance from depot |
| Package | Size, weight |
| Customer | Delivery instructions, note quality, home probability |
| Timing | Hour, day, month, peak hour flag, weekend flag |
| Driver | Experience, daily deliveries, fatigue score |
| Weather | Condition, temperature |
| Engineered | Risk composite score, address risk, timing risk, driver risk index, weather risk |

---

## 🧠 ML Pipeline

1. **Feature Engineering** — 5 composite risk signals derived from base features
2. **Baseline** — Logistic Regression with class weighting
3. **Tuning** — XGBoost with RandomizedSearchCV (40 combinations, AUC scoring)
4. **Calibration** — CalibratedClassifierCV (isotonic) for reliable probabilities
5. **Explainability** — SHAP TreeExplainer for per-prediction factor breakdown
6. **Evaluation** — Optimal threshold selection, confusion matrix, model comparison

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/deliveryshield.git
cd deliveryshield
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate dataset
```bash
python src/generate_data.py
```

### 5. Train model (~3 min)
```bash
python src/train_model.py
```

### 6. Launch app
```bash
streamlit run app/app.py
```

---

## 🖥️ App Features

- **Live risk score** — updates in real time as you adjust sliders
- **SHAP explanations** — shows which factors are driving the risk score
- **Recommended actions** — actionable dispatch recommendations
- **Batch prediction** — upload a CSV of deliveries, get risk scores for all at once
- **Model performance tab** — comparison charts, confusion matrix, SHAP summary

---

## 📦 Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Primary gradient boosted classifier |
| LightGBM | Secondary model for comparison |
| scikit-learn | Logistic Regression baseline, RandomizedSearchCV, calibration |
| SHAP | Per-prediction explainability |
| Streamlit | Web interface |
| Pandas / NumPy | Data pipeline and feature engineering |
| Plotly / Matplotlib | Visualizations |
| imbalanced-learn | SMOTE (tested, removed — scale_pos_weight performed better) |

---

## 📁 Dataset

Inspired by the **[Amazon Last Mile Routing Research Challenge (ALMRRC)](https://registry.opendata.aws/amazon-last-mile-routing-research-challenge/)** dataset, released publicly via MIT, covering 213,000+ delivery stops across 5 US cities.

This implementation uses a synthetic dataset with the same feature schema, enriched with weather and driver fatigue signals. The ALMRRC dataset focuses on route sequencing; this project addresses the complementary problem of pre-dispatch failure prediction.

**Dataset stats:** 50,000 rows · 25 features · ~10.3% failure rate · ~15 MB

---

## 💡 Novelty

The Amazon ALMRRC challenge asked: *"given a route, predict the best stop sequence."*

DeliveryShield asks a different question: *"given a delivery, will it fail — and why?"*

This shifts the intervention from post-dispatch route optimization to **pre-dispatch risk triage**, which is where the $17.78/failure cost can actually be prevented.

---

*Built for Amazon ML Summer School 2025 application.*