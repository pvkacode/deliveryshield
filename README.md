# 🚚 DeliveryShield

**AI-powered first-attempt delivery failure predictor**

Predicts whether a specific Amazon delivery will fail on the first attempt — *before the driver leaves the depot* — using XGBoost + SHAP explainability.

---

## Results

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression (baseline) | ~82% | ~0.61 | ~0.88 |
| XGBoost (main model) | ~88% | ~0.72 | ~0.93 |

---

## Project Structure

```
deliveryshield/
├── data/               ← Generated dataset (created on first run)
├── src/
│   ├── generate_data.py    ← Synthetic ALMRRC-style dataset generator
│   └── train_model.py      ← XGBoost + SHAP training pipeline
├── app/
│   └── app.py              ← Streamlit web application
├── models/             ← Saved model artifacts (created after training)
├── notebooks/          ← (Optional) EDA notebook
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Clone / download the project

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the dataset
```bash
python src/generate_data.py
```

### 5. Train the model
```bash
python src/train_model.py
```

### 6. Launch the app
```bash
cd app
streamlit run app.py
```

---

## Dataset

Inspired by the [Amazon Last Mile Routing Research Challenge (ALMRRC)](https://registry.opendata.aws/amazon-last-mile-routing-research-challenge/) dataset, released publicly via MIT. This implementation uses a synthetic dataset with the same feature schema, enriched with weather and driver fatigue features.

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **XGBoost** — gradient boosted classifier
- **SHAP** — model explainability
- **Streamlit** — web interface
- **Plotly** — interactive visualizations

---

*Built for Amazon ML Summer School 2025 application.*
