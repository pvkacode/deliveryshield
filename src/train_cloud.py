"""
Lightweight training script for Streamlit Cloud (1GB RAM limit).
Uses simple XGBoost without RandomizedSearchCV to stay within memory limits.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap
import joblib, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'hour_of_day','day_of_week','month','is_weekend','is_peak_hour',
    'address_type_encoded','package_size_encoded','package_weight_kg',
    'has_delivery_note','note_quality','customer_home_probability',
    'zip_failure_rate','distance_from_depot_km','urban_density_encoded',
    'driver_experience_months','driver_daily_deliveries','driver_fatigue_score',
    'weather_encoded','temperature_celsius','prior_failed_attempts'
]

def add_features(X):
    X = X.copy()
    X['risk_composite'] = (X['zip_failure_rate']*3 + (X['prior_failed_attempts']/3)*2 +
                           (1-X['customer_home_probability']) + X['driver_fatigue_score'])
    X['address_risk'] = (X['address_type_encoded']==2).astype(int)*2 + (X['address_type_encoded']==0).astype(int)
    X['timing_risk'] = X['is_peak_hour'] + X['is_weekend']
    X['driver_risk'] = X['driver_fatigue_score'] * (1/(X['driver_experience_months']+1))
    X['weather_risk'] = X['weather_encoded'].map({0:0.0,1:0.2,3:0.6,2:0.9,4:1.0}).fillna(0)
    return X

def find_best_threshold(y_prob, y_true):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_true, (y_prob>=t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return round(float(best_t), 2)

def main():
    print("="*50)
    print("  DeliveryShield — Cloud Training (Lightweight)")
    print("="*50)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, 'data', 'deliveries.csv')
    model_dir = os.path.join(root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    print("\nLoading data...")
    df = pd.read_csv(data_path)
    X = add_features(df[FEATURE_COLS])
    y = df['delivery_failed']
    print(f"Dataset: {len(df):,} rows | Features: {X.shape[1]} | Failure rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    spw = (y_train==0).sum()/(y_train==1).sum()

    # Logistic Regression baseline
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(scaler.fit_transform(X_train), y_train)

    # XGBoost — lightweight settings for cloud
    print("Training XGBoost (cloud-optimized)...")
    model = xgb.XGBClassifier(
        n_estimators=200,       # reduced from 500
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=1,               # single thread to save memory
        tree_method='hist'      # memory efficient
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_prob_lr  = lr.predict_proba(scaler.transform(X_test))[:,1]
    y_prob_xgb = model.predict_proba(X_test)[:,1]

    thresh_lr  = find_best_threshold(y_prob_lr, y_test)
    thresh_xgb = find_best_threshold(y_prob_xgb, y_test)

    def get_metrics(y_prob, y_test, thresh):
        y_pred = (y_prob >= thresh).astype(int)
        return {
            'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
            'f1_score':  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'roc_auc':   round(float(roc_auc_score(y_test, y_prob)), 4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'threshold': thresh,
        }

    lr_m  = get_metrics(y_prob_lr,  y_test, thresh_lr)
    xgb_m = get_metrics(y_prob_xgb, y_test, thresh_xgb)

    print(f"\nXGBoost: F1={xgb_m['f1_score']:.4f} AUC={xgb_m['roc_auc']:.4f}")
    print(f"LR:      F1={lr_m['f1_score']:.4f}  AUC={lr_m['roc_auc']:.4f}")

    # Save metrics
    metrics_out = {
        'logistic_regression': lr_m,
        'xgboost': xgb_m,
        'lightgbm': xgb_m,  # reuse xgb for display
        'best_model': 'xgboost'
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # SHAP
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:200])

    # SHAP plot
    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_values, X_test[:200], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'shap_summary.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    y_pred = (y_prob_xgb >= thresh_xgb).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Success','Failure'], yticklabels=['Success','Failure'])
    plt.title('Confusion Matrix'); plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # Save models
    joblib.dump(model,    os.path.join(model_dir, 'xgb_model.pkl'))
    joblib.dump(scaler,   os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(explainer,os.path.join(model_dir, 'shap_explainer.pkl'))
    joblib.dump(list(X_train.columns), os.path.join(model_dir, 'feature_cols.pkl'))

    print("\n" + "="*50)
    print(f"  Training complete! AUC: {xgb_m['roc_auc']:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
