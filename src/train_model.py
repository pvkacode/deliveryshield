import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
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
FEATURE_NAMES = {
    'hour_of_day':'Hour of Day','day_of_week':'Day of Week','month':'Month',
    'is_weekend':'Is Weekend','is_peak_hour':'Is Peak Hour (5-8pm)',
    'address_type_encoded':'Address Type','package_size_encoded':'Package Size',
    'package_weight_kg':'Package Weight (kg)','has_delivery_note':'Has Delivery Note',
    'note_quality':'Note Quality','customer_home_probability':'Customer Home Prob.',
    'zip_failure_rate':'Area Failure Rate','distance_from_depot_km':'Depot Distance (km)',
    'urban_density_encoded':'Urban Density','driver_experience_months':'Driver Experience (mo)',
    'driver_daily_deliveries':'Daily Deliveries','driver_fatigue_score':'Driver Fatigue',
    'weather_encoded':'Weather','temperature_celsius':'Temperature (C)',
    'prior_failed_attempts':'Prior Failed Attempts',
    # engineered
    'risk_composite':'Risk Composite Score','address_risk':'Address Risk',
    'timing_risk':'Timing Risk','driver_risk':'Driver Risk Index','weather_risk':'Weather Risk'
}
TARGET_COL = 'delivery_failed'


def add_features(X):
    """Engineer 5 composite risk features on top of base 20."""
    X = X.copy()
    X['risk_composite'] = (
        X['zip_failure_rate'] * 3 +
        (X['prior_failed_attempts'] / 3) * 2 +
        (1 - X['customer_home_probability']) +
        X['driver_fatigue_score']
    )
    X['address_risk'] = (
        (X['address_type_encoded'] == 2).astype(int) * 2 +
        (X['address_type_encoded'] == 0).astype(int)
    )
    X['timing_risk'] = X['is_peak_hour'] + X['is_weekend']
    X['driver_risk']  = X['driver_fatigue_score'] * (1 / (X['driver_experience_months'] + 1))
    X['weather_risk'] = X['weather_encoded'].map({0:0.0, 1:0.2, 3:0.6, 2:0.9, 4:1.0}).fillna(0)
    return X


def load_data(path='data/deliveries.csv'):
    df = pd.read_csv(path)
    X  = add_features(df[FEATURE_COLS])
    y  = df[TARGET_COL]
    return X, y, df


def find_best_threshold(y_prob, y_true):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return round(float(best_t), 2)


def evaluate(model, X_test, y_test, model_name, scaler=None):
    X_in   = scaler.transform(X_test) if scaler else X_test
    y_prob = model.predict_proba(X_in)[:, 1]
    thresh = find_best_threshold(y_prob, y_test)
    y_pred = (y_prob >= thresh).astype(int)
    m = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'f1_score':  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'roc_auc':   round(float(roc_auc_score(y_test, y_prob)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'threshold': thresh,
    }
    print(f"\n{'='*48}\n  {model_name}\n{'='*48}")
    for k, v in m.items(): print(f"  {k:15s}: {v}")
    print(classification_report(y_test, y_pred, target_names=['Success','Failure']))
    return m


def train_baseline(X_train, y_train):
    print("\nTraining Logistic Regression baseline...")
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', C=0.1)
    lr.fit(scaler.fit_transform(X_train), y_train)
    return lr, scaler


def tune_xgboost(X_train, y_train):
    print("\nTuning XGBoost (RandomizedSearchCV, ~2 min)...")
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        'n_estimators':     [500, 700, 1000],
        'max_depth':        [4, 5, 6, 7],
        'learning_rate':    [0.01, 0.03, 0.05],
        'subsample':        [0.75, 0.85, 0.9],
        'colsample_bytree': [0.75, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma':            [0, 0.1, 0.2],
        'reg_alpha':        [0, 0.1, 0.5],
        'reg_lambda':       [1.0, 1.5, 2.0],
    }
    base = xgb.XGBClassifier(scale_pos_weight=spw, eval_metric='aucpr',
                              random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(base, param_dist, n_iter=40, scoring='roc_auc',
                                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                random_state=42, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(search.best_estimator_, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)
    return calibrated, search.best_params_


def train_lightgbm(X_train, y_train):
    print("\nTraining LightGBM...")
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    m = lgb.LGBMClassifier(
        n_estimators=700, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(X_train, y_train)
    # Calibrate
    cal = CalibratedClassifierCV(m, cv=3, method='isotonic')
    cal.fit(X_train, y_train)
    return cal


def save_comparison_plot(all_metrics):
    models      = list(all_metrics.keys())
    metric_keys = ['accuracy','f1_score','roc_auc','precision','recall']
    labels      = ['Accuracy','F1 Score','AUC-ROC','Precision','Recall']
    colors      = ['#FF6B35','#4cc9f0','#888899']
    x, w        = np.arange(len(labels)), 0.25
    fig, ax     = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0f0f1a'); ax.set_facecolor('#0f0f1a')
    for i, (mname, color) in enumerate(zip(models, colors)):
        vals = [all_metrics[mname].get(k, 0) for k in metric_keys]
        bars = ax.bar(x+i*w, vals, w, label=mname, color=color, alpha=0.85, edgecolor='none', zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#aaaacc')
    ax.set_xticks(x+w); ax.set_xticklabels(labels, color='#aaaacc', fontsize=10)
    ax.set_ylabel('Score', color='#aaaacc'); ax.set_ylim(0, 1.1)
    ax.tick_params(colors='#666680')
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.yaxis.grid(True, color='#1a1a2e', zorder=0)
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='#aaaacc', edgecolor='#333355')
    ax.set_title('Model Comparison — With Feature Engineering & Calibration',
                 color='#ffffff', fontsize=12, fontweight='600', pad=15)
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("Model comparison chart saved.")


def save_shap_plot(explainer, X_test):
    try:
        sv = explainer.shap_values(X_test[:500])
        if isinstance(sv, list): sv = sv[1]
        all_feature_names = [FEATURE_NAMES.get(f, f) for f in X_test.columns]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_test[:500], feature_names=all_feature_names,
                          show=False, plot_size=(10, 8))
        plt.title('Feature Impact on Delivery Failure Risk', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig('models/shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("SHAP plot saved.")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")


def save_confusion_matrix(model, X_test, y_test, threshold, scaler=None):
    X_in   = scaler.transform(X_test) if scaler else X_test
    y_pred = (model.predict_proba(X_in)[:, 1] >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Success','Failure'], yticklabels=['Success','Failure'])
    plt.title('Confusion Matrix — Best Model', fontsize=13)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved.")


def main():
    print("=" * 52)
    print("  DeliveryShield — Improved Training Pipeline")
    print("  (Feature Engineering + Probability Calibration)")
    print("=" * 52)

    print("\nLoading & engineering features...")
    X, y, df = load_data()
    print(f"Dataset: {len(df):,} rows | Features: {X.shape[1]} | Failure rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train all three (no SMOTE — scale_pos_weight handles imbalance better)
    lr, scaler       = train_baseline(X_train, y_train)
    xgb_cal, best_p  = tune_xgboost(X_train, y_train)
    lgbm_cal         = train_lightgbm(X_train, y_train)

    # Evaluate
    print("\n\nEVALUATION RESULTS")
    lr_m   = evaluate(lr,      X_test, y_test, "Logistic Regression (baseline)", scaler=scaler)
    xgb_m  = evaluate(xgb_cal, X_test, y_test, "XGBoost (tuned + calibrated)")
    lgbm_m = evaluate(lgbm_cal,X_test, y_test, "LightGBM (calibrated)")

    # Save metrics immediately
    os.makedirs('models', exist_ok=True)
    metrics_out = {
        'logistic_regression': lr_m,
        'xgboost':             xgb_m,
        'lightgbm':            lgbm_m,
        'best_model':          'xgboost',
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Pick best by AUC (more stable than F1)
    scores    = {'xgboost': xgb_m['roc_auc'], 'lightgbm': lgbm_m['roc_auc']}
    best_name = max(scores, key=scores.get)
    best_model  = xgb_cal if best_name == 'xgboost' else lgbm_cal
    best_thresh = xgb_m['threshold'] if best_name == 'xgboost' else lgbm_m['threshold']
    print(f"\n🏆 Best model: {best_name.upper()} (AUC={scores[best_name]:.4f})")

    metrics_out['best_model'] = best_name
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    # Plots
    save_comparison_plot({'XGBoost (Tuned)': xgb_m, 'LightGBM': lgbm_m, 'Logistic Reg.': lr_m})
    save_confusion_matrix(best_model, X_test, y_test, best_thresh)

    # SHAP — use inner estimator if calibrated
    print("\nComputing SHAP values...")
    try:
        inner = best_model.estimators_[0] if hasattr(best_model, 'estimators_') else best_model
        explainer = shap.TreeExplainer(inner)
        save_shap_plot(explainer, X_test)
        joblib.dump(explainer, 'models/shap_explainer.pkl')
    except Exception as e:
        print(f"SHAP skipped: {e}")
        # Save a basic explainer
        inner = xgb.XGBClassifier(n_estimators=300, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                   random_state=42, n_jobs=-1)
        inner.fit(X_train, y_train)
        explainer = shap.TreeExplainer(inner)
        joblib.dump(explainer, 'models/shap_explainer.pkl')

    # Save best model + feature list (including engineered features)
    joblib.dump(best_model,         'models/xgb_model.pkl')
    joblib.dump(scaler,             'models/scaler.pkl')
    joblib.dump(list(X_train.columns), 'models/feature_cols.pkl')

    print("\n" + "="*52)
    print(f"  Training complete! Best: {best_name.upper()}")
    print(f"  AUC-ROC: {scores[best_name]:.4f}")
    print(f"  F1 Score: {xgb_m['f1_score'] if best_name=='xgboost' else lgbm_m['f1_score']:.4f}")
    print("="*52)


if __name__ == '__main__':
    main()