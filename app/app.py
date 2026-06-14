import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import json, os, io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DeliveryShield", page_icon="🚚",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem !important; max-width: 1400px !important; margin: 0 auto !important; }
section[data-testid="stSidebar"] { display: none; }
.stApp { background: #0a0a0f; color: #e8e8f0; }

/* Nav */
.nav-bar { background: rgba(12,12,20,0.97); backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(255,255,255,0.06); padding: 0 2.5rem;
  height: 58px; display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100; margin: -1.5rem -2.5rem 1.5rem; }
.nav-logo { font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700;
  color:#fff; letter-spacing:-0.02em; display:flex; align-items:center; gap:8px; }
.nav-logo span { color:#FF6B35; }
.nav-badge { background:rgba(255,107,53,0.12); color:#FF6B35;
  border:1px solid rgba(255,107,53,0.25); font-size:0.68rem; font-weight:600;
  padding:3px 10px; border-radius:20px; letter-spacing:0.06em; text-transform:uppercase; }

/* Live risk banner */
.live-banner { border-radius:14px; padding:1.5rem 1.75rem; margin-bottom:1.25rem;
  position:relative; overflow:hidden; transition: all 0.3s ease; }
.live-banner-low  { background:linear-gradient(135deg,rgba(34,197,94,0.1) 0%,rgba(34,197,94,0.03) 100%); border:1px solid rgba(34,197,94,0.2); }
.live-banner-med  { background:linear-gradient(135deg,rgba(245,158,11,0.1) 0%,rgba(245,158,11,0.03) 100%); border:1px solid rgba(245,158,11,0.2); }
.live-banner-high { background:linear-gradient(135deg,rgba(239,68,68,0.12) 0%,rgba(239,68,68,0.03) 100%); border:1px solid rgba(239,68,68,0.25); }
.live-tag { font-size:0.6rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:0.3rem; }
.live-score { font-family:'Space Grotesk',sans-serif; font-size:3.5rem; font-weight:700;
  line-height:1; letter-spacing:-0.04em; }
.live-desc { font-size:0.82rem; color:#8888a0; margin-top:0.6rem; line-height:1.5; }
.live-pulse { display:inline-block; width:7px; height:7px; border-radius:50%;
  margin-right:6px; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Progress bar */
.risk-bar-track { height:6px; background:rgba(255,255,255,0.06); border-radius:3px;
  margin:0.75rem 0 0; overflow:hidden; }
.risk-bar-fill { height:100%; border-radius:3px; transition: width 0.4s ease; }

/* Metric cards */
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-bottom:1.25rem; }
.metric-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
  border-radius:10px; padding:1rem; }
.metric-label { font-size:0.67rem; color:#555570; font-weight:600;
  text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem; }
.metric-val { font-family:'Space Grotesk',sans-serif; font-size:1.5rem;
  font-weight:700; color:#fff; letter-spacing:-0.02em; }

/* Action items */
.action-item { display:flex; align-items:flex-start; gap:0.7rem;
  padding:0.8rem 1rem; background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.06); border-radius:10px;
  margin-bottom:0.5rem; font-size:0.83rem; color:#ccccdd; line-height:1.4; }

/* Section labels */
.section-label { font-size:0.63rem; font-weight:700; letter-spacing:0.12em;
  text-transform:uppercase; color:#444460; margin-bottom:0.85rem;
  padding-bottom:0.5rem; border-bottom:1px solid rgba(255,255,255,0.04); }
.section-heading { font-family:'Space Grotesk',sans-serif; font-size:0.95rem;
  font-weight:600; color:#fff; margin-bottom:1rem; letter-spacing:-0.01em; }

/* Perf metric rows */
.perf-metric { display:flex; align-items:center; justify-content:space-between;
  padding:0.9rem 1.1rem; background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.07); border-radius:10px; margin-bottom:0.55rem; }
.perf-name { font-size:0.83rem; color:#9999bb; }
.perf-val  { font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:600; }
.perf-bar-track { height:3px; background:rgba(255,255,255,0.06); border-radius:2px; margin-top:6px; width:140px; }
.perf-bar-fill  { height:100%; border-radius:2px; }
.divider { height:1px; background:rgba(255,255,255,0.05); margin:1.5rem 0; }

/* Batch table */
.stDataFrame { background:rgba(255,255,255,0.02) !important; }

/* Button */
.stButton > button {
  background:linear-gradient(135deg,#FF6B35 0%,#e05520 100%) !important;
  color:white !important; border:none !important; border-radius:10px !important;
  padding:0.7rem 1.5rem !important; font-family:'Inter',sans-serif !important;
  font-weight:600 !important; font-size:0.88rem !important;
  width:100% !important; box-shadow:0 4px 18px rgba(255,107,53,0.28) !important;
  transition:all 0.2s !important; }
.stButton > button:hover { transform:translateY(-1px) !important;
  box-shadow:0 6px 24px rgba(255,107,53,0.42) !important; }

/* Sliders & selects */
.stSlider > div > div > div { background:#FF6B35 !important; }
.stSelectbox > div > div { background:rgba(255,255,255,0.04) !important;
  border:1px solid rgba(255,255,255,0.09) !important; border-radius:8px !important; }
label[data-testid="stWidgetLabel"] { color:#8888aa !important;
  font-size:0.77rem !important; font-weight:500 !important; }
.stCheckbox > label { color:#ccccdd !important; font-size:0.85rem !important; }
::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
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
    'is_weekend':'Is Weekend','is_peak_hour':'Is Peak Hour',
    'address_type_encoded':'Address Type','package_size_encoded':'Package Size',
    'package_weight_kg':'Package Weight (kg)','has_delivery_note':'Has Delivery Note',
    'note_quality':'Note Quality','customer_home_probability':'Customer Home Prob.',
    'zip_failure_rate':'Area Failure Rate','distance_from_depot_km':'Depot Distance (km)',
    'urban_density_encoded':'Urban Density','driver_experience_months':'Driver Experience (mo)',
    'driver_daily_deliveries':'Daily Deliveries','driver_fatigue_score':'Driver Fatigue',
    'weather_encoded':'Weather','temperature_celsius':'Temperature (C)',
    'prior_failed_attempts':'Prior Failed Attempts'
}
ADDRESS_MAP = {'House':1,'Apartment':0,'Gated Community':2,'Business':3,'PO Box':4}
PACKAGE_MAP = {'Small':3,'Medium':2,'Large':1,'Extra Large':0}
WEATHER_MAP = {'Clear':0,'Cloudy':1,'Rain':3,'Heavy Rain':2,'Snow':4}
DENSITY_MAP = {'Urban':2,'Suburban':1,'Rural':0}
DAYS        = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


def add_features(X):
    """Same feature engineering as training pipeline."""
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

# ── model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model     = joblib.load('models/xgb_model.pkl')
        explainer = joblib.load('models/shap_explainer.pkl')
        with open('models/metrics.json') as f:
            metrics = json.load(f)
        return model, explainer, metrics
    except FileNotFoundError:
        return None, None, None

# ── prediction helper ─────────────────────────────────────────────────────────
def predict(model, row_dict):
    df = pd.DataFrame([row_dict])[FEATURE_COLS]
    df = add_features(df)
    # Use saved feature cols if available
    try:
        feat_cols = joblib.load('models/feature_cols.pkl')
        df = df[feat_cols]
    except:
        pass
    return float(model.predict_proba(df)[0][1]), df

def risk_meta(prob):
    if prob > 0.6:
        return "HIGH RISK",  "#ef4444", "live-banner-high", "Immediate action recommended — this delivery is likely to fail."
    elif prob > 0.35:
        return "MEDIUM RISK","#f59e0b", "live-banner-med",  "Elevated risk detected. Consider precautionary steps."
    else:
        return "LOW RISK",   "#22c55e", "live-banner-low",  "Delivery is likely to succeed on the first attempt."

# ── SHAP chart ────────────────────────────────────────────────────────────────
def shap_fig(explainer, input_df):
    sv = explainer.shap_values(input_df)
    if isinstance(sv, list): sv = sv[1]
    vals  = sv[0] if len(sv.shape) > 1 else sv
    names = [FEATURE_NAMES[f] for f in input_df.columns]
    pairs = sorted(zip(vals, names), key=lambda x: abs(x[0]), reverse=True)[:8]
    v, l  = zip(*pairs)
    cols  = ['#ef4444' if x > 0 else '#22c55e' for x in v]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('none'); ax.set_facecolor('none')
    ax.barh(range(len(l)), v, color=cols, height=0.5, edgecolor='none')
    ax.set_yticks(range(len(l))); ax.set_yticklabels(l, fontsize=9, color='#aaaacc')
    ax.axvline(0, color='#2a2a3a', linewidth=0.8)
    ax.set_xlabel('Impact on failure risk', fontsize=8.5, color='#555570')
    ax.tick_params(axis='x', colors='#555570', labelsize=8)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.tick_params(left=False)
    plt.tight_layout(pad=0.8)
    return fig

# ── recommendations ───────────────────────────────────────────────────────────
def recommendations(prob, address, has_note, prior_fails, weather, peak):
    r = []
    if prob > 0.40:  r.append(("🔴","Contact customer before dispatch — high failure risk"))
    if address in ['Apartment','Gated Community']: r.append(("🏢","Collect access code / buzzer before sending driver"))
    if not has_note: r.append(("📝","Request delivery instructions from customer"))
    if prior_fails > 0: r.append(("🔁",f"Address has {prior_fails} prior failure(s) — verify before dispatch"))
    if weather in ['Rain','Heavy Rain','Snow']: r.append(("🌧️","Poor weather — add buffer time or pre-alert customer"))
    if peak: r.append(("⏰","Peak hour — allow extra time, driver under pressure"))
    if not r: r.append(("✅","Low risk — no special action needed"))
    return r

# ── batch scoring ─────────────────────────────────────────────────────────────
def score_batch(model, df_raw):
    results = []
    for _, row in df_raw.iterrows():
        try:
            inp = {
                'hour_of_day':            int(row.get('hour_of_day', 14)),
                'day_of_week':            int(row.get('day_of_week', 0)),
                'month':                  int(row.get('month', 6)),
                'is_weekend':             int(row.get('is_weekend', 0)),
                'is_peak_hour':           int(row.get('is_peak_hour', 0)),
                'address_type_encoded':   int(row.get('address_type_encoded', 1)),
                'package_size_encoded':   int(row.get('package_size_encoded', 2)),
                'package_weight_kg':      float(row.get('package_weight_kg', 2.5)),
                'has_delivery_note':      int(row.get('has_delivery_note', 1)),
                'note_quality':           int(row.get('note_quality', 2)),
                'customer_home_probability': float(row.get('customer_home_probability', 0.6)),
                'zip_failure_rate':       float(row.get('zip_failure_rate', 0.1)),
                'distance_from_depot_km': float(row.get('distance_from_depot_km', 15)),
                'urban_density_encoded':  int(row.get('urban_density_encoded', 2)),
                'driver_experience_months': int(row.get('driver_experience_months', 12)),
                'driver_daily_deliveries': int(row.get('driver_daily_deliveries', 60)),
                'driver_fatigue_score':   float(row.get('driver_fatigue_score', 0.5)),
                'weather_encoded':        int(row.get('weather_encoded', 0)),
                'temperature_celsius':    float(row.get('temperature_celsius', 20)),
                'prior_failed_attempts':  int(row.get('prior_failed_attempts', 0)),
            }
            prob, _ = predict(model, inp)
            label, color, _, _ = risk_meta(prob)
            results.append({'Risk Score': f"{prob*100:.1f}%", 'Risk Level': label,
                            '⚠️ Action': '✅ OK' if prob < 0.15 else '⚠️ Review' if prob < 0.40 else '🔴 Act Now'})
        except Exception as e:
            results.append({'Risk Score': 'Error', 'Risk Level': str(e), '⚠️ Action': '—'})
    return pd.DataFrame(results)

# ── load ──────────────────────────────────────────────────────────────────────
model, explainer, metrics = load_model()

st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">🚚 Delivery<span>Shield</span></div>
  <div class="nav-badge">Amazon ALMRRC Research</div>
</div>""", unsafe_allow_html=True)

if model is None:
    st.markdown("""<div style="display:flex;align-items:center;justify-content:center;
      height:75vh;flex-direction:column;gap:1rem;">
      <div style="font-size:2.5rem">⚠️</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;color:#fff">Model not trained yet</div>
      <div style="color:#666680;font-size:0.87rem">Run <code style="background:rgba(255,255,255,0.08);
      padding:2px 8px;border-radius:4px">python src/train_model.py</code> from the project root</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📦 Batch Predict", "📊 Model Performance", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1.1], gap="large")

    with col_form:
        st.markdown('<div class="section-heading">Delivery details</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">📦 Package & Address</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            address_type = st.selectbox("Address type", list(ADDRESS_MAP.keys()), index=1, key='at')
            package_size = st.selectbox("Package size", list(PACKAGE_MAP.keys()), key='ps')
        with c2:
            urban_density = st.selectbox("Area type", list(DENSITY_MAP.keys()), key='ud')
            prior_fails   = st.selectbox("Prior failures at address", [0,1,2,3], index=1, key='pf')
        package_weight = st.slider("Package weight (kg)", 0.1, 30.0, 2.5, 0.1, key='pw')
        zip_failure    = st.slider("Area historical failure rate", 0.0, 0.5, 0.25, 0.01, key='zf')
        distance       = st.slider("Distance from depot (km)", 0.5, 80.0, 15.0, 0.5, key='d')

        st.markdown('<div class="section-label" style="margin-top:1.25rem">👤 Customer</div>', unsafe_allow_html=True)
        has_note     = st.checkbox("Customer left delivery instructions", value=True, key='hn')
        note_quality = st.select_slider("Instruction quality", [0,1,2,3],
                         format_func=lambda x:['None','Vague','OK','Detailed'][x],
                         value=2, key='nq') if has_note else 0
        customer_home = st.slider("Likelihood customer is home", 0.0, 1.0, 0.35, 0.05, key='ch')

        st.markdown('<div class="section-label" style="margin-top:1.25rem">🗓️ Timing & Weather</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            hour  = st.slider("Hour of delivery", 8, 21, 18, key='h')
            month = st.slider("Month", 1, 12, 6, key='mo')
        with c2:
            day     = st.selectbox("Day", DAYS, key='dy')
            weather = st.selectbox("Weather", list(WEATHER_MAP.keys()), key='w')
        temperature = st.slider("Temperature (°C)", -10, 45, 20, key='t')
        day_enc    = DAYS.index(day)
        is_weekend = int(day_enc >= 5)
        is_peak    = int(17 <= hour <= 20)

        st.markdown('<div class="section-label" style="margin-top:1.25rem">🧑‍✈️ Driver</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: driver_exp = st.slider("Experience (months)", 1, 60, 12, key='de')
        with c2: daily_del  = st.slider("Deliveries today", 30, 120, 60, key='dd')
        fatigue = round(daily_del / 120, 3)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Run prediction →", use_container_width=True, type="primary")

    # ── right panel ───────────────────────────────────────────────────────────
    with col_result:
        # Build input every render for live score
        input_data = {
            'hour_of_day': hour, 'day_of_week': day_enc, 'month': month,
            'is_weekend': is_weekend, 'is_peak_hour': is_peak,
            'address_type_encoded': ADDRESS_MAP[address_type],
            'package_size_encoded': PACKAGE_MAP[package_size],
            'package_weight_kg': package_weight,
            'has_delivery_note': int(has_note), 'note_quality': note_quality,
            'customer_home_probability': customer_home,
            'zip_failure_rate': zip_failure, 'distance_from_depot_km': distance,
            'urban_density_encoded': DENSITY_MAP[urban_density],
            'driver_experience_months': driver_exp,
            'driver_daily_deliveries': daily_del, 'driver_fatigue_score': fatigue,
            'weather_encoded': WEATHER_MAP[weather],
            'temperature_celsius': temperature,
            'prior_failed_attempts': prior_fails
        }
        prob, input_df = predict(model, input_data)
        label, color, banner_cls, desc = risk_meta(prob)
        bar_pct = int(prob * 100)

        # Live risk banner (always visible, updates with sliders)
        st.markdown(f"""
        <div class="live-banner {banner_cls}">
          <div class="live-tag" style="color:{color}">
            <span class="live-pulse" style="background:{color}"></span>LIVE RISK SCORE
          </div>
          <div class="live-score" style="color:{color}">{prob*100:.1f}<span style="font-size:1.4rem;opacity:0.55">%</span></div>
          <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{bar_pct}%;background:{color}"></div>
          </div>
          <div class="live-desc">{label} — {desc}</div>
        </div>""", unsafe_allow_html=True)

        # Quick stats
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-val" style="color:{color};font-size:1.1rem">{label}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-val">{min(prob, 1-prob)*200:.0f}<span style="font-size:0.9rem;color:#555570">%</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Full SHAP + actions only after button click
        if predict_btn or st.session_state.get('_predicted'):
            st.session_state['_predicted'] = True

            st.markdown('<div class="section-heading">Key risk factors</div>', unsafe_allow_html=True)
            fig = shap_fig(explainer, input_df)
            st.pyplot(fig, use_container_width=True, transparent=True)
            plt.close()
            st.caption("🟥 Red = increases failure risk &nbsp;|&nbsp; 🟩 Green = reduces failure risk")

            st.markdown('<div class="section-heading" style="margin-top:1rem">Recommended actions</div>', unsafe_allow_html=True)
            for icon, text in recommendations(prob, address_type, has_note, prior_fails, weather, is_peak):
                st.markdown(f'<div class="action-item"><span style="flex-shrink:0">{icon}</span>{text}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin-top:1.5rem;padding:1.5rem;background:rgba(255,255,255,0.02);
              border:1px dashed rgba(255,255,255,0.08);border-radius:12px;
              text-align:center;color:#444460;">
              <div style="font-size:1.5rem;margin-bottom:0.5rem">🎯</div>
              <div style="font-size:0.85rem">Click <strong style="color:#FF6B35">Run prediction</strong> to see
              SHAP explanations and recommended actions</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-heading">Batch delivery risk scoring</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#666680;font-size:0.85rem;margin-bottom:1.5rem">Upload a CSV of deliveries and get risk scores for all of them at once. Download the results to share with your dispatch team.</div>', unsafe_allow_html=True)

    # Template download
    template_cols = ['hour_of_day','day_of_week','month','is_weekend','is_peak_hour',
                     'address_type_encoded','package_size_encoded','package_weight_kg',
                     'has_delivery_note','note_quality','customer_home_probability',
                     'zip_failure_rate','distance_from_depot_km','urban_density_encoded',
                     'driver_experience_months','driver_daily_deliveries','driver_fatigue_score',
                     'weather_encoded','temperature_celsius','prior_failed_attempts']
    sample_rows = [
        [14,0,6,0,0,1,2,2.5,1,2,0.6,0.1,15,2,12,60,0.5,0,20,0],
        [18,4,12,0,1,0,0,5.0,0,0,0.2,0.3,25,0,3,100,0.83,3,15,2],
        [9,2,3,0,0,2,1,8.0,1,3,0.9,0.05,8,2,36,45,0.375,1,22,0],
    ]
    template_df = pd.DataFrame(sample_rows, columns=template_cols)
    csv_template = template_df.to_csv(index=False).encode()

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown('<div class="section-label">Step 1 — Download template</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download CSV template", csv_template,
                           "deliveryshield_template.csv", "text/csv",
                           use_container_width=True)
        st.markdown("""
        <div style="font-size:0.78rem;color:#555570;margin-top:0.75rem;line-height:1.6">
          Fill in one row per delivery.<br>
          <strong style="color:#888">address_type_encoded:</strong> House=1, Apt=0, Gated=2, Business=3<br>
          <strong style="color:#888">weather_encoded:</strong> Clear=0, Cloudy=1, Rain=3, HeavyRain=2, Snow=4<br>
          <strong style="color:#888">urban_density_encoded:</strong> Urban=2, Suburban=1, Rural=0
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-label">Step 2 — Upload & score</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload your CSV", type=['csv'], label_visibility="collapsed")

        if uploaded:
            raw = pd.read_csv(uploaded)
            st.markdown(f'<div style="font-size:0.8rem;color:#666680;margin-bottom:0.75rem">{len(raw)} deliveries loaded</div>', unsafe_allow_html=True)

            with st.spinner("Scoring deliveries..."):
                results = score_batch(model, raw)
                combined = pd.concat([raw.reset_index(drop=True), results], axis=1)

            # Colour the risk level column
            high = results['Risk Level'] == 'HIGH RISK'
            med  = results['Risk Level'] == 'MEDIUM RISK'

            st.dataframe(results, use_container_width=True, height=350)

            # Summary stats
            n_high = (results['Risk Level'] == 'HIGH RISK').sum()
            n_med  = (results['Risk Level'] == 'MEDIUM RISK').sum()
            n_low  = (results['Risk Level'] == 'LOW RISK').sum()
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-top:1rem">
              <div class="metric-card" style="flex:1;text-align:center">
                <div class="metric-label">High Risk</div>
                <div class="metric-val" style="color:#ef4444">{n_high}</div>
              </div>
              <div class="metric-card" style="flex:1;text-align:center">
                <div class="metric-label">Medium Risk</div>
                <div class="metric-val" style="color:#f59e0b">{n_med}</div>
              </div>
              <div class="metric-card" style="flex:1;text-align:center">
                <div class="metric-label">Low Risk</div>
                <div class="metric-val" style="color:#22c55e">{n_low}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            out_csv = combined.to_csv(index=False).encode()
            st.download_button("⬇️ Download scored results", out_csv,
                               "deliveryshield_results.csv", "text/csv",
                               use_container_width=True)
        else:
            st.markdown("""
            <div style="padding:3rem;background:rgba(255,255,255,0.02);
              border:1px dashed rgba(255,255,255,0.07);border-radius:12px;
              text-align:center;color:#444460;">
              <div style="font-size:2rem;margin-bottom:0.75rem">📄</div>
              <div style="font-size:0.88rem">Upload a CSV file to score multiple deliveries at once</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    xgb_m  = metrics.get('xgboost', {})
    lr_m   = metrics.get('logistic_regression', {})
    lgbm_m = metrics.get('lightgbm', xgb_m)
    best   = metrics.get('best_model', 'xgboost').upper()

    st.markdown(f'<div class="section-heading">Model performance &nbsp;<span style="font-size:0.72rem;color:#FF6B35;font-weight:500;background:rgba(255,107,53,0.1);padding:3px 10px;border-radius:20px;border:1px solid rgba(255,107,53,0.2)">Best: {best}</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#666680;font-size:0.83rem;margin-bottom:1.5rem">Evaluated on 10,000 held-out deliveries. Three models compared — best selected automatically by F1 score. SMOTE applied to training set to handle class imbalance.</div>', unsafe_allow_html=True)

    metric_defs = [
        ("Accuracy",  "accuracy",  "% correct predictions overall"),
        ("F1 Score",  "f1_score",  "Harmonic mean of precision & recall"),
        ("AUC-ROC",   "roc_auc",   "Ranking quality across all thresholds"),
        ("Precision", "precision", "Of flagged deliveries, % actually failed"),
        ("Recall",    "recall",    "Of actual failures, % correctly flagged"),
    ]
    model_cols = [
        ("XGBoost (Tuned)", xgb_m,  "#FF6B35"),
        ("LightGBM",        lgbm_m, "#4cc9f0"),
        ("Logistic Reg.",   lr_m,   "#888899"),
    ]

    cols = st.columns(3)
    for col, (mname, mdata, color) in zip(cols, model_cols):
        with col:
            st.markdown(f'<div class="section-label" style="color:{color}">{mname}</div>', unsafe_allow_html=True)
            for label, key, desc in metric_defs:
                val = mdata.get(key, 0)
                pct = int(val * 100)
                st.markdown(f"""
                <div class="perf-metric">
                  <div style="flex:1">
                    <div class="perf-name">{label}</div>
                    <div style="font-size:0.67rem;color:#3a3a55;margin-top:1px">{desc}</div>
                    <div class="perf-bar-track">
                      <div class="perf-bar-fill" style="width:{pct}%;background:{color}"></div>
                    </div>
                  </div>
                  <div class="perf-val" style="color:{color}">{val:.3f}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    def try_img(a, b): return a if os.path.exists(a) else b if os.path.exists(b) else None
    c1, c2, c3 = st.columns(3)
    with c1:
        p = try_img('../models/model_comparison.png','models/model_comparison.png')
        if p: st.image(p, caption='Model comparison — all metrics', use_container_width=True)
    with c2:
        p = try_img('../models/shap_summary.png','models/shap_summary.png')
        if p: st.image(p, caption='Global SHAP feature importance', use_container_width=True)
    with c3:
        p = try_img('../models/confusion_matrix.png','models/confusion_matrix.png')
        if p:
            # Re-render confusion matrix with dark theme
            import matplotlib.image as mpimg
            img = mpimg.imread(p)
            st.image(p, caption='Confusion matrix on test set', use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.markdown("""
        <div style="margin-bottom:2rem">
          <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
            text-transform:uppercase;color:#FF6B35;margin-bottom:0.75rem">The Problem</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;
            color:#fff;line-height:1.2;letter-spacing:-0.02em;margin-bottom:1rem">
            5–10% of deliveries fail.<br>That's $17.78 per attempt.<br>
            <em style="color:#FF6B35">At massive scale.</em>
          </div>
          <div style="color:#8888a0;font-size:0.88rem;line-height:1.75">
            Existing tools fix delivery failures <em>after</em> they happen — rerouting, rescheduling,
            customer service. DeliveryShield intervenes <strong style="color:#ccc">before the driver
            leaves the depot</strong>, at the dispatch stage where cost can actually be prevented.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
        for num, text in [
            ("01","Dispatch inputs delivery details — address, weather, customer instructions, driver state"),
            ("02","XGBoost / LightGBM model scores the delivery against 20 features from 50,000 historical attempts"),
            ("03","SHAP explainer surfaces top factors driving risk — not just a score, but a reason"),
            ("04","Dispatcher receives action recommendation before the van leaves — preventing cost before it occurs"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;margin-bottom:0.9rem;align-items:flex-start">
              <div style="font-family:'Space Grotesk',sans-serif;font-size:0.67rem;font-weight:700;
                color:#FF6B35;background:rgba(255,107,53,0.1);border:1px solid rgba(255,107,53,0.2);
                border-radius:6px;padding:3px 7px;flex-shrink:0;margin-top:2px">{num}</div>
              <div style="font-size:0.84rem;color:#aaaacc;line-height:1.55">{text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.5rem;padding:1.25rem;background:rgba(255,107,53,0.06);
          border:1px solid rgba(255,107,53,0.15);border-radius:12px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#FF6B35;
            text-transform:uppercase;margin-bottom:0.5rem">Novelty</div>
          <div style="font-size:0.84rem;color:#aaaacc;line-height:1.6">
            Most last-mile ML research focuses on route optimisation. DeliveryShield shifts the
            intervention point <strong style="color:#fff">upstream to dispatch</strong> — predicting
            failure before the driver leaves, where cost can actually be prevented.
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-label">Tech stack</div>', unsafe_allow_html=True)
        for name, desc, color in [
            ("XGBoost + LightGBM", "Gradient boosted classifiers with hyperparameter tuning", "#FF6B35"),
            ("SHAP",               "TreeExplainer for per-prediction explainability",          "#4cc9f0"),
            ("SMOTE",              "Synthetic minority oversampling for class balance",         "#a78bfa"),
            ("scikit-learn",       "Logistic Regression baseline + RandomizedSearchCV",        "#888899"),
            ("Streamlit",          "Web interface with live predictions and batch mode",        "#ff4b4b"),
            ("Pandas / NumPy",     "Data pipeline and feature engineering",                    "#4ade80"),
        ]:
            st.markdown(f"""
            <div class="perf-metric" style="margin-bottom:0.5rem">
              <div>
                <div style="font-size:0.85rem;color:#fff;font-weight:500">{name}</div>
                <div style="font-size:0.72rem;color:#444460;margin-top:2px">{desc}</div>
              </div>
              <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.83rem;color:#8888a0;line-height:1.75">
          Inspired by the <strong style="color:#ccc">Amazon Last Mile Routing Research
          Challenge (ALMRRC)</strong> dataset, released publicly via MIT, covering
          213,000+ delivery stops across 5 US cities. This project uses a synthetic
          dataset with the same feature schema, enriched with weather and driver fatigue signals.
        </div>
        <div style="margin-top:1rem;font-size:0.72rem;color:#333350">
          Built for Amazon ML Summer School 2025 application.
        </div>""", unsafe_allow_html=True)