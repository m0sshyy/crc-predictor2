import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CRC Risk Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #070e1c;
    color: #e2e8f4;
}
.main .block-container {
    padding: 1.5rem 2.8rem 4rem;
    max-width: 1280px;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Top bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #0d1a30; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 0.85rem 1.6rem; margin-bottom: 1.5rem;
}
.topbar-brand { font-weight: 800; font-size: 1.05rem; color: #e2e8f4; }
.topbar-brand span { color: #38bdf8; }
.topbar-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.67rem;
    letter-spacing: 0.12em; background: rgba(56,189,248,0.1);
    color: #38bdf8; border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px; padding: 0.2rem 0.75rem;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d2146 0%, #0c3060 45%, #083d3d 100%);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 20px; padding: 2.8rem 3rem;
    margin-bottom: 2rem; position: relative; overflow: hidden;
    display: flex; align-items: center; gap: 2rem;
}
.hero::before {
    content: ''; position: absolute; top: -80px; right: 260px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-text { flex: 1; min-width: 0; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.67rem;
    letter-spacing: 0.2em; color: #38bdf8; text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero h1 {
    font-size: 1.65rem; font-weight: 800; line-height: 1.3;
    color: #ffffff; margin-bottom: 0.75rem;
}
.hero h1 em { font-style: normal; color: #67e8f9; }
.hero p { font-size: 0.87rem; color: rgba(255,255,255,0.6); line-height: 1.75; max-width: 500px; }
.hero-art { flex-shrink: 0; }

/* ── Section label ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: #38bdf8;
    font-weight: 600; display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1rem;
}
.sec-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(to right, rgba(56,189,248,0.3), transparent);
}

/* ── Dark cards ── */
.dcard {
    background: #0d1a30; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.4rem 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 1rem;
}
.dcard-title {
    font-size: 0.75rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 1rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Streamlit widget overrides ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-size: 0.8rem !important; font-weight: 500 !important; color: #94a3b8 !important;
}
div[data-testid="stNumberInput"] input {
    background: #111f38 !important; border: 1.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 9px !important; color: #e2e8f4 !important;
    font-family: 'Sora', sans-serif !important;
}
div[data-baseweb="select"] > div {
    background: #111f38 !important; border: 1.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 9px !important;
}
div[data-baseweb="select"] span { color: #e2e8f4 !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #0891b2) !important;
    color: #ffffff !important; font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.8rem 2rem !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(14,165,233,0.5) !important;
}

/* ── Result boxes ── */
.result-box { border-radius: 16px; padding: 1.8rem; text-align: center; }
.result-high     { background: linear-gradient(135deg,#1c0a0a,#2d1010); border: 1.5px solid #7f1d1d; }
.result-moderate { background: linear-gradient(135deg,#1c1507,#2d2010); border: 1.5px solid #78350f; }
.result-low      { background: linear-gradient(135deg,#071c10,#0a2d1a); border: 1.5px solid #14532d; }
.risk-badge {
    display: inline-block; border-radius: 30px; padding: 0.25rem 0.9rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.6rem;
}
.badge-high     { background: rgba(239,68,68,0.2);  color: #fca5a5; border: 1px solid #7f1d1d; }
.badge-moderate { background: rgba(245,158,11,0.2); color: #fcd34d; border: 1px solid #78350f; }
.badge-low      { background: rgba(34,197,94,0.2);  color: #86efac; border: 1px solid #14532d; }
.risk-num { font-size: 3rem; font-weight: 800; line-height: 1; margin-bottom: 0.3rem; }
.risk-num-high     { color: #ef4444; }
.risk-num-moderate { color: #f59e0b; }
.risk-num-low      { color: #22c55e; }
.risk-conf { font-size: 0.82rem; color: #64748b; }

/* ── Prob bars ── */
.prob-row { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.55rem; }
.prob-lbl { font-size:0.75rem; font-weight:600; color:#94a3b8; width:68px; }
.prob-track { flex:1; height:9px; background:rgba(255,255,255,0.05); border-radius:5px; overflow:hidden; }
.prob-fill  { height:100%; border-radius:5px; }
.prob-pct   { font-family:'JetBrains Mono',monospace; font-size:0.72rem; width:38px; text-align:right; }

/* ── Pills ── */
.pill { display:inline-block; border-radius:20px; padding:0.22rem 0.65rem;
        font-size:0.72rem; font-family:'JetBrains Mono',monospace; margin:0.18rem; }
.pill-r { background:rgba(239,68,68,0.12);  border:1px solid rgba(239,68,68,0.3);  color:#fca5a5; }
.pill-g { background:rgba(34,197,94,0.10);  border:1px solid rgba(34,197,94,0.3);  color:#86efac; }

/* ── SHAP badge ── */
.shap-badge {
    display:inline-block; background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.25);
    border-radius:6px; padding:0.15rem 0.5rem;
    font-family:'JetBrains Mono',monospace; font-size:0.62rem;
    color:#38bdf8; letter-spacing:0.08em; margin-left:0.5rem; vertical-align:middle;
}

/* ── Clinical card ── */
.clin-card {
    background:#0d1a30; border-left:4px solid #0ea5e9;
    border-radius:0 12px 12px 0; padding:1.3rem 1.5rem;
}

/* ── Disclaimer ── */
.disclaimer {
    background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.2);
    border-radius:12px; padding:1rem 1.2rem;
    font-size:0.78rem; color:#64748b; line-height:1.6; margin-top:2rem;
}
.disclaimer strong { color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('crc_xgboost_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Ensure `crc_xgboost_model.pkl` is in the same directory.")
    st.stop()

FEATURE_COLS = [
    'Age', 'Gender', 'Family_History', 'Smoking_History', 'Alcohol_Consumption',
    'Diabetes', 'Inflammatory_Bowel_Disease', 'Genetic_Mutation',
    'Obesity_Risk_Level', 'Diet_Risk_Level', 'Physical_Inactivity_Risk',
    'Screening_History_Irregular', 'Screening_History_Never', 'Screening_History_Regular',
    'Genetic_Age_Interaction', 'Medical_Comorbidity_Score', 'Lifestyle_Index'
]
FEATURE_LABELS = {
    'Age':'Age','Gender':'Gender','Family_History':'Family History',
    'Smoking_History':'Smoking','Alcohol_Consumption':'Alcohol Use',
    'Diabetes':'Diabetes','Inflammatory_Bowel_Disease':'IBD',
    'Genetic_Mutation':'Genetic Mutation','Obesity_Risk_Level':'Obesity Level',
    'Diet_Risk_Level':'Diet Risk','Physical_Inactivity_Risk':'Physical Inactivity',
    'Screening_History_Irregular':'Irregular Screening',
    'Screening_History_Never':'Never Screened',
    'Screening_History_Regular':'Regular Screening',
    'Genetic_Age_Interaction':'Genetic × Age',
    'Medical_Comorbidity_Score':'Comorbidity Score',
    'Lifestyle_Index':'Lifestyle Index'
}

# ─────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">🔬 <span>CRC</span>&nbsp;Risk Predictor</div>
  <div class="topbar-tag">XGBoost · SHAP · v1.0</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🔬 Clinical Decision Support · Academic Research Tool</div>
  <h1>AI-Based Predictive Model for <em>Early Detection</em><br>of Colorectal Cancer Risk Factors</h1>
  <p>Powered by XGBoost with SHAP explainability — assess individual CRC risk from clinical, genetic, and lifestyle parameters.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">Patient Data Input</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown('<div class="dcard"><div class="dcard-title">🧍 Demographics</div>', unsafe_allow_html=True)
    age       = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
    gender    = st.selectbox("Gender", ["Male", "Female"])
    screening = st.selectbox("Screening History", ["Regular", "Irregular", "Never"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="dcard"><div class="dcard-title">🧬 Clinical & Genetic</div>', unsafe_allow_html=True)
    family_history   = st.selectbox("Family History of CRC", ["No", "Yes"])
    genetic_mutation = st.selectbox("Genetic Mutation (MLH1 / BRCA)", ["No", "Yes"])
    diabetes         = st.selectbox("Diabetes", ["No", "Yes"])
    ibd              = st.selectbox("Inflammatory Bowel Disease (IBD)", ["No", "Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="dcard"><div class="dcard-title">🥗 Lifestyle Factors</div>', unsafe_allow_html=True)
    smoking   = st.selectbox("Smoking History", ["No", "Yes"])
    alcohol   = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    obesity   = st.selectbox("Obesity Level", ["Normal", "Overweight", "Obese"])
    diet_risk = st.selectbox("Diet Risk Level", ["Low", "Moderate", "High"])
    activity  = st.selectbox("Physical Activity Level", ["High", "Moderate", "Low"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Analyse CRC Risk Profile", use_container_width=True)

# ─────────────────────────────────────────────
#  PREDICTION + SHAP
# ─────────────────────────────────────────────
if predict_btn:
    gen_val  = 1 if gender == "Male" else 0
    fam_val  = 1 if family_history == "Yes" else 0
    smok_val = 1 if smoking == "Yes" else 0
    alc_val  = 1 if alcohol == "Yes" else 0
    diab_val = 1 if diabetes == "Yes" else 0
    ibd_val  = 1 if ibd == "Yes" else 0
    gene_val = 1 if genetic_mutation == "Yes" else 0

    obesity_map  = {"Normal":0,"Overweight":1,"Obese":2}
    diet_map     = {"Low":0,"Moderate":1,"High":2}
    activity_map = {"High":0,"Moderate":1,"Low":2}

    genetic_age_interaction = gene_val * age
    medical_score = ibd_val + diab_val
    lifestyle_idx = smok_val + alc_val + diet_map[diet_risk] + obesity_map[obesity] + activity_map[activity]

    input_df = pd.DataFrame([{
        'Age':age,'Gender':gen_val,'Family_History':fam_val,
        'Smoking_History':smok_val,'Alcohol_Consumption':alc_val,
        'Diabetes':diab_val,'Inflammatory_Bowel_Disease':ibd_val,
        'Genetic_Mutation':gene_val,'Obesity_Risk_Level':obesity_map[obesity],
        'Diet_Risk_Level':diet_map[diet_risk],'Physical_Inactivity_Risk':activity_map[activity],
        'Screening_History_Irregular':1 if screening=="Irregular" else 0,
        'Screening_History_Never':    1 if screening=="Never"     else 0,
        'Screening_History_Regular':  1 if screening=="Regular"   else 0,
        'Genetic_Age_Interaction':genetic_age_interaction,
        'Medical_Comorbidity_Score':medical_score,
        'Lifestyle_Index':lifestyle_idx
    }])[FEATURE_COLS]

    prediction    = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = {0:"Low", 1:"Moderate", 2:"High"}
    result  = classes[int(prediction)]

    # ── SHAP ──
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        pred_class  = int(prediction)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[pred_class]).flatten()
        else:
            sv_arr = np.array(shap_values)
            if   sv_arr.ndim == 3: sv = sv_arr[0, :, pred_class]
            elif sv_arr.ndim == 2: sv = sv_arr[0]
            else:                  sv = sv_arr.flatten()
        sv = np.array(sv).flatten()
        if sv.shape[0] != len(FEATURE_COLS):
            raise ValueError(f"Shape mismatch: {sv.shape[0]} vs {len(FEATURE_COLS)}")
        shap_ok = True
    except Exception as e:
        shap_ok  = False
        shap_err = str(e)

    st.divider()
    st.markdown('<div class="sec-label">Risk Assessment Result</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1.4])

    with r1:
        css  = result.lower()
        icon = "🔴" if result=="High" else ("🟡" if result=="Moderate" else "🟢")
        conf = np.max(probabilities)*100
        st.markdown(f"""
        <div class="result-box result-{css}">
          <div class="risk-badge badge-{css}">Predicted Risk Level</div><br>
          <div class="risk-num risk-num-{css}">{icon} {result}</div>
          <div class="risk-conf" style="margin-top:0.4rem">
            Model confidence: <strong style="color:#e2e8f4">{conf:.1f}%</strong>
          </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="dcard">', unsafe_allow_html=True)
        st.markdown("**Class Probability Distribution**")
        clr = {"Low":"#22c55e","Moderate":"#f59e0b","High":"#ef4444"}
        for label, prob in zip(["Low","Moderate","High"], probabilities):
            pct = prob*100
            st.markdown(f"""
            <div class="prob-row">
              <span class="prob-lbl">{label}</span>
              <div class="prob-track">
                <div class="prob-fill" style="width:{pct:.1f}%;background:{clr[label]};"></div>
              </div>
              <span class="prob-pct" style="color:{clr[label]}">{pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SHAP Chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="sec-label">XAI Explanation'
        f'<span class="shap-badge">SHAP · SHapley Additive exPlanations</span></div>',
        unsafe_allow_html=True)

    if shap_ok:
        feat_labels = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
        shap_s = pd.Series(sv, index=feat_labels)
        shap_sorted = shap_s.reindex(shap_s.abs().sort_values(ascending=True).index).iloc[-12:]

        fig, ax = plt.subplots(figsize=(10, 5.2))
        fig.patch.set_facecolor('#0d1a30')
        ax.set_facecolor('#0d1a30')

        bar_clrs = ['#ef4444' if v > 0 else '#22c55e' for v in shap_sorted.values]
        bars = ax.barh(shap_sorted.index, shap_sorted.values,
                       color=bar_clrs, height=0.58, edgecolor='none')

        for bar, val in zip(bars, shap_sorted.values):
            x  = val + (0.002 if val >= 0 else -0.002)
            ha = 'left' if val >= 0 else 'right'
            ax.text(x, bar.get_y()+bar.get_height()/2,
                    f'{val:+.3f}', va='center', ha=ha,
                    fontsize=8, color='#cbd5e1', fontfamily='monospace')

        ax.axvline(0, color=(1,1,1,0.15), linewidth=0.9, linestyle='--')
        ax.set_xlabel('SHAP Value  (impact on predicted risk class)',
                      color='#64748b', fontsize=9, labelpad=8)
        ax.tick_params(axis='x', colors='#64748b', labelsize=8)
        ax.tick_params(axis='y', colors='#cbd5e1', labelsize=9)
        for sp in ['top','right']:
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color((1,1,1,0.08))
        ax.spines['left'].set_color((1,1,1,0.08))

        p1 = mpatches.Patch(color='#ef4444', label='↑ Increases risk')
        p2 = mpatches.Patch(color='#22c55e', label='↓ Decreases risk')
        leg = ax.legend(handles=[p1, p2], loc='lower right',
                        framealpha=0, fontsize=8)
        for t in leg.get_texts(): t.set_color('#94a3b8')

        ax.set_title(f'Feature Contributions → Predicted: {result} Risk',
                     color='#e2e8f4', fontsize=11, fontweight='700', pad=12, loc='left')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Pills ──
        st.markdown("<br>", unsafe_allow_html=True)
        p1c, p2c = st.columns(2)
        risk_f = shap_s[shap_s > 0].sort_values(ascending=False)
        prot_f = shap_s[shap_s < 0].sort_values(ascending=True)

        with p1c:
            st.markdown('<div class="dcard">', unsafe_allow_html=True)
            st.markdown("**🔴 Risk-Increasing Factors**")
            if len(risk_f):
                pills = "".join([f'<span class="pill pill-r">{f} &nbsp;+{v:.3f}</span>'
                                 for f, v in risk_f.head(6).items()])
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown(
                    f"<br><small style='color:#64748b'>These push the model toward "
                    f"<strong style='color:#ef4444'>{result}</strong> risk.</small>",
                    unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#64748b'>None detected.</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with p2c:
            st.markdown('<div class="dcard">', unsafe_allow_html=True)
            st.markdown("**🟢 Risk-Protective Factors**")
            if len(prot_f):
                pills = "".join([f'<span class="pill pill-g">{f} &nbsp;{v:.3f}</span>'
                                 for f, v in prot_f.head(6).items()])
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown(
                    "<br><small style='color:#64748b'>These lower the predicted risk level.</small>",
                    unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#64748b'>None detected.</small>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning(f"SHAP could not be generated. Error: {shap_err}")

    # ── Clinical Interpretation ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Clinical Interpretation</div>', unsafe_allow_html=True)

    interp = {
        "High": {
            "summary": "This profile is associated with <strong style='color:#ef4444'>HIGH</strong> colorectal cancer risk.",
            "actions": [
                "Immediate referral for colonoscopy is recommended.",
                "Genetic counselling if mutation markers are present.",
                "Lifestyle intervention: smoking cessation, diet, weight management.",
                "Increase screening frequency to annual or biennial.",
            ]
        },
        "Moderate": {
            "summary": "This profile is associated with <strong style='color:#f59e0b'>MODERATE</strong> colorectal cancer risk.",
            "actions": [
                "Schedule colonoscopy within 1–3 years.",
                "Lifestyle modifications: diet, exercise, alcohol reduction.",
                "Monitor comorbidities (diabetes, IBD) closely.",
                "Educate patient on early warning symptoms.",
            ]
        },
        "Low": {
            "summary": "This profile is associated with <strong style='color:#22c55e'>LOW</strong> colorectal cancer risk.",
            "actions": [
                "Continue regular screening per national guidelines.",
                "Maintain healthy lifestyle: balanced diet and physical activity.",
                "Reassess annually or if new risk factors emerge.",
                "Patient education on CRC prevention.",
            ]
        }
    }
    info  = interp[result]
    items = "".join([f"<li style='margin-bottom:0.4rem;color:#94a3b8'>{a}</li>"
                     for a in info['actions']])
    st.markdown(f"""
    <div class="clin-card">
      <p style="font-size:0.92rem;margin-bottom:0.9rem">{info['summary']}</p>
      <p style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#38bdf8;
         letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem">Recommended Actions</p>
      <ul style="padding-left:1.2rem;font-size:0.86rem">{items}</ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      <strong>⚠️ Medical Disclaimer:</strong> This tool is developed for
      <strong>academic and research purposes only</strong>. It is not a substitute for
      professional medical diagnosis, advice, or treatment. All clinical decisions must be
      made by a qualified healthcare professional. SHAP values represent model feature
      contributions and do not imply direct medical causality.
    </div>
    """, unsafe_allow_html=True)
