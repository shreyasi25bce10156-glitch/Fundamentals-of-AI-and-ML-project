"""
Student Performance Predictor — Streamlit Web Interface
=======================================================
Loads the trained model artefacts and presents an interactive form
for real-time prediction of student Pass / Fail outcome.

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .hero-sub {
        text-align: center;
        color: #999;
        font-size: 0.95rem;
        margin-top: -0.3rem;
        margin-bottom: 1.8rem;
    }
    .result-pass {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
    }
    .result-fail {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
    }
    .info-card {
        background: #f8f9fa;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .sidebar-section {
        background: #fafbfc;
        border-radius: 0.5rem;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #667eea;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── load artefacts (cached) ──────────────────────────────────
@st.cache_resource
def load_artefacts():
    try:
        return joblib.load("model_artifacts.pkl")
    except FileNotFoundError:
        st.error(
            "`model_artifacts.pkl` not found.\n\n"
            "Please run the training pipeline first:\n"
            "```\n"
            "python generate_dataset.py\n"
            "python train_model.py\n"
            "```"
        )
        st.stop()


art = load_artefacts()
model = art["model"]
scaler = art["scaler"]
feature_names = art["feature_names"]
feat_imp = art["feature_importances"]
class_map = art["class_mapping"]
model_name = art["model_name"]
metrics = art["metrics"]
needs_scaling = art["needs_scaling"]

# inverse class mapping  → {"Pass": 1, "Fail": 0}
inv_map = {v: k for k, v in class_map.items()}


# ── sidebar inputs ───────────────────────────────────────────
st.sidebar.markdown("## 📋 Student Profile")
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-section">📚 Academic</div>', unsafe_allow_html=True)
study_hours   = st.sidebar.slider("Study Hours / Day",       0.5, 10.0, 3.0, 0.5)
attendance    = st.sidebar.slider("Attendance (%)",          30.0, 100.0, 75.0, 1.0)
prev_gpa      = st.sidebar.slider("Previous GPA (out of 10)", 0.0, 10.0, 7.0, 0.1)
assign_rate   = st.sidebar.slider("Assignment Completion (%)",0.0, 100.0, 70.0, 1.0)
absences      = st.sidebar.slider("Number of Absences",       0, 30, 5)

st.sidebar.markdown('<div class="sidebar-section">🌙 Lifestyle</div>', unsafe_allow_html=True)
sleep_hours   = st.sidebar.slider("Sleep Hours / Day",        3.0, 10.0, 7.0, 0.5)
internet_hrs  = st.sidebar.slider("Non-Academic Internet (hrs)", 0.0, 8.0, 2.0, 0.5)

st.sidebar.markdown('<div class="sidebar-section">👥 Background</div>', unsafe_allow_html=True)
extra_sel     = st.sidebar.selectbox("Extracurricular Activities", ["No", "Yes"], index=0)
parent_sel    = st.sidebar.selectbox(
    "Parental Education",
    ["High School", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"],
    index=1,
)

extra_val  = 1 if extra_sel == "Yes" else 0
parent_val = {"High School": 1, "Bachelor's Degree": 2,
              "Master's Degree": 3, "Doctoral Degree": 4}[parent_sel]

st.sidebar.markdown("---")
predict = st.sidebar.button("🔮  Predict Performance", type="primary", use_container_width=True)


# ── main area — header ───────────────────────────────────────
st.markdown('<div class="hero-title">Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">BTech AIML Introductory Course Project &nbsp;|&nbsp; Binary Classification</div>', unsafe_allow_html=True)

# input summary cards
st.markdown("### 📊 Input Summary")
c1, c2, c3 = st.columns(3)
labels_short = [
    "Study Hrs/Day", "Attendance %", "Previous GPA",
    "Assignment %", "Absences", "Sleep Hrs/Day",
    "Internet Hrs", "Extracurricular", "Parental Edu.",
]
vals_short = [
    study_hours, attendance, prev_gpa,
    assign_rate, absences, sleep_hours,
    internet_hrs, extra_sel, parent_sel,
]
for idx, (lbl, val) in enumerate(zip(labels_short, vals_short)):
    with [c1, c2, c3][idx % 3]:
        st.metric(label=lbl, value=val)

st.markdown("---")


# ── prediction block ─────────────────────────────────────────
if predict:
    # build feature vector in exact column order
    row = np.array([[study_hours, attendance, prev_gpa, sleep_hours,
                     extra_val, internet_hrs, parent_val,
                     absences, assign_rate]])
    row_df = pd.DataFrame(row, columns=feature_names)

    X_in = scaler.transform(row_df) if needs_scaling else row_df
    pred_label_idx = int(model.predict(X_in)[0])
    probs = model.predict_proba(X_in)[0]
    pred_label = class_map[pred_label_idx]
    confidence = probs[pred_label_idx] * 100
    pass_prob = probs[inv_map["Pass"]] * 100
    fail_prob = probs[inv_map["Fail"]] * 100

    # result box
    if pred_label == "Pass":
        st.markdown(f"""
        <div class="result-pass">
            <div style="font-size:3rem">✅</div>
            <div style="font-size:2rem;font-weight:800;color:#155724">PASS</div>
            <div style="color:#155724;font-size:1.1rem">
                The model predicts this student will <b>pass</b> — {confidence:.1f}% confidence
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-fail">
            <div style="font-size:3rem">❌</div>
            <div style="font-size:2rem;font-weight:800;color:#721c24">FAIL</div>
            <div style="color:#721c24;font-size:1.1rem">
                The model predicts this student is <b>at risk of failing</b> — {confidence:.1f}% confidence
            </div>
        </div>""", unsafe_allow_html=True)

    # probability bars
    st.markdown("### 📈 Prediction Probabilities")
    pc, fc = st.columns(2)
    with pc:
        st.progress(pass_prob / 100)
        st.metric("Pass Probability", f"{pass_prob:.1f}%", delta_color="normal")
    with fc:
        st.progress(fail_prob / 100)
        st.metric("Fail Probability", f"{fail_prob:.1f}%", delta_color="normal")

    st.markdown("---")

    # feature importance chart
    st.markdown("### 🔍 Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(feat_imp)
    normed = feat_imp[order] / feat_imp.max() if feat_imp.max() > 0 else feat_imp[order]
    ax.barh(range(len(feature_names)), feat_imp[order],
            color=plt.cm.RdYlGn(normed), edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i].replace("_", " ").title() for i in order])
    ax.set_xlabel("Relative Importance", fontweight="bold")
    ax.set_title(f"What Matters Most — {model_name}", fontweight="bold", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    for patch, val in zip(ax.patches, feat_imp[order]):
        if val > 0.005:
            ax.text(val + feat_imp.max() * 0.01,
                    patch.get_y() + patch.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # interpretive note
    top_idx = np.argsort(feat_imp)[::-1][0]
    st.info(
        f"💡 **Key insight:** `{feature_names[top_idx].replace('_', ' ').title()}` "
        f"is the strongest predictor for this model (importance = {feat_imp[top_idx]:.3f})."
    )

else:
    # ── landing page (no prediction yet) ─────────────────────
    st.markdown("### 🤖 Model Information")
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, title, val in [
        (mc1, "Model", model_name),
        (mc2, "Accuracy", f"{metrics['Accuracy']:.1%}"),
        (mc3, "F1 Score", f"{metrics['F1 Score']:.1%}"),
        (mc4, "CV Accuracy", f"{metrics['CV Mean']:.1%} (±{metrics['CV Std']:.1%})"),
    ]:
        with col:
            st.markdown(f'<div class="info-card"><b>{title}</b><br>{val}</div>',
                        unsafe_allow_html=True)

    st.markdown("---")

    # feature importance preview
    st.markdown("### 🔍 Feature Importance (Preview)")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(feat_imp)
    normed = feat_imp[order] / feat_imp.max() if feat_imp.max() > 0 else feat_imp[order]
    ax.barh(range(len(feature_names)), feat_imp[order],
            color=plt.cm.RdYlGn(normed), edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i].replace("_", " ").title() for i in order])
    ax.set_xlabel("Relative Importance", fontweight="bold")
    ax.set_title(f"Feature Importance — {model_name}", fontweight="bold", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
    This application predicts whether a student will **Pass** or **Fail** based on
    9 academic, lifestyle, and background features.

    **ML Pipeline:**
    1. **Input** — Student attributes entered via the sidebar form
    2. **Scaling** — Features are standardised (zero mean, unit variance) using the
       same `StandardScaler` fitted during training
    3. **Prediction** — The trained classifier outputs a class label and per-class probabilities
    4. **Visualisation** — Results, confidence bars, and feature importance are rendered

    **Features (9):**
    | Feature | Description |
    |---|---|
    | `study_hours_per_day` | Average daily self-study hours |
    | `attendance_percentage` | Class attendance rate (0–100 %) |
    | `previous_gpa` | GPA from prior semester (0–10) |
    | `sleep_hours_per_day` | Average nightly sleep |
    | `extracurricular_activities` | Participates in extracurriculars (0 / 1) |
    | `internet_usage_hours` | Non-academic screen time per day |
    | `parental_education_level` | Highest parental education (1–4) |
    | `number_of_absences` | Total absences in the semester |
    | `assignment_completion_rate` | % of assignments submitted |
    """)

    st.info("👆  Adjust the sliders in the sidebar, then click **Predict Performance** to see the result.")


# ── footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#bbb;font-size:0.82rem;">
    BTech AIML Introductory Course Project &nbsp;·&nbsp; Student Performance Prediction<br>
    Built with Streamlit · Scikit-learn · Pandas · Matplotlib
</div>
""", unsafe_allow_html=True)
