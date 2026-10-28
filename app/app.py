
import os
import sys
import json
from datetime import datetime

import pandas as pd
import streamlit as st

# --------------------------------------------------
# PROJECT SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_quality

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    "dm.s", "ash.s", "cp.s", "ee.s", "ndf.s", "adf.s",
    "starch.s", "pH", "ammonia.s", "glucose.s", "fructose.s",
    "ethanol.s", "lactic.ac.s", "acetic.ac.s", "propionic.ac.s",
    "butyric.ac.s", "dm.loss",
]

INPUT_FIELDS = [
    ("dm.s", "Dry Matter (%)", 32.0, 0.1),
    ("ash.s", "Ash", 3.6, 0.1),
    ("cp.s", "Crude Protein", 7.2, 0.1),
    ("ee.s", "Ether Extract", 2.3, 0.1),
    ("ndf.s", "NDF", 40.0, 0.1),
    ("adf.s", "ADF", 21.0, 0.1),
    ("starch.s", "Starch", 30.0, 0.1),
    ("pH", "pH", 3.8, 0.1),
    ("ammonia.s", "Ammonia", 5.0, 0.1),
    ("glucose.s", "Glucose", 0.5, 0.1),
    ("fructose.s", "Fructose", 0.3, 0.1),
    ("ethanol.s", "Ethanol", 1.0, 0.1),
    ("lactic.ac.s", "Lactic Acid", 5.0, 0.1),
    ("acetic.ac.s", "Acetic Acid", 1.2, 0.1),
    ("propionic.ac.s", "Propionic Acid", 0.1, 0.01),
    ("butyric.ac.s", "Butyric Acid", 0.06, 0.01),
    ("dm.loss", "Dry Matter Loss", 8.0, 0.1),
]

MOISTURE_THRESHOLD = 68.28

st.set_page_config(
    page_title="SilageSense | Feed Quality Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
    :root {
        --bg: #0b1422;
        --panel: #142337;
        --panel-hover: #1c3047;
        --border: #2c435b;
        --green: #43c879;
        --text: #f3f7fb;
        --muted: #9fb1c5;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 90% 0%, #173a32 0%, transparent 32%),
            linear-gradient(135deg, #0b1422, #101d2e);
        color: var(--text);
    }

    .block-container {
        max-width: 1500px;
        padding: 2rem 2rem 3rem;
    }

    h1 {
        font-size: clamp(2rem, 3vw, 2.8rem) !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3, h4 {
        color: #f4f8fc !important;
        letter-spacing: -0.35px;
    }

    p, label {
        color: #d7e2ee;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a1320;
        border-right: 1px solid #24384e;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }

    /* Sidebar navigation cards */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 11px;
        padding: 10px 12px;
        transition: background 0.18s ease, border-color 0.18s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #17283a;
        border-color: #2d4359;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(100deg, #173d30, #19372f);
        border-color: #347653;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #a7f3c0 !important;
        font-weight: 700;
    }

    /* Inputs */
    div[data-baseweb="input"] > div {
        background: #172538;
        border: 1px solid #354a61;
        border-radius: 10px;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: var(--green);
        box-shadow: 0 0 0 2px #43c87925;
    }

    div[data-baseweb="input"] input {
        color: white;
    }

    /* Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #168447, #2cac62);
        border: 1px solid #43c879;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        min-height: 48px;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.12);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px #16844730;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a2b41, #132237);
        border: 1px solid #304760;
        border-radius: 16px;
        padding: 20px;
        min-height: 120px;
    }

    div[data-testid="stMetricLabel"] {
        color: #b5c6d8;
    }

    div[data-testid="stMetricValue"] {
        color: #8af0ad;
        font-weight: 750;
    }

    /* Alerts and dividers */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #2b4057;
        margin: 1.5rem 0;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #2c435b;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Download button */
    div.stDownloadButton > button {
        background: #17283b;
        border: 1px solid #3a7955;
        color: #a7f3c0;
        border-radius: 10px;
        min-height: 44px;
    }

    /* Hero */
    .hero {
        background: linear-gradient(120deg, #17382f, #17283e 72%);
        border: 1px solid #315746;
        border-radius: 20px;
        padding: 28px 30px;
        margin-bottom: 26px;
        box-shadow: 0 12px 30px #00000018;
    }

    .hero-title {
        color: #f7fbff;
        font-size: clamp(1.7rem, 3vw, 2.5rem);
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #bfd0dc;
        margin-top: 8px;
        line-height: 1.6;
    }

    .section-card {
        background: linear-gradient(145deg, #17283d, #132236);
        border: 1px solid #304760;
        border-radius: 15px;
        padding: 20px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# STATE
# --------------------------------------------------

if "analyses" not in st.session_state:
    st.session_state.analyses = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def read_metrics():
    path = os.path.join(RESULTS_DIR, "heldout_metrics.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def result_to_csv(result):
    row = {
        "timestamp": result["timestamp"],
        **result["inputs"],
        "predicted_quality_group": result["quality"],
        "cluster": result["cluster"],
        "moisture_percent": result["moisture"],
        "moisture_baseline": result["moisture_baseline"],
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig")


def show_result(result):
    quality = result["quality"]
    cluster = result["cluster"]
    moisture = result["moisture"]
    baseline = result["moisture_baseline"]

    st.subheader("📊 Sample Analysis")

    if quality == "Excellent":
        st.success("🌟 Model-assigned quality group: Excellent")
    elif quality == "Good":
        st.info("👍 Model-assigned quality group: Good")
    else:
        st.warning(f"⚠️ Model-assigned quality group: {quality}")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Quality Group", quality)

    with c2:
        st.metric("K-Means Cluster", f"Cluster {cluster}")

    with c3:
        st.metric("Calculated Moisture", f"{moisture:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### 💧 Moisture indicator")
        st.progress(min(max(moisture / 100.0, 0.0), 1.0))
        st.caption(f"Calculated as 100 − dry matter = {moisture:.2f}%")
        st.caption(
            f"Project baseline threshold: {MOISTURE_THRESHOLD:.2f}% moisture"
        )

    with right:
        st.markdown("### ⚖️ Baseline comparison")
        st.metric("Moisture-only baseline", baseline)

        if baseline == "Excellent":
            st.success("The sample meets the project's moisture-only threshold.")
        else:
            st.warning("The sample does not meet the project's moisture-only threshold.")

    st.markdown("### 💡 Advisory")

    if quality == "Excellent":
        st.write(
            "The model assigned this sample to the Excellent group. "
            "Confirm the result with laboratory measurements and farm context."
        )
    elif quality == "Good":
        st.write(
            "The model assigned this sample to the Good group. "
            "Review the full laboratory profile before making feeding decisions."
        )
    else:
        st.write(
            "The model assigned this sample to the Lower group. "
            "Consider further laboratory analysis and a qualified advisory review."
        )

    st.download_button(
        "⬇️ Download sample result (CSV)",
        data=result_to_csv(result),
        file_name=f"silage_sample_{result['timestamp'].replace(':', '-').replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"download_{result['timestamp']}",
        use_container_width=True,
    )

    st.caption(
        "Experimental research output. The model group is not a laboratory-confirmed "
        "grade and should not replace professional feed-quality testing."
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------


with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 4px 16px;">
            <div style="font-size: 25px; font-weight: 800; color: #f4f8fc;">
                🌾 SilageSense
            </div>
            <div style="color: #9fb1c5; font-size: 13px; margin-top: 6px;">
                FEED QUALITY INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("WORKSPACE")

    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🧪  Analyze Sample",
            "🧩  Cluster Explorer",
            "📈  Model Evaluation",
            "📁  Reports & History",
        ],
        index=1,
        label_visibility="collapsed",
        key="main_navigation",
    )

    # Remove the decorative emoji when checking page names.
    page = page.split("  ", 1)[-1].strip()

    st.divider()
    st.caption("MODEL SUMMARY")

    st.markdown(
        """
        <div style="
            background: #142337;
            border: 1px solid #2c435b;
            border-radius: 12px;
            padding: 14px;
        ">
            <div style="color: #a9bbcd; font-size: 12px;">
                CLUSTERING MODEL
            </div>
            <div style="color: #8af0ad; font-size: 23px; font-weight: 750;">
                K-Means · 7
            </div>
            <div style="color: #a9bbcd; font-size: 12px;">
                17 input features
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.caption("Experimental research prototype")



# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">🌾 Feed &amp; Silage Quality Advisor</div>
    <div class="hero-subtitle">
        Understand feed sample groupings through unsupervised learning.
        Explore cluster patterns and compare results with a moisture-only baseline.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":
    st.subheader("📊 Project Overview")
    st.write(
        "Welcome to SilageSense. Analyze an individual sample, inspect model "
        "evaluation results, or explore the saved clustering model."
    )

    metrics = read_metrics()

    total_analyses = len(st.session_state.analyses)
    c1, c2, c3 = st.columns(3)

    c1.metric("Analyses this session", total_analyses)
    c2.metric("K-Means clusters", "7")
    c3.metric("Input features", "17")

    st.markdown("### 🚀 Quick start")

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(
            '<div class="section-card"><h4>🧪 Analyze a sample</h4>'
            '<p>Enter laboratory measurements and view the model-assigned group.</p></div>',
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            '<div class="section-card"><h4>📈 Review evaluation</h4>'
            '<p>Inspect held-out clustering metrics and the moisture baseline.</p></div>',
            unsafe_allow_html=True,
        )

    if metrics:
        st.markdown("### Latest saved evaluation")
        a, b, c = st.columns(3)
        if "test_silhouette" in metrics:
            a.metric("Test silhouette", f"{metrics['test_silhouette']:.3f}")
        if "test_ari_vs_fqi_grades" in metrics:
            b.metric("Test ARI", f"{metrics['test_ari_vs_fqi_grades']:.3f}")
        if "test_cluster_purity" in metrics:
            c.metric("Test cluster purity", f"{metrics['test_cluster_purity']:.1%}")
    else:
        st.info(
            "No held-out evaluation file was found. Run src/evaluation.py to "
            "generate the project's evaluation results."
        )

    st.caption(
        "Reference quality groups are derived from FQI and are proxy labels, "
        "not independent laboratory-confirmed grades."
    )


# --------------------------------------------------
# ANALYZE SAMPLE PAGE
# --------------------------------------------------

elif page == "Analyze Sample":
    st.subheader("🧪 Feed Sample Parameters")
    st.info(
        "Enter measured values from a laboratory report. The default values "
        "are examples; replace them with the actual sample measurements."
    )

    input_data = {}

    group_columns = st.columns(3)

    group_titles = [
        "🌿 Nutritional composition",
        "🧪 Sugars & fermentation",
        "🧫 Organic acids & loss",
    ]

    # Keep the same input grouping used by the original app.
    input_groups = [
        INPUT_FIELDS[0:6],
        INPUT_FIELDS[6:12],
        INPUT_FIELDS[12:17],
    ]

    for column, title, fields in zip(group_columns, group_titles, input_groups):
        with column:
            st.markdown(f"#### {title}")
            for feature, label, default, step in fields:
                input_data[feature] = st.number_input(
                    label,
                    value=float(default),
                    step=float(step),
                    format="%.3f" if step < 0.1 else "%.2f",
                    key=f"input_{feature}",
                )

    st.divider()

    st.markdown("### Ready to analyze?")
    st.caption("Check the measurements above before running the model.")

    if st.button(
        "🔍 Analyze Feed Sample",
        type="primary",
        use_container_width=True,
        key="analyze_feed_sample",
    ):
        try:
            with st.spinner("Analyzing measurements..."):
                if set(input_data.keys()) != set(FEATURES):
                    st.error("The input fields do not match the model feature list.")
                    st.stop()

                if not all(pd.notna(value) for value in input_data.values()):
                    st.error("Please enter valid numeric values for all measurements.")
                    st.stop()

                if not 0 <= input_data["dm.s"] <= 100:
                    st.error("Dry Matter must be between 0 and 100%.")
                    st.stop()

                if input_data["pH"] < 0 or input_data["pH"] > 14:
                    st.error("pH must be between 0 and 14.")
                    st.stop()

                result = predict_quality(input_data)

                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "inputs": input_data.copy(),
                    "quality": str(result["quality"]),
                    "cluster": int(result["cluster"]),
                    "moisture": float(result["moisture"]),
                    "moisture_baseline": str(result["moisture_baseline"]),
                }

                st.session_state.latest_result = record
                st.session_state.analyses.append(record)

        except Exception as error:
            st.error("Prediction failed. Check the details below.")
            st.exception(error)

    if st.session_state.latest_result:
        st.divider()
        show_result(st.session_state.latest_result)


# --------------------------------------------------
# CLUSTER EXPLORER PAGE
# --------------------------------------------------

elif page == "Cluster Explorer":
    st.subheader("🧩 Cluster Explorer")
    st.write(
        "Explore the saved K-Means model's learned centers. These are centers "
        "in the model's scaled feature space, not raw laboratory measurements."
    )

    model_path = os.path.join(MODELS_DIR, "kmeans.pkl")

    try:
        import joblib

        model = joblib.load(model_path)
        centers = getattr(model, "cluster_centers_", None)

        if centers is None:
            st.warning("The saved model does not expose cluster centers.")
        else:
            center_df = pd.DataFrame(centers)
            center_df.columns = (
                FEATURES if center_df.shape[1] == len(FEATURES)
                else [f"Feature {i + 1}" for i in range(center_df.shape[1])]
            )
            center_df.index = [
                f"Cluster {i}" for i in range(len(center_df))
            ]

            st.metric("Clusters in saved model", len(center_df))
            st.markdown("### Scaled cluster-center heatmap")
            st.caption(
                "Values are standardized model-space coordinates. Positive and "
                "negative values indicate positions relative to the scaler's center."
            )
            st.dataframe(
                center_df.style.background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )

            selected_cluster = st.selectbox(
                "Inspect a cluster",
                list(range(len(center_df))),
                format_func=lambda value: f"Cluster {value}",
            )

            profile = center_df.iloc[selected_cluster].sort_values()
            profile_df = profile.rename("Scaled center").to_frame()
            st.markdown(f"### Cluster {selected_cluster} profile")
            st.bar_chart(profile_df)

    except Exception as error:
        st.error("Could not load the saved K-Means cluster centers.")
        st.exception(error)


# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------

elif page == "Model Evaluation":
    st.subheader("📈 Model Evaluation")
    st.write("Metrics loaded from the saved held-out evaluation, if available.")

    metrics = read_metrics()

    if not metrics:
        st.warning(
            "No results/heldout_metrics.json file was found. Run "
            "python src/evaluation.py from the project root to generate it."
        )
    else:
        metric_specs = [
            ("test_silhouette", "Test silhouette", ".3f"),
            ("test_ari_vs_fqi_grades", "Test ARI", ".3f"),
            ("test_cluster_purity", "Test cluster purity", ".1%"),
            ("test_cluster_balanced_accuracy", "3-class balanced accuracy", ".1%"),
            ("test_cluster_binary_balanced_accuracy", "Binary balanced accuracy", ".1%"),
            ("moisture_baseline_test_balanced_accuracy", "Moisture baseline accuracy", ".1%"),
        ]

        available = [
            (key, label, fmt)
            for key, label, fmt in metric_specs
            if key in metrics and isinstance(metrics[key], (int, float))
        ]

        for start in range(0, len(available), 3):
            columns = st.columns(3)
            for column, (key, label, fmt) in zip(
                columns, available[start:start + 3]
            ):
                value = metrics[key]
                formatted = format(value, fmt)
                column.metric(label, formatted)

        st.markdown("### ⚙️ Evaluation Setup")
        st.caption(
            "How the model was trained and evaluated on held-out samples."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Samples Evaluated",
                metrics.get("dataset_rows_evaluated", "N/A"),
            )
            st.metric(
                "Training Samples",
                metrics.get("train_samples", "N/A"),
            )

        with c2:
            st.metric(
                "Test Samples",
                metrics.get("test_samples", "N/A"),
            )
            st.metric(
                "Test Split",
                f"{metrics['test_size']:.0%}"
                if isinstance(metrics.get("test_size"), (int, float))
                else "N/A",
            )

        with c3:
            st.metric(
                "K-Means Clusters",
                metrics.get("kmeans_clusters", "N/A"),
            )
            st.metric(
                "Random State",
                metrics.get("random_state", "N/A"),
            )

        threshold = metrics.get("moisture_threshold_learned_on_train")

        if isinstance(threshold, (int, float)):
            st.info(
                f"💧 Moisture-only baseline threshold learned on training data: "
                f"**{threshold:.2f}% moisture**"
            )

        with st.expander("View technical configuration"):
            st.json(metrics)

        if (
            "moisture_baseline_test_balanced_accuracy" in metrics
            and "test_cluster_binary_balanced_accuracy" in metrics
        ):
            comparison = pd.DataFrame({
                "Method": [
                    "Cluster model (binary)",
                    "Moisture baseline (binary)",
                ],
                "Balanced accuracy": [
                    metrics["test_cluster_binary_balanced_accuracy"],
                    metrics["moisture_baseline_test_balanced_accuracy"],
                ],
            }).set_index("Method")

            st.markdown("### Binary balanced-accuracy comparison")
            st.bar_chart(comparison)

        st.info(
            "The reference groups are derived from FQI, not independent lab grades. "
            "These results describe one held-out split and do not guarantee "
            "performance on new farms or future samples."
        )


# --------------------------------------------------
# REPORTS & HISTORY PAGE
# --------------------------------------------------

elif page == "Reports & History":
    st.subheader("📁 Reports & History")
    st.write(
        "Analyses are currently stored in Streamlit session state. They are not "
        "persisted to a database and may disappear when the session resets."
    )

    analyses = st.session_state.analyses

    if not analyses:
        st.info("No analyses have been run in this session yet.")
    else:
        history_df = pd.DataFrame([
            {
                "Timestamp": item["timestamp"],
                "Quality group": item["quality"],
                "Cluster": item["cluster"],
                "Moisture (%)": round(item["moisture"], 2),
                "Moisture baseline": item["moisture_baseline"],
            }
            for item in analyses
        ])

        st.dataframe(history_df, use_container_width=True)

        history_csv = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download session history (CSV)",
            data=history_csv,
            file_name="silagesense_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear session history", type="secondary"):
            st.session_state.analyses = []
            st.session_state.latest_result = None
            st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "SilageSense • Experimental unsupervised-learning advisory • "
    "Use laboratory testing and qualified expertise for real feeding decisions."
)
