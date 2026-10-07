import streamlit as st
import sys
import os

# Allow importing from src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.predict import predict_quality


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Feed & Silage Quality Advisor",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌾 Feed & Silage Quality Advisor")

st.markdown(
    """
    **Unsupervised Learning based Feed & Silage Quality Grouping**

    Enter the measured properties of a feed/silage sample below.
    The model will assign the sample to a learned quality group.
    """
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("🧪 Feed Sample Parameters")

st.info(
    "Enter the laboratory measurements of the feed/silage sample. "
    "The values below are example values and can be changed."
)


col1, col2, col3 = st.columns(3)


with col1:
    dm_s = st.number_input(
        "Dry Matter (dm.s) %",
        value=32.0,
        step=0.1
    )

    ash_s = st.number_input(
        "Ash (ash.s)",
        value=3.6,
        step=0.1
    )

    cp_s = st.number_input(
        "Crude Protein (cp.s)",
        value=7.2,
        step=0.1
    )

    ee_s = st.number_input(
        "Ether Extract (ee.s)",
        value=2.3,
        step=0.1
    )

    ndf_s = st.number_input(
        "NDF (ndf.s)",
        value=40.0,
        step=0.1
    )

    adf_s = st.number_input(
        "ADF (adf.s)",
        value=21.0,
        step=0.1
    )


with col2:
    starch_s = st.number_input(
        "Starch (starch.s)",
        value=30.0,
        step=0.1
    )

    ph = st.number_input(
        "pH",
        value=3.8,
        step=0.1
    )

    ammonia_s = st.number_input(
        "Ammonia (ammonia.s)",
        value=5.0,
        step=0.1
    )

    glucose_s = st.number_input(
        "Glucose (glucose.s)",
        value=0.5,
        step=0.1
    )

    fructose_s = st.number_input(
        "Fructose (fructose.s)",
        value=0.3,
        step=0.1
    )

    ethanol_s = st.number_input(
        "Ethanol (ethanol.s)",
        value=1.0,
        step=0.1
    )


with col3:
    lactic_ac_s = st.number_input(
        "Lactic Acid (lactic.ac.s)",
        value=5.0,
        step=0.1
    )

    acetic_ac_s = st.number_input(
        "Acetic Acid (acetic.ac.s)",
        value=1.2,
        step=0.1
    )

    propionic_ac_s = st.number_input(
        "Propionic Acid (propionic.ac.s)",
        value=0.1,
        step=0.01
    )

    butyric_ac_s = st.number_input(
        "Butyric Acid (butyric.ac.s)",
        value=0.06,
        step=0.01
    )

    dm_loss = st.number_input(
        "Dry Matter Loss (dm.loss)",
        value=8.0,
        step=0.1
    )


st.divider()


# --------------------------------------------------
# ANALYSIS BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Analyze Feed Sample",
    type="primary",
    use_container_width=True
):

    input_data = {
        "dm.s": dm_s,
        "ash.s": ash_s,
        "cp.s": cp_s,
        "ee.s": ee_s,
        "ndf.s": ndf_s,
        "adf.s": adf_s,
        "starch.s": starch_s,
        "pH": ph,
        "ammonia.s": ammonia_s,
        "glucose.s": glucose_s,
        "fructose.s": fructose_s,
        "ethanol.s": ethanol_s,
        "lactic.ac.s": lactic_ac_s,
        "acetic.ac.s": acetic_ac_s,
        "propionic.ac.s": propionic_ac_s,
        "butyric.ac.s": butyric_ac_s,
        "dm.loss": dm_loss
    }

    try:

        result = predict_quality(input_data)

        quality = result["quality"]
        cluster = result["cluster"]
        moisture = result["moisture"]
        baseline = result["moisture_baseline"]


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.subheader("📊 Analysis Result")


        if quality == "Excellent":

            st.success(
                f"### 🌟 Quality Group: {quality}"
            )

            advisory = (
                "The sample belongs to the Excellent quality group "
                "according to the learned clustering model."
            )

        elif quality == "Good":

            st.info(
                f"### 👍 Quality Group: {quality}"
            )

            advisory = (
                "The sample belongs to the Good quality group. "
                "The silage characteristics show a moderate quality profile."
            )

        else:

            st.warning(
                f"### ⚠️ Quality Group: {quality}"
            )

            advisory = (
                "The sample belongs to the Lower quality group. "
                "Further laboratory analysis and management review are recommended."
            )


        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Quality Group",
                quality
            )

        with c2:
            st.metric(
                "Cluster",
                f"Cluster {cluster}"
            )

        with c3:
            st.metric(
                "Moisture",
                f"{moisture:.2f}%"
            )


        st.divider()


        # --------------------------------------------------
        # ADVISORY
        # --------------------------------------------------

        st.subheader("💡 Advisory")

        st.write(advisory)


        # --------------------------------------------------
        # MOISTURE BASELINE
        # --------------------------------------------------

        st.subheader("💧 Moisture Baseline")

        if baseline == "Excellent":

            st.success(
                "Moisture-only baseline: Excellent"
            )

        else:

            st.warning(
                "Moisture-only baseline: Not Excellent"
            )

        st.caption(
            "The moisture baseline uses the project-specific "
            "data-derived threshold of 68.28% moisture."
        )


        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.divider()

        st.caption(
            "⚠️ This is an experimental unsupervised-learning advisory "
            "system. It is intended for research and demonstration and "
            "should not replace laboratory feed-quality testing."
        )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)