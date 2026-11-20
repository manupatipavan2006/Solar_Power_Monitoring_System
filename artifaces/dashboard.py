import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Solar Power Monitoring System",
    page_icon="☀️",
    layout="wide"
)

API_URL = "https://solar-power-api-ycln.onrender.com"

# ---------------------------------
# CUSTOM DESIGN
# ---------------------------------

st.title("☀️ Solar Power Monitoring System")

st.markdown(
    "### AI-Based Solar Power Prediction & Performance Analysis"
)

st.markdown("---")

# ---------------------------------
# CHECK API CONNECTION
# ---------------------------------

try:
    response = requests.get(API_URL + "/", timeout=5)
    api_status = response.status_code == 200
except requests.RequestException:
    api_status = False

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "API Status",
        "Connected" if api_status else "Disconnected"
    )

with col2:
    st.metric("ML Model", "Random Forest")

with col3:
    st.metric("Monitoring Plants", "2")

st.markdown("---")

# ---------------------------------
# SIDEBAR INPUTS
# ---------------------------------

st.sidebar.header("Solar Plant Configuration")

plant_id = st.sidebar.selectbox(
    "Select Solar Plant",
    [1, 2]
)

timestamp = st.sidebar.text_input(
    "Timestamp",
    datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
)

dc_power = st.sidebar.number_input(
    "DC Power",
    min_value=0.0,
    value=74854.57143
)

ambient_temperature = st.sidebar.number_input(
    "Ambient Temperature (°C)",
    value=24.06972253
)

module_temperature = st.sidebar.number_input(
    "Module Temperature (°C)",
    value=30.36452833
)

irradiation = st.sidebar.number_input(
    "Solar Irradiation",
    min_value=0.0,
    value=0.219947506
)

actual_ac_power = st.sidebar.number_input(
    "Actual AC Power",
    min_value=0.0,
    value=7359.1142858
)

# ---------------------------------
# REQUEST DATA
# ---------------------------------

input_data = {
    "timestamp": timestamp,
    "plant_id": plant_id,
    "dc_power": dc_power,
    "ambient_temperature": ambient_temperature,
    "module_temperature": module_temperature,
    "irradiation": irradiation,
    "actual_ac_power": actual_ac_power
}

# ---------------------------------
# PREDICTION SECTION
# ---------------------------------

st.header("🤖 Solar Power Prediction")

if st.button("Predict Solar Power", use_container_width=True):

    try:
        response = requests.post(
            API_URL + "/predict",
            json=input_data,
            timeout=15
        )

        response.raise_for_status()

        result = response.json()

        st.success("Prediction completed successfully!")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Predicted AC Power",
                f"{result['predicted_ac_power']:.3f}"
            )

        with c2:
            st.metric(
                "Plant ID",
                result["plant_id"]
            )

    except requests.RequestException as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------
# PERFORMANCE ANALYSIS
# ---------------------------------

st.markdown("---")

st.header("📊 Performance Deviation Analysis")

if st.button("Analyze Plant Performance", use_container_width=True):

    try:
        response = requests.post(
            API_URL + "/analyze",
            json=input_data,
            timeout=15
        )

        response.raise_for_status()

        result = response.json()

        st.subheader("Analysis Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Actual AC Power",
                f"{result['actual_ac_power']:.3f}"
            )

        with c2:
            st.metric(
                "Predicted AC Power",
                f"{result['predicted_ac_power']:.3f}"
            )

        with c3:
            st.metric(
                "Absolute Residual",
                f"{result['absolute_residual']:.3f}"
            )

        if result["status"] == "Normal":
            st.success("✅ Normal Plant Performance")
        else:
            st.error("🚨 Performance Deviation Detected")

        st.info(result["interpretation"])

        st.write(
            "Deviation Threshold:",
            result["deviation_threshold"]
        )

        comparison = pd.DataFrame({
            "Power Type": ["Actual", "Predicted"],
            "AC Power": [
                result["actual_ac_power"],
                result["predicted_ac_power"]
            ]
        })

        st.subheader("Actual vs Predicted Power")

        st.bar_chart(
            comparison.set_index("Power Type")
        )

    except requests.RequestException as e:
        st.error(f"Analysis failed: {e}")

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")

st.caption(
    "Solar Power Monitoring System | "
    "Machine Learning: Random Forest Regression | "
    "FastAPI Backend"
)