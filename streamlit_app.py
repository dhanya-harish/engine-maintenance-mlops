
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import os

# Configuration
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "dhani10/engine-maintenance-model")
MODEL_FILE = os.getenv("MODEL_FILE", "best_model.pkl")

# Expected features (match your training data exactly - should be snake_case)
EXPECTED_COLS = [
    'engine_rpm', 'lub_oil_pressure', 'fuel_pressure',
    'coolant_pressure', 'lub_oil_temp', 'coolant_temp'
]

@st.cache_resource
def load_model():
    """Load the model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILE,
            repo_type="model",
            token=os.getenv("HF_TOKEN")
        )
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Engine Condition Predictor",
        layout="centered",
        page_icon="🚧"
    )

    st.title("Predictive Maintenance — Engine Condition")
    st.markdown("Monitor engine health using real-time sensor data")
    st.caption(f"Model: {HF_MODEL_REPO}")

    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()

    if model is None:
        st.stop()

    # Input form
    st.header("Engine Sensor Readings")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            engine_rpm = st.slider("Engine RPM", 100, 2500, 1200)
            lub_oil_pressure = st.slider("Lub Oil Pressure (bar)", 0.5, 7.0, 3.0, 0.1)
            fuel_pressure = st.slider("Fuel Pressure (bar)", 0.5, 20.0, 6.0, 0.1)

        with col2:
            coolant_pressure = st.slider("Coolant Pressure (bar)", 0.5, 7.0, 2.0, 0.1)
            lub_oil_temp = st.slider("Lub Oil Temp (°C)", 70.0, 110.0, 80.0, 0.1)
            coolant_temp = st.slider("Coolant Temp (°C)", 60.0, 100.0, 75.0, 0.1)

        submitted = st.form_submit_button("Analyze Engine Condition", type="primary")

    if submitted:
        # Create input data with EXACT column names from training
        input_data = pd.DataFrame([{
            'engine_rpm': engine_rpm,
            'lub_oil_pressure': lub_oil_pressure,
            'fuel_pressure': fuel_pressure,
            'coolant_pressure': coolant_pressure,
            'lub_oil_temp': lub_oil_temp,
            'coolant_temp': coolant_temp
        }])

        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            # Display results
            st.header("Analysis Results")

            if prediction == 1:
                st.error("**FAULTY ENGINE DETECTED**")
                st.progress(probability[1])
                st.warning(f"**Risk Probability:** {probability[1]*100:.1f}%")
                st.markdown("""
                **Recommended Actions:**
                - Schedule immediate maintenance
                - Inspect lubrication system
                - Check cooling system
                """)
            else:
                st.success("**ENGINE OPERATING NORMALLY**")
                st.progress(probability[0])
                st.info(f"**Health Score:** {probability[0]*100:.1f}%")
                st.markdown("""
                **Status:** Continue routine monitoring
                **Next maintenance:** As scheduled
                """)

            # Show input data
            with st.expander("View Input Data"):
                st.dataframe(input_data)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check that the model expects the correct feature names")

if __name__ == "__main__":
    main()
