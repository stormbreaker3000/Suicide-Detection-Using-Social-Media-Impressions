"""
Streamlit App for Suicide Sentiment Analysis

Models used:
1. Logistic Regression + TF-IDF
2. BiLSTM + Tokenizer

Final Verdict:
- suicide
- non-suicide
- borderline

Author: Subham Bagchi
"""

import os
import logging
import pickle
from typing import Optional

import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
MODELS_DIR = "Models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "Vectorizer_model.pkl")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "LR_model.pkl")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "Tokenizer_model.pkl")
BILSTM_MODEL_PATH = os.path.join(MODELS_DIR, "BILSTM_model.pkl")

SUICIDE = "suicide"
NON_SUICIDE = "non-suicide"

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def load_pickle_model(path: str):
    """Safely load a pickle model."""
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        print(f"Model not found: {path}")
        return None
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None


def preprocess_text(text: str) -> str:
    """Lowercase and strip text."""
    return text.lower().strip()


def predict_lr(model, vectorizer, text: str) -> Optional[str]:
    """Predict using Logistic Regression + TF-IDF."""
    try:
        features = vectorizer.transform([text])
        pred = model.predict(features)[0]
        return pred
    except Exception as e:
        logging.error(f"LR prediction failed: {e}")
        return None


def predict_bilstm(model, tokenizer, text: str) -> Optional[str]:
    """Predict using BiLSTM model."""
    try:
        max_len = 100
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding="post")
        prob = model.predict(padded)[0][0]
        return SUICIDE if prob > 0.5 else NON_SUICIDE
    except Exception as e:
        logging.error(f"BiLSTM prediction failed: {e}")
        return None


def compare_models(text, lr_model, vectorizer, bilstm_model, tokenizer) -> dict:
    """
    Compare both models and return final verdict.
    """
    lr_pred = predict_lr(lr_model, vectorizer, text) if lr_model and vectorizer else None
    bilstm_pred = predict_bilstm(bilstm_model, tokenizer, text) if bilstm_model and tokenizer else None

    if lr_pred and bilstm_pred:
        verdict = lr_pred if lr_pred == bilstm_pred else "borderline"
    else:
        verdict = "unknown"

    color_map = {
        "suicide": "red",
        "non-suicide": "green",
        "borderline": "yellow",
        "unknown": "gray"
    }
    print(lr_pred,bilstm_pred)
    return {
        "lr": lr_pred,
        "lr_color": color_map[lr_pred],
        "bilstm": bilstm_pred,
        "bilstm_color": color_map[bilstm_pred],
        "verdict": verdict,
        "verdict_color": color_map[verdict]
    }

# --------------------------------------------------
# Streamlit Styling
# --------------------------------------------------
st.set_page_config(page_title="Suicide Detection App", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    color: #2C3E50;
}       
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.result-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    font-size: 18px;
    font-weight: bold;
}
.result-table td {
    padding: 14px;
    text-align: center;
    border-radius: 10px;
}
.label-cell {
    background-color: #F4F6F7;
    color: #2C3E50;
    width: 40%;
}
.green { background-color: #D5F5E3; color: #145A32; }
.red { background-color: #FADBD8; color: #922B21; }
.yellow { background-color: #FCF3CF; color: #7D6608; }
.gray { background-color: #EBEDEF; color: #566573; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
vectorizer, lr_model, tokenizer, bilstm_model = None, None, None, None
vectorizer = load_pickle_model(VECTORIZER_PATH) if vectorizer is None else vectorizer
lr_model = load_pickle_model(LR_MODEL_PATH)  if lr_model is None else lr_model
tokenizer = load_pickle_model(TOKENIZER_PATH)  if tokenizer is None else tokenizer
bilstm_model = load_pickle_model(BILSTM_MODEL_PATH)  if bilstm_model is None else bilstm_model

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# UI Layout
# --------------------------------------------------
st.markdown("<div class='main-title'>🧠 Suicide Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Dual-Model Prediction with Final Verdict</div>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([2, 1])

# --------------------------------------------------
# Input Section
# --------------------------------------------------
with col1:
    user_text = st.text_area("💬 Enter a comment", height=175)
    submit = st.button("🔍 Analyze", use_container_width=True)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
with col2:
    st.subheader("📊 Predictions")

    if submit and user_text.strip():
        processed = preprocess_text(user_text)

        result = compare_models(
            processed,
            lr_model,
            vectorizer,
            bilstm_model,
            tokenizer
        )

        st.session_state.history.append({
            "Comment": user_text,
            "Verdict": result["verdict"].capitalize(),
            "Color": result["verdict_color"].capitalize()
        })

        def render_result_table(result):
            st.markdown(
                f"""
                <table class="result-table">
                    <tr>
                        <td class="label-cell">Log Reg</td>
                        <td class="{ result['lr_color'] }">
                            { (result['lr'] or 'N/A').upper() }
                        </td>
                    </tr>
                    <tr>
                        <td class="label-cell">BiLSTM</td>
                        <td class="{ result['bilstm_color'] }">
                            { (result['bilstm'] or 'N/A').upper() }
                        </td>
                    </tr>
                    <tr>
                        <td class="label-cell">Final Verdict</td>
                        <td class="{ result['verdict_color'] }">
                            { result['verdict'].upper() }
                        </td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

        render_result_table(result)

    elif submit:
        st.warning("Please enter a valid comment.")

# --------------------------------------------------
# History Section
# --------------------------------------------------
st.divider()
st.subheader("📜 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    def highlight(row):
        return [f"background-color: {row.Color}"] * len(row)

    st.dataframe(
        history_df.style.apply(highlight, axis=1),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No predictions yet.")
