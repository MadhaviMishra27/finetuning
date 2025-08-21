# app.py

import os
import torch
import joblib
import streamlit as st
from transformers import BertTokenizerFast, BertForSequenceClassification

# ----------------------------
# Paths
# ----------------------------
MODEL_DIR = "BERT/bert-finetuned-final"
LABEL_ENCODER_PATH = "BERT/label_encoder.pkl"


@st.cache_resource
def load_resources():
    # ✅ Load tokenizer safely
    try:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    except OSError:
        st.warning("Tokenizer files not found in fine-tuned folder. Using base tokenizer instead.")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ✅ Load model
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # ✅ Load label encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    else:
        raise FileNotFoundError(f"{LABEL_ENCODER_PATH} not found! Did you save it after training?")

    return tokenizer, model, label_encoder

tokenizer, model, label_encoder = load_resources()

# ----------------------------
# Prediction function
# ----------------------------
def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📝 BERT Sentiment Analysis")
st.write("Enter a product review below to predict its sentiment.")

user_input = st.text_area("Review:", "")

if st.button("Predict"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("⚠️ Please enter a review before predicting.")
