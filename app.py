import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re

# --- Custom CSS for background and styling ---
def inject_custom_css():
    st.markdown("""
        <style>
            /* Background image */
            body {
                background-image: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1350&q=80');
                background-size: cover;
                background-attachment: fixed;
            }
            /* Transparent container */
            .main-container {
                background-color: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 0 30px rgba(0, 0, 0, 0.3);
                max-width: 700px;
                margin: 4rem auto;
            }
            h1, h2, h3 {
                color: #1f2937;
                text-align: center;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .stTextArea textarea {
                background-color: #f8fafc;
                border-radius: 10px;
                font-size: 1rem;
            }
            .stButton>button {
                background-color: #4f46e5;
                color: white;
                font-weight: bold;
                padding: 0.75rem 1.5rem;
                border-radius: 30px;
                font-size: 1.1rem;
                margin: 1rem 0;
            }
            .stButton>button:hover {
                background-color: #4338ca;
            }
            .confidence-box {
                background-color: #f1f5f9;
                border-left: 5px solid #4f46e5;
                padding: 0.75rem 1rem;
                border-radius: 10px;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

# --- Load the trained Keras model (GRU) ---
@st.cache_resource
def load_sentiment_model():
    return tf.keras.models.load_model('gru_model.h5')

# --- Load the tokenizer ---
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --- App Layout ---
inject_custom_css()
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("💡 Enter a movie review below, and our smart GRU model will reveal whether it's Positive, Neutral, or Negative.")

review = st.text_area("📝 Your Movie Review:", height=180)

if st.button("🚀 Predict Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        max_len = 200
        padded = pad_sequences(seq, maxlen=max_len)

        # Predict
        model = load_sentiment_model()
        preds = model.predict(padded)
        pred_proba = preds[0]
        classes = ['Negative', 'Neutral', 'Positive']
        pred_index = np.argmax(pred_proba)
        pred_label = classes[pred_index]

        # Display prediction
        st.markdown(f"### 🧠 **Predicted Sentiment: `{pred_label}`**")
        st.markdown("### 📊 **Confidence Scores:**")
        for i, label in enumerate(classes):
            st.markdown(
                f"<div class='confidence-box'><b>{label}:</b> {pred_proba[i]*100:.2f}%</div>",
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)