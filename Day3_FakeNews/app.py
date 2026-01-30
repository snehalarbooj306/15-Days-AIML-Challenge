# =============================
# Imports
# =============================
import os
import re
import joblib
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

# =============================
# NLTK Setup
# =============================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =============================
# Text Cleaning Function
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =============================
# Load or Train Model (SAFE)
# =============================
@st.cache_resource
def load_or_train_model():
    # If model already exists, load it
    if os.path.exists("fake_news_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer

    # Otherwise train from Hugging Face dataset
    dataset = load_dataset("clmentbisaillon/fake_and_real_news")

    df = pd.DataFrame(dataset["train"])
    df = df[["text", "label"]]

    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)

    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    return model, vectorizer

# =============================
# Load Model Safely
# =============================
try:
    model, vectorizer = load_or_train_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("❌ Model failed to load or train.")
    st.stop()

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Fake News Detection | Snehal",
    layout="centered"
)

# =============================
# UI Styling (UNCHANGED)
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
}
h1, h2, h3, p, label {
    color: #1f2933 !important;
    font-family: "Segoe UI", sans-serif;
}
textarea {
    border-radius: 10px !important;
    font-size: 15px !important;
}
button {
    background-color: #6366f1 !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    padding: 0.5rem 1.2rem !important;
}
.result-box {
    padding: 18px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 18px;
    font-weight: 600;
}
.fake { background-color: #fee2e2; color: #991b1b; }
.real { background-color: #dcfce7; color: #166534; }
.uncertain { background-color: #fef9c3; color: #854d0e; }
.watermark {
    text-align: center;
    margin-top: 40px;
    font-size: 13px;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# =============================
# UI Content
# =============================
st.title("📰 Fake News Detection System")
st.write(
    "Paste a news article below to check whether it is **Fake** or **Real**. "
    "Best results are obtained with full news paragraphs."
)

user_input = st.text_area(
    "News Article Text",
    height=180,
    placeholder="Paste a full news article or multiple paragraphs here..."
)

# =============================
# Prediction Logic
# =============================
if st.button("Check News"):
    if not model_ready:
        st.error("Model not ready. Please refresh the app.")
        st.stop()

    if len(user_input.strip()) < 100:
        st.markdown(
            "<div class='result-box uncertain'>⚠️ Text is too short to determine whether it is fake or real.</div>",
            unsafe_allow_html=True
        )
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        probs = model.predict_proba(vectorized)[0]

        fake_prob, real_prob = probs
        confidence = max(fake_prob, real_prob)

        if confidence < 0.65:
            st.markdown(
                "<div class='result-box uncertain'>🤔 Cannot confidently determine whether the news is fake or real.</div>",
                unsafe_allow_html=True
            )
        elif real_prob > fake_prob:
            st.markdown(
                f"<div class='result-box real'>✅ Real News<br>Confidence: {real_prob*100:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box fake'>🛑 Fake News<br>Confidence: {fake_prob*100:.2f}%</div>",
                unsafe_allow_html=True
            )

# =============================
# Watermark
# =============================
st.markdown(
    "<div class='watermark'>Snehal — 15 Days AI/ML Challenge</div>",
    unsafe_allow_html=True
)
