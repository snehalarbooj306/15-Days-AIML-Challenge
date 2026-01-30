import os
import re
import joblib
import nltk
import pandas as pd
import numpy as np
import streamlit as st

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# NLTK Setup
# -----------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# Load or Train Model (CLOUD SAFE)
# -----------------------------
@st.cache_resource
def load_or_train():
    if os.path.exists("fake_news_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
    else:
        fake_url = "https://huggingface.co/datasets/streamlit/example-fake-news/resolve/main/Fake.csv"
        true_url = "https://huggingface.co/datasets/streamlit/example-fake-news/resolve/main/True.csv"
        
        fake_df = pd.read_csv(fake_url)
        true_df = pd.read_csv(true_url)


        fake_df["label"] = 0
        true_df["label"] = 1

        df = pd.concat([fake_df, true_df], ignore_index=True)
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


model, vectorizer = load_or_train()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection | Snehal",
    layout="centered"
)

# -----------------------------
# UI Styling (UNCHANGED)
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection System")
st.write("Paste a full news article to check whether it is **Fake** or **Real**.")

user_input = st.text_area(
    "News Article Text",
    height=180,
    placeholder="Paste full news article here..."
)

if st.button("Check News"):
    if len(user_input.strip()) < 100:
        st.markdown(
            "<div class='result-box uncertain'>⚠️ Text too short to determine authenticity.</div>",
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
                "<div class='result-box uncertain'>🤔 Cannot confidently classify this text.</div>",
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

st.markdown(
    "<div class='watermark'>Snehal — 15 Days AI/ML Challenge</div>",
    unsafe_allow_html=True
)


