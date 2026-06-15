import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    return " ".join(w for w in words if w not in stop_words)

# Load trained artifacts
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection | Snehal", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #fdfbfb, #ebedee); }
h1, h2, p { font-family: Segoe UI; }
textarea { border-radius: 10px; }
button {
    background-color: #6366f1 !important;
    color: white !important;
    border-radius: 10px;
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

st.title("📰 Fake News Detection System")

text = st.text_area(
    "News Article Text",
    height=180,
    placeholder="Paste a full news article here..."
)

if st.button("Check News"):
    if len(text.strip()) < 100:
        st.markdown("<div class='result-box uncertain'>⚠️ Text too short.</div>", unsafe_allow_html=True)
    else:
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        fake_prob, real_prob = model.predict_proba(vect)[0]
        confidence = max(fake_prob, real_prob)

        if confidence < 0.65:
            st.markdown("<div class='result-box uncertain'>🤔 Uncertain result.</div>", unsafe_allow_html=True)
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

st.markdown("<div class='watermark'>Snehal — 15 Days AI/ML Challenge</div>", unsafe_allow_html=True)
