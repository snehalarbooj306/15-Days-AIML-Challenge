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
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
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
# Create Sample Data (Fallback)
# -----------------------------
def create_sample_data():
    """Create sample data if external datasets are not available"""
    fake_samples = [
        "Scientists discover aliens living among us in secret underground cities.",
        "Breaking: President announces plan to replace all money with chocolate coins.",
        "Miracle cure discovered: drinking lemon water cures all diseases instantly.",
        "Local man claims he can predict future by reading coffee stains.",
        "Government confirms existence of time travel technology in basement.",
        "New study shows that thinking positive thoughts can change DNA structure.",
        "Celebrity chef reveals cooking with magical ingredients from fairy realm.",
        "Doctors hate this one weird trick that makes you live forever.",
        "Archaeological team discovers dinosaur bones that are only 100 years old.",
        "Weather control machine discovered in abandoned warehouse downtown."
    ]
    
    real_samples = [
        "The Federal Reserve announced a quarter-point interest rate increase following inflation concerns.",
        "New climate change report shows global temperatures rising faster than previously predicted.",
        "Tech company announces breakthrough in quantum computing with new processor design.",
        "Local school district receives federal grant for improved STEM education programs.",
        "Medical researchers publish findings on effectiveness of new cancer treatment protocol.",
        "City council approves budget for infrastructure improvements including road repairs.",
        "University study examines impact of social media usage on teenage mental health.",
        "Environmental agency reports on air quality improvements following emission regulations.",
        "Economic indicators suggest steady growth in manufacturing sector this quarter.",
        "Public health officials recommend updated vaccination schedules for children."
    ]
    
    # Create DataFrame
    data = []
    for text in fake_samples:
        data.append({"text": text, "label": 0})
    for text in real_samples:
        data.append({"text": text, "label": 1})
    
    return pd.DataFrame(data)

# -----------------------------
# Load or Train Model (FIXED)
# -----------------------------
@st.cache_resource
def load_or_train():
    if os.path.exists("fake_news_model.pkl") and os.path.exists("tfidf_vectorizer.pkl"):
        try:
            model = joblib.load("fake_news_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            return model, vectorizer
        except:
            pass  # If loading fails, retrain
    
    # Try multiple data sources
    df = None
    
    # Option 1: Try original URLs (with error handling)
    try:
        fake_url = "https://huggingface.co/datasets/streamlit/example-fake-news/resolve/main/Fake.csv"
        true_url = "https://huggingface.co/datasets/streamlit/example-fake-news/resolve/main/True.csv"
        
        fake_df = pd.read_csv(fake_url)
        true_df = pd.read_csv(true_url)
        
        fake_df["label"] = 0
        true_df["label"] = 1
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
    except Exception as e1:
        # Option 2: Try alternative URLs
        try:
            # These are more reliable fake news dataset URLs
            fake_url = "https://raw.githubusercontent.com/nishitpatel01/Fake_News_Detection/master/train.csv"
            df_alt = pd.read_csv(fake_url)
            # Assuming the dataset has 'title', 'text', and 'label' columns
            if 'title' in df_alt.columns and 'text' in df_alt.columns:
                df_alt['combined_text'] = df_alt['title'].fillna('') + ' ' + df_alt['text'].fillna('')
                df = df_alt[['combined_text', 'label']].rename(columns={'combined_text': 'text'})
            else:
                raise Exception("Alternative dataset structure not as expected")
                
        except Exception as e2:
            # Option 3: Use sample data as last resort
            st.warning("⚠️ Using sample training data. For better accuracy, please upload your own dataset.")
            df = create_sample_data()
    
    # Ensure we have the required columns
    if df is None or 'text' not in df.columns or 'label' not in df.columns:
        df = create_sample_data()
    
    # Clean and prepare data
    df = df.dropna(subset=['text'])  # Remove rows with missing text
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df["clean_text"].str.len() > 0]
    
    if len(df) < 10:  # If we don't have enough data
        df = create_sample_data()
        df["clean_text"] = df["text"].apply(clean_text)
    
    X = df["clean_text"]
    y = df["label"]
    
    # Train model
    vectorizer = TfidfVectorizer(max_features=5000, min_df=1, max_df=0.95)
    X_tfidf = vectorizer.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tfidf, y)
    
    # Save model
    try:
        joblib.dump(model, "fake_news_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    except:
        pass  # Don't fail if we can't save
    
    return model, vectorizer

# Load model with error handling
try:
    model, vectorizer = load_or_train()
except Exception as e:
    st.error(f"Error loading/training model: {str(e)}")
    st.stop()

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
    if len(user_input.strip()) < 50:  # Reduced minimum length
        st.markdown(
            "<div class='result-box uncertain'>⚠️ Text too short to determine authenticity. Please provide more content.</div>",
            unsafe_allow_html=True
        )
    else:
        try:
            cleaned = clean_text(user_input)
            if len(cleaned.strip()) == 0:
                st.markdown(
                    "<div class='result-box uncertain'>⚠️ No meaningful content found after cleaning.</div>",
                    unsafe_allow_html=True
                )
            else:
                vectorized = vectorizer.transform([cleaned])
                probs = model.predict_proba(vectorized)[0]
                
                fake_prob, real_prob = probs
                confidence = max(fake_prob, real_prob)
                
                if confidence < 0.6:  # Lowered threshold
                    st.markdown(
                        f"<div class='result-box uncertain'>🤔 Cannot confidently classify this text.<br>Confidence: {confidence*100:.2f}%</div>",
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
        except Exception as e:
            st.markdown(
                "<div class='result-box uncertain'>⚠️ Error processing text. Please try again.</div>",
                unsafe_allow_html=True
            )

st.markdown(
    "<div class='watermark'>Snehal — 15 Days AI/ML Challenge</div>",
    unsafe_allow_html=True
)
