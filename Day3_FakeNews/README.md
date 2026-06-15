# 📰 Day 3 – Fake News Detection (NLP)

## 📌 Overview

This project predicts whether a news article is **Fake** or **Real** using Natural Language Processing (NLP) and Machine Learning.

Users can paste a news article into the application and receive a prediction along with a confidence score.

This project was built as part of my **15 Days AI/ML Challenge**.

---

## 🧠 Workflow

### 1️⃣ Text Preprocessing

* Converts text to lowercase
* Removes punctuation and special characters
* Removes common English stopwords
* Retains meaningful words for analysis

### 2️⃣ TF-IDF Vectorization

* Converts text into numerical features
* Assigns higher importance to informative words
* Reduces the impact of commonly occurring words

### 3️⃣ Machine Learning Model

* Algorithm: Logistic Regression
* Uses TF-IDF features as input
* Performs binary classification (Fake vs Real)

### 4️⃣ Prediction Logic

* Detects insufficient input text
* Handles low-confidence predictions
* Returns Fake, Real, or Uncertain results

---

## ✨ Features

* Interactive Streamlit interface
* NLP-based text classification
* Confidence score display
* Graceful handling of uncertain predictions
* Lightweight deployment
* Custom project watermark

---

## 🚀 Tech Stack

* Python
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Streamlit
* NLTK

---

## ⚠️ Disclaimer

This project is intended for educational and learning purposes only and should not be used as a substitute for professional fact-checking services.

---

## 👩‍💻 Author

**Snehal Arbooj**
B.Tech CSE (AI & ML)

**15 Days AI/ML Challenge**
