\# 📰 Day 3 – Fake News Detection (NLP)



\## 📌 Overview

This project detects whether a news article is \*\*Fake\*\* or \*\*Real\*\* using

Natural Language Processing (NLP) and Machine Learning.



The user pastes a news article, and the system predicts its authenticity

along with confidence.



This project is part of my \*\*15 Days AI/ML Challenge\*\*.



---



\## 🧠 How It Works



\### 1️⃣ Text Preprocessing

\- Converts text to lowercase

\- Removes punctuation and numbers

\- Removes common stopwords (like \*the, is, and\*)

\- Keeps only meaningful words



\### 2️⃣ TF-IDF Vectorization

\- Converts text into numerical features

\- Important words get higher weight

\- Common words get lower weight



\### 3️⃣ Model Training

\- Algorithm: Logistic Regression

\- Trained on a large fake vs real news dataset

\- High accuracy on unseen data



\### 4️⃣ Prediction Logic

\- If text is too short → marked as \*\*Uncertain\*\*

\- If confidence is low → marked as \*\*Uncertain\*\*

\- Otherwise → classified as \*\*Fake\*\* or \*\*Real\*\*



---



\## 🖥️ Features

\- Clean and colorful Streamlit UI

\- Confidence-based predictions

\- Handles uncertain inputs gracefully

\- Auto-training in cloud (no dataset upload)

\- Watermark on UI:

&nbsp; \*\*Snehal — 15 Days AI/ML Challenge\*\*



---



\## 🚀 Tech Stack

\- Python

\- Scikit-learn

\- NLP (TF-IDF)

\- Streamlit

\- Hugging Face Datasets



---



\## ⚠️ Disclaimer

This project is for \*\*learning purposes only\*\* and should not be used as a

real-world fact-checking system.



---



\## 👩‍💻 Author

\*\*Snehal Arbooj\*\*  

B.Tech CSE (AI \& ML)



