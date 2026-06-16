from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# ============================
# Load Model and Vectorizer
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "sentiment_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
)

# ============================
# Routes
# ============================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    sentiment_class = ""
    review = ""
    confidence = None

    if request.method == "POST":

        review = request.form["review"]

        vectorized_review = vectorizer.transform([review])

        prediction = model.predict(vectorized_review)[0]

        probs = model.predict_proba(vectorized_review)[0]

        confidence = max(probs) * 100

        if confidence < 80:
            result = "🤔 Mixed / Uncertain Review"
            sentiment_class = "uncertain"

        elif prediction == "positive":
            result = "😊 Positive Review"
            sentiment_class = "positive"

        else:
            result = "😞 Negative Review"
            sentiment_class = "negative"

    return render_template(
        "index.html",
        result=result,
        sentiment_class=sentiment_class,
        review=review,
        confidence=round(confidence, 2) if confidence else None
    )

# ============================
# Run App
# ============================

if __name__ == "__main__":
    app.run()