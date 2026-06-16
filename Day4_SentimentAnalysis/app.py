from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

result = None
sentiment_class = ""
review = ""
confidence = None

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    sentiment_class = ""



    if request.method == "POST":

        review = request.form["review"]

        vectorized_review = vectorizer.transform([review])

        prediction = model.predict(vectorized_review)[0]

        probs = model.predict_proba(vectorized_review)[0]

        negative_prob = probs[0]
        positive_prob = probs[1]

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

    return render_template("index.html", result=result, sentiment_class=sentiment_class, review=review , confidence=round(confidence, 2))

if __name__ == "__main__":
    app.run(debug=True)