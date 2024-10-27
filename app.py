# File: app.py

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the vectorizer and model
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        tweet = request.form["tweet"]

        # Transform input data using the loaded vectorizer
        tweet_vectorized = vectorizer.transform([tweet])

        # Predict using the loaded model
        prediction = model.predict(tweet_vectorized)
        result = "Real" if prediction[0] == 1 else "Fake"

    else:
        result = False  # Set result to False if input is empty

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True) 