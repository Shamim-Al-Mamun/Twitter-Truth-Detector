
# 🐦 Twitter Real vs Fake Detector

A Flask-based web application that predicts whether a tweet is **real** or **fake** using a machine learning model trained on labeled tweet data.

---

## 🧠 Technique Used

### Logistic Regression
- A **supervised machine learning algorithm** used for binary classification.
- Predicts probabilities using the **sigmoid function**, then classifies tweets based on a threshold (typically 0.5).
- Common in NLP tasks due to its simplicity and effectiveness.

#### 🧩 Workflow:
1. **Data Preprocessing**
   - Tokenization, stopword removal, and vectorization using **TF-IDF**.
2. **Model Training**
   - Trained on labeled tweets to learn distinctions between real and fake content.
3. **Prediction (Inference)**
   - New tweets are processed with the saved vectorizer and classified by the model.

---

## 📜 File Descriptions

### `templates/index.html`
- HTML user interface of the app.
- Users can input a tweet and get prediction results.

### `static/style.css`
- CSS styling for the frontend.
- Improves layout, color, and design to enhance user experience.

### `app.py`
- Main backend script using **Flask**.
- Defines routes for:
  - Rendering the homepage (`GET`)
  - Handling form submission and returning results (`POST`)
- Loads `model.pkl` and `vectorizer.pkl` to make predictions.

### `train_model.py`
- Script to:
  - Preprocess tweet data
  - Train a **Logistic Regression** classifier
  - Save the trained model and vectorizer to `.pkl` files

### `model.pkl` & `vectorizer.pkl`
- `model.pkl`: Contains the trained Logistic Regression model (serialized with `pickle`).
- `vectorizer.pkl`: Stores the TF-IDF vectorizer for converting text input to numerical format.
- These are loaded in `app.py` during prediction time.

---

## ✅ Features

- Input any tweet to check whether it's **Real** or **Fake**.
- Simple and elegant UI.
- Backend prediction using trained ML model.
- Uses **Flask**, **Scikit-learn**, and **TF-IDF Vectorization**.

---

## 💡 Future Improvements

- Add more complex models like SVM or Deep Learning for better accuracy.
- Connect to live Twitter API for real-time analysis.
- Display confidence scores or explanation for predictions.

---

## 🚀 Installation Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Shamim-Al-Mamun/Twitter-Truth-Detector

   cd Twitter-Truth-Detector

   pip install -r requirements.txt
   
   python train_model.py

   python app.py




## 📄 License

This project is open-source and free to use for educational and non-commercial purposes.

---

> Built with ❤️ by [Shamim Al Mamun](https://github.com/Shamim-Al-Mamun)


