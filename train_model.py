# File: train_model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# Example training data
data = pd.DataFrame({
    'text': ["Some example real news", "Some example fake news", "Another real news"],
    'label': [1, 0, 1]  # 1 for Real, 0 for Fake
})

# Prepare training data
X = data['text']
y = data['label']

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save both the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
