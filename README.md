Project Details:

templates/index.html
A simple HTML file that serves as the user interface for the project.
Displays a webpage in the browser where users can input tweets and view the prediction results.

static/style.css
A CSS file responsible for designing the appearance of the webpage.
Provides styling and layout to enhance user experience.

app.py
The main Python script that runs the project using the Flask framework.
Handles HTTP requests and responses, and defines routes for rendering the webpage and processing user inputs.
Includes backend logic to connect user input with the machine learning model and return predictions.

Train_model.py
A Python script used for text preprocessing and training the machine learning model.
Prepares sample labeled data for training and processes textual data for machine learning by converting it into numerical features.
Implements the Logistic Regression algorithm to train a classifier capable of distinguishing between real and fake tweets.
Saves the trained model and vectorizer to .pkl files for later use.

Model.pkl & vectorizer.pkl
Model.pkl: A serialized representation of the trained Logistic Regression model, created using Python's pickle library.
Vectorizer.pkl: Stores the vectorizer (e.g., CountVectorizer or TfidfVectorizer) used to convert text into numerical vectors for model input.
These files are used during inference to process user input and generate predictions.
Not human-readable but essential for the operational workflow of the project.

Technique Used

Logistic Regression
A supervised machine learning algorithm used for binary classification.
Operates as a linear model to predict the probability of an input belonging to one of two classes (e.g., real or fake).
Uses a sigmoid function to estimate probabilities, which are then mapped to class labels based on a threshold (usually 0.5).
Workflow:
Data Preprocessing: Converts raw tweets into a numerical representation using techniques like tokenization, stopword removal, and vectorization (e.g., TF-IDF).
Model Training: Logistic Regression is trained on labeled data to learn patterns distinguishing real tweets from fake ones.
Inference: New tweets are processed using the saved vectorizer and fed into the trained model to predict whether they are real or fake.


