import os
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Set the path to the directory where your models are located
models_dir = os.path.dirname(os.path.abspath(__file__))  # get current directory

# Load your trained models and preprocessing functions
vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
pca_model = joblib.load(os.path.join(models_dir, 'pca.pkl'))
kmeans_model = joblib.load(os.path.join(models_dir, 'kmeans.pkl'))
umap_model = joblib.load(os.path.join(models_dir, 'umap.pkl'))
dbscan_model = joblib.load(os.path.join(models_dir, 'dbscan.pkl'))

# Define functions for preprocessing and prediction
def preprocess_input(text):
    # Implement your preprocessing steps using loaded models
    # Example: transform text using loaded vectorizer and pca_model
    vectorized_input = vectorizer.transform([text])
    transformed_input = pca_model.transform(vectorized_input)
    return transformed_input

def predict_author(input_data):
    # Implement your prediction logic using loaded kmeans_model
    predicted_label = kmeans_model.predict(input_data)
    return predicted_label[0]  # assuming single prediction for user input

# Define your Flask routes
@app.route('/')
def index():
    return render_template('index.html')  # render your HTML form

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']  # get user input from the form
        processed_input = preprocess_input(user_input)
        predicted_author = predict_author(processed_input)
        return render_template('result.html', author=predicted_author)

if __name__ == '__main__':
    app.run(debug=True)
