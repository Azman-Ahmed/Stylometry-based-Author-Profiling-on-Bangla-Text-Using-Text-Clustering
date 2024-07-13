from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your pre-trained models and vectorizer
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
pca_model = joblib.load('models/pca.pkl')
umap_model = joblib.load('models/umap.pkl')


def preprocess_text(text):

    cleaned_text = text  # Replace with your actual cleaning and tokenization logic
    vectorized_text = vectorizer.transform([cleaned_text])
    pca_transformed = pca_model.transform(vectorized_text)
    umap_transformed = umap_model.transform(pca_transformed)
    return umap_transformed

# Route for handling user input
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        processed_data = preprocess_text(input_text)

        prediction_result = "Your prediction result"
        return render_template('result.html', prediction=prediction_result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
