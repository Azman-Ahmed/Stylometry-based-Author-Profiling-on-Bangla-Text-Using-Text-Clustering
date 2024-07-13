import joblib
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your trained models and preprocessing functions
models_dir = "E:/400C application/pythonProject"

vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
pca_model = joblib.load(os.path.join(models_dir, 'pca.pkl'))
umap_model = joblib.load(os.path.join(models_dir, 'umap.pkl'))
kmeans_model = joblib.load(os.path.join(models_dir, 'kmeans.pkl'))
dbscan_model = joblib.load(os.path.join(models_dir, 'dbscan.pkl'))


def preprocess_text(text):
    # Vectorize the text input
    vectorized_text = vectorizer.transform([text])

    # Apply PCA transformation
    pca_output = pca_model.transform(vectorized_text.toarray())

    # Apply UMAP transformation
    umap_output = umap_model.transform(pca_output)

    return umap_output

def predict_with_kmeans(data):
    # Predict cluster using KMeans
    cluster_label = kmeans_model.predict(data)
    return cluster_label[0]

def predict_with_dbscan(data):
    # Predict cluster using DBSCAN
    cluster_label = dbscan_model.fit_predict(data)
    return cluster_label[0]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming JSON input like {'text': 'your_text_here'}

    if 'text' not in data:
        return jsonify({'error': 'Missing text input'}), 400

    text = data['text']

    try:
        # Preprocess the text
        umap_output = preprocess_text(text)

        # Predict with KMeans
        kmeans_prediction = predict_with_kmeans(umap_output)

        # Predict with DBSCAN
        dbscan_prediction = predict_with_dbscan(umap_output)

        return jsonify({
            'text': text,
            'kmeans_cluster': kmeans_prediction,
            'dbscan_cluster': dbscan_prediction
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
