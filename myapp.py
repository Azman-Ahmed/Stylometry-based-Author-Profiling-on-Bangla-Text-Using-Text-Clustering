from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
pca_model = joblib.load('models/pca.pkl')
umap_model = joblib.load('models/umap.pkl')
dbscan_model = joblib.load('models/dbscan.pkl')
kmeans_model = joblib.load('models/kmeans.pkl')

def preprocess_text(text):
    cleaned_text = text
    vectorized_text = vectorizer.transform([cleaned_text])
    pca_transformed = pca_model.transform(vectorized_text.toarray())
    umap_transformed = umap_model.transform(pca_transformed)
    return umap_transformed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        processed_data = preprocess_text(input_text)
        prediction = kmeans_model.predict(processed_data)
        coordinates = processed_data[0]
        response = {
            'index': str(prediction[0]),
            'coordinates': {
                'x': float(coordinates[0]),
                'y': float(coordinates[1]),
                'z': float(coordinates[2]) if len(coordinates) > 2 else None
            }
        }

        return jsonify(response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
