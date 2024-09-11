from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

vectorizer = joblib.load('models/vectorizer.pkl')
pca_model = joblib.load('models/pca_100.pkl')
tsne_model = joblib.load('models/pca_tsne.pkl')
dbscan_model = joblib.load('models/dbscan.pkl')
kmeans_model = joblib.load('models/kmeans.pkl')

def tokenize_bangla(text: str):
  r = re.compile(r'([\s\।{}]+)'.format(re.escape('!"#$%&\'()*+,’।-./:;<=>?@[\\]^_`{|}~')))
  list_: list[str] = r.split(text)
  list_ = [item.replace(" ", "").replace("\n", "").replace("\t", "") if not item.isspace() else item for item in list_ if item.strip()]
  return list_

def preprocess_text(text: str):
    tokenized = tokenize_bangla(text)
    processed_text = ' '.join(tokenized)

    vectorized_text = vectorizer.transform([processed_text])
    pca_transformed = pca_model.transform(vectorized_text.toarray())
    tsne_transformed = tsne_model.transform(pca_transformed)
    return tsne_transformed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        processed_data = preprocess_text(input_text)
        coordinates = processed_data[0]
        
        kmeans_prediction = kmeans_model.predict(processed_data)
        dbscan_prediction = dbscan_model.fit_predict(processed_data)
        
        response = {
            'kmeans': int(kmeans_prediction[0]),
            'dbscan': int(dbscan_prediction[0]),
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
