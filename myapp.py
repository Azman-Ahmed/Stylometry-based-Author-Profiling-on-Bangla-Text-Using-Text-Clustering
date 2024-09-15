from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import re
import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

app = Flask(__name__)

CORS(app)

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

with open('models/X_pca_for_tsne.json', 'r') as file:
    data = json.load(file)

pca_list: list = data["X_pca_for_tsne"]
author_list: list = data["author_list"]

file.close()

pca_array_full = np.array(pca_list)

def preprocess_text(text: str):
    tokenized = tokenize_bangla(text)
    processed_text = ' '.join(tokenized)

    vectorized_text = vectorizer.transform([processed_text])
    pca_transformed = pca_model.transform(vectorized_text.toarray())
    # print(pca_transformed)

    merged_array = np.vstack((pca_transformed, pca_array_full))

    tsne_transformed = tsne_model.fit_transform(merged_array)
    print(tsne_transformed)
    
    # tsne_transformed = np.asarray(tsne_transformed, dtype=np.float64)
    # print(tsne_transformed)
    
    return tsne_transformed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        input_text = data.get('input_text', None)
        if input_text:
            pass
        else:
            return 'No input_text key found in form data.', 400
        print(input_text[:400])

        processed_data = preprocess_text(input_text)

        coordinates = processed_data[0]

        np_arr_to_fit = np.array(processed_data[1:])
        np_arr_to_predict = np.array([processed_data[0]])

        kmeans_final = KMeans(n_clusters=16)
        kmeans_final.fit(np_arr_to_fit)
        kmeans_prediction = kmeans_final.predict(np_arr_to_predict)
        
        print(kmeans_prediction)

        dbscan_final = DBSCAN(eps=2.5, min_samples=8)
        dbscan_prediction = dbscan_final.fit_predict(processed_data)
        
        field: list[list[float]] = []
        
        for i in range(len(processed_data)):
            prediction_output: list = processed_data[i].tolist()
            if i == 0:
                # input index
                prediction_output.append(int(kmeans_prediction[0]))  # Convert to int
            else:
                prediction_output.append(int(author_list[i-1]))  # Convert to int
            
            field.append(prediction_output)
        
        response = {
            'field': field,
            'kmeans': int(kmeans_prediction[0]),
            'dbscan': int(dbscan_final.labels_[0]),
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
