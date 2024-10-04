from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)

CORS(app)

vectorizer = joblib.load('embeds/vectorizer.pkl')
pca = joblib.load('embeds/pca.pkl')
open_tsne = joblib.load('embeds/open_tsne.pkl')
knn = joblib.load('embeds/knn.pkl')

def tokenize_bangla(text: str):
  r = re.compile(r'([\s\।{}]+)'.format(re.escape('!"#$%&\'()*+,’।-./:;<=>?@[\\]^_`{|}~')))
  list_: list[str] = r.split(text)
  list_ = [item.replace(" ", "").replace("\n", "").replace("\t", "") if not item.isspace() else item for item in list_ if item.strip()]
  return list_

def preprocess_text(text: str):
    tokenized = tokenize_bangla(text)
    processed_text = ' '.join(tokenized)

    vectorized_text = vectorizer.transform([processed_text])
    pca_transformed = pca.transform(vectorized_text.toarray())

    tsne_transformed = open_tsne.transform(pca_transformed)
    print(tsne_transformed)
    
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
        print(coordinates)

        knn_pred = knn.predict(processed_data)
        print(f"KNN pred: {knn_pred}")
        
        response = {
            'field': open_tsne.tolist(),
            'pred': int(knn_pred[0]),
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
