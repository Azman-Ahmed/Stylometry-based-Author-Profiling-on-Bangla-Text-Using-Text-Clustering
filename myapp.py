import pickle
from flask import Flask, render_template, request


with open('kmeans.pkl', 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Welcome"

@app.route('/wtf', methods=["GET", "POST"])
def wtf():
    if request.method == "GET":
        return render_template('tem.html')
    else:
        text1 = request.form['text1']


        prediction = model.predict([text1])  
        return render_template('tem.html', text2=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
    