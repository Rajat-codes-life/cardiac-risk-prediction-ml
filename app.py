from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('le_sex.pkl', 'rb') as f:
    le_sex = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_data = {}
    if request.method == 'POST':
        input_data = request.form
        sex_encoded = 1 if request.form['sex'] == 'Male' else 0
        features = np.array([[
            float(request.form['age']),
            float(request.form['weight']),
            float(request.form['height']),
            sex_encoded,
            float(request.form['bmi']),
            float(request.form['temperature']),
            float(request.form['heart_rate']),
            float(request.form['spo2']),
            int(request.form['ecg'])
        ]])
        pred = model.predict(features)[0]
        result = 'At Risk' if pred == 1 else 'Normal'
    return render_template('index.html', result=result, data=input_data)

if __name__ == "__main__":
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True)