from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
kmeans = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)
        df_scaled = scaler.transform(df)
        cluster = kmeans.predict(df_scaled)
        df['Cluster'] = cluster
        return render_template('index.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

    return "No file uploaded"

if __name__ == "__main__":
    app.run(debug=True)
