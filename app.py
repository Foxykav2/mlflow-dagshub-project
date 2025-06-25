from flask import Flask, request, jsonify
import joblib  # or pickle depending on your model save method

app = Flask(__name__)

# Load your model (adjust path as needed)
model = joblib.load('C:/Users/dany7/OneDrive/Bureau/devops/DVC, MLFlow and Dagshub/mlflow-dagshub-project/mlflow-with-daghub/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Prepare input and predict (adjust as per your model)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
