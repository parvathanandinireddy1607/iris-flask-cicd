from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")


@app.route("/")
def home():
    return "Iris Classifier API is Running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Features are required"}), 400

    features = data["features"]

    # Iris dataset requires exactly 4 features
    if len(features) != 4:
        return jsonify({"error": "Exactly 4 features are required"}), 400

    prediction = model.predict([features])

    # Convert model output to class name
    class_names = ["setosa", "versicolor", "virginica"]
    predicted_class = class_names[int(prediction[0])]

    return jsonify({"prediction": predicted_class})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
