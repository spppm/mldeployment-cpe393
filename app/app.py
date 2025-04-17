from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
 

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/health", methods=["GET","POST"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        features = request.args.get("features", default="", type=str)
        try:
            features = [list(map(float, f.split(","))) for f in features.split(";")]  # Handle multiple sets from GET request
        except ValueError:
            return jsonify({"error": "Invalid input format. Features must be numeric values."}), 400
    else:
        data = request.get_json()

        # Ensure "features" key exists
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        # Ensure "features" is a list
        if not isinstance(data["features"], list):
            return jsonify({"error": "'features' must be a list"}), 400

        # Validate each input has exactly 4 float values
        for entry in data["features"]:
            if not isinstance(entry, list) or len(entry) != 4:
                return jsonify({"error": "Each input must be a list of exactly 4 float values"}), 400
            if not all(isinstance(x, (int, float)) for x in entry):
                return jsonify({"error": "Each value in 'features' must be a float or int"}), 400

        features = data["features"]

    # Convert input features to NumPy array
    input_features = np.array(features)

    # Get predictions for all input rows
    predictions = model.predict(input_features).tolist()

    # Get confidence scores for each prediction
    confidence_scores = [round(model.predict_proba(input_features)[i][pred], 2) for i, pred in enumerate(predictions)]

    return jsonify({
        "predictions": predictions,
        "confidence_scores": confidence_scores
    })
    
num_features = xgb_model.n_features_in_

@app.route("/predictHouse", methods=["GET", "POST"])
def housePrice():
    if request.method == "GET":
        features = request.args.get("features", default="", type=str)
        try:
            features = [list(map(float, f.split(","))) for f in features.split(";")]  # Multiple rows via ;
        except ValueError:
            return jsonify({"error": "Invalid input format. Features must be numeric values."}), 400
    else:
        data = request.get_json()

        # Ensure "features" key exists
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        # Ensure it's a list
        if not isinstance(data["features"], list):
            return jsonify({"error": "'features' must be a list"}), 400

        # Validate each input
        for entry in data["features"]:
            if not isinstance(entry, list) or len(entry) != num_features:
                return jsonify({
                    "error": f"Each input must be a list of exactly {num_features} float values"
                }), 400
            if not all(isinstance(x, (int, float)) for x in entry):
                return jsonify({"error": "Each value in 'features' must be a float or int"}), 400

        features = data["features"]

    # Convert to NumPy array
    input_features = np.array(features)

    # Make prediction
    predictions = xgb_model.predict(input_features).tolist()

    return jsonify({
        "predicted_prices": predictions
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000,debug=True) #check your port number ( if it is in use, change the port number)
