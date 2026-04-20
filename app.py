"""
Automated Early Autism Detection through Facial Image Processing
Simple Flask Web Application
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import numpy as np
import base64
import io
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key="autism_project"

# ─── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "autism_model.h5")

model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    print("⚠️  No trained model found. Run train_model.py first.")

# ─── Helper: Preprocess Image ─────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert PIL image → grayscale → 48x48 → normalized numpy array."""
    img = img.convert("L")           # grayscale
    img = img.resize((48, 48))       # resize to model input size
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize [0,1]
    arr = arr.reshape(1, 48, 48, 1)  # batch + channel dim
    return arr

# ─── Routes ───────────────────────────────────────────────────────────────────
# @app.route("/")
# def index():
#     return render_template("index.html")
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect(url_for("home"))

    return render_template("login.html")

@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receives an image (file upload or base64) and returns prediction."""
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    try:
        # --- Accept file upload ---
        if "file" in request.files:
            file = request.files["file"]
            img = Image.open(file.stream)

        # --- Accept base64 (webcam capture) ---
        elif request.json and "image" in request.json:
            data_url = request.json["image"]
            # Strip header like "data:image/jpeg;base64,"
            header, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))

        else:
            return jsonify({"error": "No image provided."}), 400

        # --- Preprocess & predict ---
        arr = preprocess_image(img)
        prediction = model.predict(arr)[0][0]   # single sigmoid output

        label = "High Risk of ASD" if prediction >= 0.5 else "Low Risk (Typically Developing)"
        confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        confidence_pct = round(confidence * 100, 1)

        return jsonify({
            "label": label,
            "confidence": confidence_pct,
            "raw_score": float(prediction)
        })
        

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
