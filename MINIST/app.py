"""
app.py — Flask web app for MNIST Digit Recognition
====================================================
Routes:
  GET  /           → Serve the drawing UI
  POST /predict    → Accept base64 canvas image, return predicted digit + probabilities
"""

import os
import re
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = "best_mnist_model.keras"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found.\n"
            "Please run mnist_model.ipynb first to train and save the model."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[✔] Model loaded from '{MODEL_PATH}'")
    return model

model = load_model()


# ── Helper: preprocess the canvas image ────────────────────────────────────
def preprocess_image(data_url: str) -> np.ndarray:
    """
    Convert a base64 PNG data URL (from HTML canvas) into a
    28×28 greyscale numpy array normalised to [0, 1].
    """
    # Strip the data URL header  →  pure base64
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    # Open with Pillow
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Canvas draws white-on-black; MNIST is white digit on black background
    # Flatten alpha onto black background
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("L")          # greyscale

    # Resize to 28×28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy and normalise
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data received"}), 400

    try:
        arr = preprocess_image(data["image"])
        probs = model.predict(arr, verbose=0)[0]         # shape (10,)
        digit = int(np.argmax(probs))
        confidence = float(probs[digit]) * 100

        return jsonify({
            "digit":      digit,
            "confidence": round(confidence, 2),
            "probs":      [round(float(p) * 100, 2) for p in probs],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  MNIST Digit Recognition — Flask App")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True)
