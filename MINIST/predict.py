"""
predict.py — Load the saved MNIST model and predict on test images.

Usage:
    python predict.py
    python predict.py --image path/to/custom_image.png
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import sys
from PIL import Image


def load_model(model_path="best_mnist_model.keras"):
    """Load the saved Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"[✔] Model loaded from '{model_path}'")
        return model
    except Exception as e:
        print(f"[✗] Could not load model: {e}")
        print("    → Please run `python mnist_model.py` first to train and save the model.")
        sys.exit(1)


def predict_from_mnist(model, num_samples=10):
    """Run predictions on random MNIST test images."""
    print("\n[INFO] Predicting on random MNIST test images...\n")

    (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    indices  = np.random.choice(len(X_test), num_samples, replace=False)
    X_sample = X_test[indices]
    y_true   = y_test[indices]

    probs  = model.predict(X_sample, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    confs  = np.max(probs, axis=1) * 100

    # Print results
    print(f"{'Index':>6}  {'True':>5}  {'Pred':>5}  {'Confidence':>11}  {'Status'}")
    print("-" * 50)
    for i in range(num_samples):
        status = "✅" if y_pred[i] == y_true[i] else "❌"
        print(f"{indices[i]:>6}  {y_true[i]:>5}  {y_pred[i]:>5}  {confs[i]:>10.2f}%  {status}")

    accuracy = np.mean(y_pred == y_true) * 100
    print(f"\n  Accuracy on this sample: {accuracy:.1f}%")

    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("MNIST Predictions", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_sample[i].squeeze(), cmap="gray")
        color = "green" if y_pred[i] == y_true[i] else "red"
        ax.set_title(
            f"Pred: {y_pred[i]}  ({confs[i]:.1f}%)\nTrue: {y_true[i]}",
            color=color, fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("live_predictions.png", dpi=120)
    print("\n[✔] Plot saved → live_predictions.png")
    plt.show()


def predict_custom_image(model, image_path):
    """Predict on a user-provided grayscale image (28×28 or resized)."""
    print(f"\n[INFO] Predicting on custom image: {image_path}\n")

    img = Image.open(image_path).convert("L")       # Greyscale
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    probs  = model.predict(img_array, verbose=0)[0]
    y_pred = np.argmax(probs)
    conf   = probs[y_pred] * 100

    print(f"  Predicted Digit : {y_pred}")
    print(f"  Confidence      : {conf:.2f}%")
    print(f"\n  Full probability distribution:")
    for digit, p in enumerate(probs):
        bar = "█" * int(p * 30)
        print(f"    {digit}: {bar:<30}  {p*100:6.2f}%")

    # Show image
    plt.figure(figsize=(4, 4))
    plt.imshow(np.array(img), cmap="gray")
    plt.title(f"Predicted: {y_pred}  ({conf:.1f}%)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("custom_prediction.png", dpi=120)
    print("\n[✔] Plot saved → custom_prediction.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Predictor")
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a custom 28×28 image (optional)"
    )
    parser.add_argument(
        "--model", type=str, default="best_mnist_model.keras",
        help="Path to the saved Keras model"
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        predict_custom_image(model, args.image)
    else:
        predict_from_mnist(model)
