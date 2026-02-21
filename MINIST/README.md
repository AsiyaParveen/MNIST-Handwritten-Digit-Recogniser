# 🔢 MNIST Digit Recognition — ML Internship Task 2

Build a machine learning model that recognises handwritten digits (0–9) using the MNIST dataset.

---

## 📁 Project Structure

```
MINIST/
├── mnist_model.py        ← Main script: preprocess → train → evaluate → save
├── predict.py            ← Load saved model and run predictions
├── requirements.txt      ← Python dependencies
│
├── sample_digits.png     ← (generated) Sample images from dataset
├── training_history.png  ← (generated) Accuracy & loss curves
├── confusion_matrix.png  ← (generated) Per-class confusion matrix
├── predictions.png       ← (generated) Random test predictions
├── best_mnist_model.keras← (generated) Best model checkpoint
└── mnist_cnn_final.keras ← (generated) Final saved model
```

---

## 🧠 Model Architecture (CNN)

| Layer            | Output Shape   | Details                        |
|------------------|----------------|--------------------------------|
| Conv2D (32)      | 28 × 28 × 32   | 3×3 kernel, ReLU, BatchNorm    |
| MaxPooling2D     | 14 × 14 × 32   |                                |
| Conv2D (64)      | 14 × 14 × 64   | 3×3 kernel, ReLU, BatchNorm    |
| MaxPooling2D     |  7 ×  7 × 64   |                                |
| Conv2D (128)     |  7 ×  7 × 128  | 3×3 kernel, ReLU, BatchNorm    |
| Flatten          | 6272           |                                |
| Dense (256)      | 256            | ReLU + Dropout (0.5)           |
| Dense (10)       | 10             | Softmax — output probabilities |

- **Optimiser:** Adam  
- **Loss:** Categorical Cross-Entropy  
- **Callbacks:** EarlyStopping + ModelCheckpoint

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python mnist_model.py
```
> This downloads MNIST automatically, trains the CNN (~3–5 min on CPU), and saves all output plots.

### Step 3 — Run predictions
```bash
# Predict on 10 random test images
python predict.py

# Predict on a custom image
python predict.py --image path/to/digit.png
```

---

## 📊 Expected Results

| Metric              | Expected Value |
|---------------------|----------------|
| Test Accuracy       | **~99%**       |
| Epochs (early stop) | 10–15          |
| Batch Size          | 128            |
| Training Samples    | 60,000         |
| Test Samples        | 10,000         |

---

## 📤 Output Files

| File                   | Description                        |
|------------------------|------------------------------------|
| `sample_digits.png`    | Grid of one sample per digit class |
| `training_history.png` | Loss & accuracy over epochs        |
| `confusion_matrix.png` | Heatmap of predictions vs truth    |
| `predictions.png`      | 10 random test predictions         |

---

## ⚙️ Preprocessing Steps

1. **Reshape** — images from `(28, 28)` → `(28, 28, 1)` (add channel dim for CNN)
2. **Normalise** — pixel values from `[0, 255]` → `[0.0, 1.0]`
3. **One-hot encode** — labels `[0–9]` → 10-element binary vectors
