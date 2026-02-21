# MNIST-Handwritten-Digit-Recogniser
 MNIST Digit Recognition — CNN (~99% accuracy) + Flask live drawing UI · ML Internship Task 2

 🔢 MNIST Handwritten Digit Recogniser
An end-to-end Machine Learning project that trains a Convolutional Neural Network on the MNIST dataset to classify handwritten digits (0–9) with ~99% test accuracy, served through a live Flask web app where users draw digits on a canvas for real-time AI predictions.

🚀 Features
✅ Full preprocessing pipeline (normalise, reshape, one-hot encode)
✅ 3-block CNN with BatchNorm, MaxPooling & Dropout
✅ EarlyStopping + ModelCheckpoint callbacks
✅ Classification report + Confusion matrix visualisation
✅ Interactive Flask UI — draw a digit, get instant predictions
✅ Confidence score + per-class probability bars
✅ Prediction history panel
🛠️ Stack
Python · TensorFlow/Keras · Scikit-learn · Flask · Matplotlib · Seaborn

▶️ Quick Start
bash
pip install -r requirements.txt
# Train the model
jupyter notebook mnist_model.ipynb
# Launch the web app
python app.py
Open http://127.0.0.1:5000 and start drawing! ✏️
