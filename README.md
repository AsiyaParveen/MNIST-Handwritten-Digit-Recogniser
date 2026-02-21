# MNIST-Handwritten-Digit-Recogniser
 MNIST Digit Recognition — CNN (~99% accuracy) + Flask live drawing UI · ML Internship Task 2

 🔢 MNIST Handwritten Digit Recogniser
An end-to-end Machine Learning project that trains a Convolutional Neural Network on the MNIST dataset to classify handwritten digits (0–9) with ~99% test accuracy, served through a live Flask web app where users draw digits on a canvas for real-time AI predictions.

## Features
1- Full preprocessing pipeline (normalise, reshape, one-hot encode)
2- block CNN with BatchNorm, MaxPooling & Dropout
3- EarlyStopping + ModelCheckpoint callbacks
4- Classification report + Confusion matrix visualisation
5- Interactive Flask UI — draw a digit, get instant predictions
6- Confidence score + per-class probability bars
7- Prediction history panel
 Stack
Python ·
TensorFlow/Keras
· Scikit-learn ·
Flask ·
Matplotlib · 
Seaborn

## Quick Start
bash
pip install -r requirements.txt
# Train the model
jupyter notebook mnist_model.ipynb
# Launch the web app
python app.py
Open http://127.0.0.1:5000 and start drawing! ✏️
