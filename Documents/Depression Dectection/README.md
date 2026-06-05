# Depression Detection System

A Python-based emotion recognition and depression risk prediction system using computer vision and machine learning.

## Features

- Face detection with OpenCV Haar Cascade
- Emotion recognition model scaffold for CNN / CNN+LSTM
- Depression analysis from emotion history
- SQLite storage for user emotion and risk records
- Flask starter app for camera input and dashboard
- Notebook for training and evaluation

## Technologies

- Python 3.10+
- OpenCV
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Flask
- SQLite

## Project Structure

- `src/` - core application modules
- `notebooks/` - training and evaluation notebooks
- `data/` - dataset and sample data
- `requirements.txt` - Python dependencies

## Getting Started

1. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
python src/app.py
```

## Notes

- Place FER-2013 dataset files in `data/` before training.
- The current code is a scaffold and can be extended to add full model training and web UI.
