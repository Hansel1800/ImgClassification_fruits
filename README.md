# Fruit Condition Classification System


## 1. Project Objective

This project aims to build a deep learning system to classify the condition (fresh/rotten) of common fruits (banana, orange, apple) from images. The system:
- Classifies fruit images into 6 classes: Fresh Banana, Rotten Banana, Fresh Orange, Rotten Orange, Fresh Apple, Rotten Apple.
- Evaluates model accuracy and visualizes training progress.
- Provides a user-friendly web app (Streamlit) for uploading images and receiving classification results.

## 2. Main Directory / Structure

```
src/
  ├── main.py              # Main training script
  ├── models/
  │     └── model.py       # Model definition and training
  ├── data/
  │     └── data_loader.py # Data preprocessing, loading, augmentation
  ├── config/
  │     └── config.py      # Configuration parameters
  ├── utils/
  │     └── gpu_utils.py   # GPU support utilities
  ├── templates/
  │     └── index.html     # (Not used for Flask, for UI reference only)
  └── __init__.py
app.py                     # Streamlit web app interface
requirements.txt           # Required libraries
```

**Key Python files:**
- `main.py`: Model training, plotting, saving.
- `models/model.py`: FruitClassifier class (MobileNetV2).
- `data/data_loader.py`: Data preprocessing, augmentation, reorganization.
- `app.py`: Web app interface, image upload, result display.

## 3. Frameworks / Libraries Used

- Python >= 3.7
- TensorFlow >= 2.8.0
- Keras (integrated in TensorFlow)
- NumPy, Matplotlib, Pillow, scikit-learn
- Streamlit (web app)
- Plotly, pandas (statistics, visualization)

## 4. Dataset

- **Source:** `dataset/train` and `dataset/test` folders (prepare before training).
- **Classes:** 6 (fresh/rotten banana, orange, apple).
- **Number of images:** ~13,599 (as shown in the app sidebar).
- **Augmentation:** Rotation, zoom, shift, horizontal flip, shear, pixel normalization.
- **Image size:** 224x224 (MobileNetV2 standard).

## 5. Results / Performance

- **Model:** MobileNetV2 (fine-tuned, transfer learning).
- **Accuracy:** (Fill in your actual result after training, e.g., ~92%)
- **Charts:** Training/validation loss and accuracy (`images/train_val_acc.jpg`, `images/train_val_loss.jpg`).
- **Details:** Shows class probabilities, user-friendly interface.

## 6. Installation & Usage

### a. Clone repo & install requirements
```bash
git clone <repo_url>
cd App_demo
pip install -r requirements.txt
```

### b. Prepare dataset
- Create `dataset/train` and `dataset/test` folders with 6 subfolders for each class:
  ```
  dataset/
    ├── train/
    │     ├── chuoi_tuoi/
    │     ├── chuoi_hong/
    │     ├── cam_tuoi/
    │     ├── cam_hong/
    │     ├── tao_tuoi/
    │     └── tao_hong/
    └── test/
          (same structure as train)
  ```
- Place images in the correct class folders.

### c. Train the model
```bash
python src/main.py
```
- The trained model will be saved at `models_trained/fruit_classifier.h5`.

### d. Run the web app
```bash
streamlit run app.py
```
- Open the provided URL (usually http://localhost:8501).

## 7. User Interface

- **Web app (Streamlit):** Upload images, get classification results, view training charts, dataset statistics.
- **Modern, intuitive, drag-and-drop support.**

## 8. Future Directions

- Add more fruit types.
- More detailed condition classification (multi-level).
- Deploy a REST API for mobile app integration.
- Optimize prediction speed, support batch prediction.
- Add more metrics: precision, recall, F1-score, confusion matrix.
- Apply OpenCV for real-time fruit condition detection (real-time problem).
- Build and expand a comprehensive fruit image database (database) for more fruit types and diverse conditions.
-


