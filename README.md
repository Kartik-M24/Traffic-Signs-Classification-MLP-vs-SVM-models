# Traffic Sign Classification
Brief: Jupyter notebook that preprocesses a traffic-sign image dataset, trains an MLP (tf.keras) and an SVM (scikit-learn), and evaluates both models with standard metrics and confusion matrices.

## Files
- 306_Project_Part_1.ipynb — main notebook (data download, preprocessing, MLP training, SVM+PCA+GridSearch, evaluation).
- X_processed.npy, y_labels.npy - saved processed arrays
- mlp_model.keras, svc_model.pkl - saved models
- grids 1 2 and 3 (e.g. grid1.pkl) - saved batches of GridSearchCV for different kernels

## Requirements 
- Python 3.9.21 or above
- TensorFlow 2.11.0 (for Python 3.9)
- numpy, pandas, scikit-learn, matplotlib, seaborn, pillow, joblib, tqdm, kaggle

## Quick environment setup
1. Create & activate conda env
2. Install core packages via pip

## Kaggle dataset setup
1. Install kaggle (done above).  
2. Create API token on Kaggle and download `kaggle.json`.  
3. Move token to the Windows user kaggle folder:
4. From notebook download dataset (notebook uses Kaggle API):

## Using the notebook (VS Code)
1. In VS Code: select the appropriate interpreter you've set up  
2. Open 306_Project_Part_1.ipynb, restart kernel after installing packages.  
3. Run cells top-to-bottom. For long GridSearch jobs consider running the SVM gridsearch cells individually and checkpoint files are saved as `grid1.pkl` etc.

## Notes & suggestions
- TensorFlow must be installed in the same interpreter/kernel used by the notebook. If you see ModuleNotFoundError for `tensorflow` or `kaggle`, install into your venv and restart the kernel.
- To reproduce results quickly, use the saved .npy and model files rather than re-running full training.
- Kaggle API requires a personal kaggle.json token; alternative: include a pre-downloaded dataset folder in the repo

## Detailed description
This notebook implements a complete pipeline for traffic-sign classification using two approaches:
- A Multi-Layer Perceptron (MLP) built with TensorFlow (tf.keras).
- A Support Vector Machine (SVM, scikit-learn) using PCA for dimensionality reduction and GridSearchCV.

The goal is to compare performance, runtime, and practicality of a neural net vs a traditional SVM on the same preprocessed dataset.

## What the code does
1. Downloads the dataset (Kaggle API) into ./datasets and reads label CSV.
2. Traverses image subfolders, records paths and class IDs, and builds a DataFrame.
3. Preprocesses images:
   - Resizes to 32×32, converts to numpy arrays and normalizes pixel values to [0,1].
   - Saves processed arrays (X_processed.npy, y_labels.npy) so steps can be skipped on reruns.
4. Splits data into train / validation / test sets.
5. Trains models:
   - MLP: simple feed-forward network with flatten layer, two hidden layers, early stopping and LR scheduler, then saved to mlp_model.keras.
   - SVM: flatten images, standardize, apply PCA (98% variance), run GridSearchCV in batches to tune kernel/C/gamma, then train final SVM and save with joblib.
6. Evaluates both models using accuracy, precision, recall, F1 and confusion matrices, and stores or prints best SVM params.

## How it works (details)
- Preprocessing:
  - PIL.Image opens and resizes images using LANCZOS for quality.
  - Normalization divides pixel values by 255.0 and stores arrays for fast reload.
- MLP:
  - Input: 32×32×3 flattened.
  - Loss: SparseCategoricalCrossentropy(from_logits=True).
  - Callbacks: ReduceLROnPlateau and EarlyStopping to avoid overfitting and speed convergence.
- SVM:
  - High-dimensional flattened images are standardized with StandardScaler.
  - PCA reduces dimensionality while preserving 98% variance to speed SVM training.
  - GridSearchCV tries parameter batches (linear, rbf, poly) and checkpoints results to pkl files so work can be resumed.
- Evaluation:
  - Uses sklearn.metrics for classification_report and confusion_matrix.
  - Confusion matrices are normalized for class-wise performance visualization.

## Context & assumptions
- Designed for Python 3.9 with TensorFlow 2.11 compatibility.
- Dataset layout assumed: datasets/myData/<class_id>/*.png and datasets/labels.csv matching ClassId.
- GridSearchCV is run in separate batches and serialized to allow long runs on limited hardware.

## Expected outputs (after a full run)
- X_processed.npy, y_labels.npy — processed arrays
- mlp_model.keras — saved MLP model
- svc_model.pkl — trained final SVM model
- grid1.pkl, grid2.pkl, grid3.pkl — GridSearch checkpoints
- Printed metrics and confusion matrix plots for both MLP and SVM

## Report and Analysis of Two Models:
https://docs.google.com/document/d/1ofvmsX4Tc1w9XthPqT04TtHxleGjXgIhuJNQPUYlUo0/edit?usp=sharing



