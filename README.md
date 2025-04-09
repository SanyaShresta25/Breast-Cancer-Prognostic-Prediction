# Breast Cancer Classification using Neural Networks

## Overview

This project implements a neural network classifier to predict breast cancer diagnosis (malignant or benign) based on features computed from digitized images of fine needle aspirates (FNA) of breast masses. The model uses hyperparameter tuning with Keras Tuner to optimize the network architecture and training parameters.

## Dataset

The dataset used is the Wisconsin Breast Cancer Diagnostic dataset, containing the following features:
- 30 numeric features computed from digitized images
- Target variable: diagnosis (M = malignant, B = benign)

Dataset preprocessing includes:
- Removal of unnecessary columns ('id' and 'Unnamed: 32')
- Train-test split (80-20)
- Standard scaling of numeric features
- Label encoding of the target variable

## Model Architecture

The neural network architecture is optimized using Keras Tuner's RandomSearch with the following search space:

- Number of layers: 1-5
- Units per layer: 32-512 (in steps of 32)
- Activation functions: relu, tanh, or sigmoid
- Kernel initializers: glorot_uniform, he_normal, or lecun_normal
- Kernel regularizers: L1 or L2
- Dropout rate: 0.2-0.5
- Optimizers: adam, rmsprop, or sgd

The final output layer uses a sigmoid activation for binary classification.

## Training

The model training includes:
- Early stopping callback to prevent overfitting
- 100 maximum epochs
- Validation on 20% test set
- Binary cross-entropy loss function

## Results

The model evaluation includes:
- Training and validation accuracy/loss curves
- Final evaluation metrics on test set:
  - Loss
  - Accuracy

## Requirements

To run this project, you'll need:
- Python 3.x
- TensorFlow (version shown in output)
- Keras Tuner
- scikit-learn
- pandas
- matplotlib

## Installation

```bash
pip install tensorflow keras-tuner scikit-learn pandas matplotlib
```

## Usage

1. Ensure the dataset file 'breast_cancer.csv' is in the correct path
2. Run the entire notebook/script
3. View the training progress and final evaluation metrics
4. The best model architecture and parameters will be displayed

## Files

- `breast_cancer_classification.ipynb` (or `.py`): Main implementation file
- `breast_cancer.csv`: Input dataset

## Visualization

The project includes visualizations of:
1. Training vs Validation Accuracy
2. Training vs Validation Loss
3. Validation Accuracy vs Epochs
4. Validation Loss vs Epochs

## Future Improvements

Potential enhancements:
- Feature importance analysis
- Additional feature engineering
- Experiment with different neural network architectures
- Cross-validation for more robust evaluation
- Class imbalance handling if needed
