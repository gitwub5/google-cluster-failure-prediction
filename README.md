
# LSTM and Transformer Models for Failure Prediction

This repository contains implementations of LSTM and Transformer models for failure prediction using time-series data. The models are designed to process machine-level data with temporal sequences and event embeddings, aiming to predict failures effectively.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Preprocessing](#data-preprocessing)
3. [LSTM Model](#lstm-model)
4. [Transformer Model](#transformer-model)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Model Saving and Loading](#model-saving-and-loading)

---

## Overview

The primary goal of these models is to predict failures (`Failed` label) in a machine dataset based on its features and event types. The dataset includes time-series data with the following key features:

- **Features**: `average_usage_cpus`, `average_usage_memory`, `maximum_usage_cpus`, `maximum_usage_memory`
- **Target**: `Failed` (binary classification)

The models use:
- **LSTM (Long Short-Term Memory)**: To capture temporal dependencies in sequences.
- **Transformer**: To leverage self-attention mechanisms for efficient sequence modeling.

---

## Data Preprocessing

1. Ensure the dataset contains necessary columns:
   - `average_usage_cpus`, `average_usage_memory`, `maximum_usage_cpus`, `maximum_usage_memory`
   - `event_type` and `Failed` (target)
   - `machine_id` for machine-specific sequences
2. Event types are converted into categorical indices for embedding (`event_type_idx`).
3. Min-Max scaling is applied to the features for normalization.

---

## LSTM Model

### Architecture
- **Input**: Features concatenated with event type embeddings.
- **Core**: LSTM layers to process sequences.
- **Output**: Fully connected layer for regression or classification.

### Training
- **Loss Function**: MSELoss for regression tasks or BCEWithLogitsLoss for binary classification.
- **Optimizer**: Adam optimizer.

### Code Usage
```python
from lstm_model import LSTMModel, train_model, evaluate_model

# Initialize LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size, num_event_types, event_embedding_dim)
train_model(model, dataloader, criterion, optimizer, num_epochs, device)
evaluate_model(model, dataloader, device)
```

---

## Transformer Model

### Architecture
- **Input**: Features concatenated with event type embeddings.
- **Core**: Transformer encoder with multiple attention heads.
- **Output**: Fully connected layer for classification.

### Training
- **Loss Function**: BCEWithLogitsLoss for binary classification.
- **Optimizer**: Adam optimizer.

### Code Usage
```python
from transformer_model import TransformerModel, train_model, evaluate_model

# Initialize Transformer model
model = TransformerModel(input_size, embedding_dim, num_heads, num_layers, output_size, num_event_types)
train_model(model, dataloader, criterion, optimizer, num_epochs, device)
evaluate_model(model, dataloader, device)
```

---

## Model Training

1. Use PyTorch's `DataLoader` for efficient batch processing.
2. Training loop includes:
   - Forward pass through the model.
   - Loss computation using `criterion`.
   - Backward pass and optimization.

---

## Evaluation

### Metrics
- **Accuracy**: Fraction of correct predictions.
- **Precision**: True positive rate among predicted positives.
- **Recall**: True positive rate among actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

---

## Model Saving and Loading

### Saving a Model
```python
torch.save(model.state_dict(), 'model_path.pth')
```

### Loading a Model
```python
model.load_state_dict(torch.load('model_path.pth'))
```

---

## Notes

- Ensure the same preprocessing steps are applied to both training and testing datasets.
- Use GPU for faster training, if available.
- Experiment with hyperparameters (e.g., learning rate, sequence length, embedding size) to optimize performance.

---

## References
- [PyTorch Documentation](https://pytorch.org/docs/)
- Research papers and resources on LSTM and Transformer architectures.