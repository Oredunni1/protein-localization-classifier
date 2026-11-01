# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Investigate the dataset
yeast = pd.read_csv("yeast.csv")
print(yeast.head())

# check the info
yeast.info()

# check the shape
print(yeast.shape)

# check the missing values
print(yeast.isnull().sum())

# Data Preparation
# 1. Seperate the features and target
X = yeast.drop(columns=["localization_site"])
y = yeast["localization_site"]

# 2 . Encode the target Labels as integers (e.g MIT -> 0...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Train/Test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. Standardize features
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# 5. Convert to Pytorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
y_test = torch.tensor(y_test_np, dtype=torch.long)

# 6. check dimensions
# print("Train set size: ", X_train.shape)
# print("Test set size: ", X_test.shape)
# print("Number of classes: ", len(label_encoder.classes_))

# Model Architecture

# Define the model using nn.Sequential
input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = nn.Sequential (
    nn.Linear(input_dim, 16), # input layer to hidden layer
    nn.ReLU(),        # non-linearity
    nn.Linear(16, num_classes) # Hidden layer to output layer
)

print(model)

# Training Loop

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 200
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    # forward pass
    logits = model(X_train)
    loss = criterion(logits, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record Loss
    train_losses.append(loss.item())

    # Compute training accuracy
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y_train).sum().item()
    accuracy = correct / len(y_train)
    train_accuracies.append(accuracy)

    # print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy*100:2f}%")


# Track and Plot Training Metrics

# Plot training loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='--')
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)

# plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, marker='s', linestyle='--', color="green")
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()

# Model Evaluation on Test Set

# ----------Predict on test data----------
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test)
    y_pred_labels = torch.argmax(y_pred_logits, dim=1).cpu().numpy()

# ---------Convert ground truth to numpy---------------
y_test_np = y_test.cpu().numpy()

# ---------Accuracy-------
acc = accuracy_score(y_test_np, y_pred_labels)
print(f"Test Accuracy: {acc:.4f}")

# ---------Confusion Matrix --------
cm = confusion_matrix(y_test_np, y_pred_labels)

# Since we used labelEncoder on the original label
label_encoder.fit_transform(yeast["localization_site"])
class_names = label_encoder.classes_

# Display with the class names
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix on Test Set")
plt.show()