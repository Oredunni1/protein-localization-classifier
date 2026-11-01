# Yeast Protein Localization Site Classifier

A PyTorch neural network that classifies yeast proteins into their subcellular localization sites using the UCI Yeast dataset.

## 📊 Dataset
- **Source**: UCI Machine Learning Repository — Yeast dataset
- **Features**: 8 numerical features (mcg, gvh, alm, mit, erl, pox, vac, nuc)
- **Target**: `localization_site` — 10 classes
- **File**: `yeast.csv`

## 🧠 Model
A simple feed-forward neural network:
- Input: 8 → Hidden: 16 (ReLU) → Output: 10 (softmax via CrossEntropyLoss)
- Optimizer: Adam (lr = 0.01)
- Loss: CrossEntropyLoss
- Epochs: 200

## 🚀 How to Run
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn torch matplotlib

# 2. Run the training script
python yeast.py

## 📈 Output
- Training loss and accuracy curves
- Test set accuracy
- Confusion matrix with class labels

## 🛠️ Tech Stack
- Python 3.12
- PyTorch
- scikit-learn
- pandas / numpy
- matplotlib