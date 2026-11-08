"""
CLA-3 Case Study: Iris Flower Species Classification Using kNN
Author: Nyshita Edpuganti (AP24110011324)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
iris = load_iris()
X = iris.data          # shape (150, 4)
y = iris.target        # 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset loaded successfully")
print(f"Total samples : {X.shape[0]}")
print(f"Features      : {feature_names}")
print(f"Classes       : {list(target_names)}")
print()

# -------------------------------------------------
# 2. Train / Test Split (70% train, 30% test)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y          # keep class proportions same
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print()

# -------------------------------------------------
# 3. Train k-Nearest Neighbors Classifier
# -------------------------------------------------
# k = 5 is a common default; odd number avoids ties
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train, y_train)

print("Model trained: KNeighborsClassifier(n_neighbors=5)")
print()

# -------------------------------------------------
# 4. Predictions & Evaluation
# -------------------------------------------------
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set : {accuracy:.4f} ({accuracy*100:.2f}%)")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

# -------------------------------------------------
# 5. Show one correct and one incorrect prediction
# -------------------------------------------------
# Correct example (a setosa)
correct_idx = np.where((y_test == 0) & (y_pred == 0))[0][0]
print("=" * 50)
print("CORRECT PREDICTION EXAMPLE")
print("=" * 50)
print(f"True label      : {target_names[y_test[correct_idx]]}")
print(f"Predicted label : {target_names[y_pred[correct_idx]]}")
print(f"Features        : {dict(zip(feature_names, X_test[correct_idx]))}")
print()

# Incorrect example (the single misclassification)
mis_idx = np.where(y_pred != y_test)[0]
if len(mis_idx) > 0:
    i = mis_idx[0]
    print("=" * 50)
    print("INCORRECT PREDICTION EXAMPLE")
    print("=" * 50)
    print(f"True label      : {target_names[y_test[i]]}")
    print(f"Predicted label : {target_names[y_pred[i]]}")
    print(f"Features        : {dict(zip(feature_names, X_test[i]))}")
    print()
else:
    print("No misclassifications found.")

# -------------------------------------------------
# 6. Optional: Visualize Confusion Matrix
# -------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Iris kNN (k=5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved as confusion_matrix.png")
plt.close()

print("\nDone.")
