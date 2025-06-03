import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data (user-provided format)
df = pd.read_csv("data/domain_dataset_cleaned.csv")

# Features and label
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values.astype(np.float32)

# Normalize input for better convergence
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define Shallow Neural Network as per MADONNA
model = Sequential(
    [
        Input(shape=(X.shape[1],)),
        Dense(28, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Training with early stopping
callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2,
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# ======= Predict =======
y_pred = (model.predict(X_test) > 0.5).astype(int)

# ======= Classification Report =======
print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# ======= Confusion Matrix =======
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("data/confusion_matrix.png")
plt.show()

# ======= Metrics Bar Plot =======
# Extract metrics
metrics_to_plot = ["precision", "recall", "f1-score"]
macro_scores = [report["macro avg"][m] for m in metrics_to_plot]
weighted_scores = [report["weighted avg"][m] for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
bar_width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - bar_width / 2, macro_scores, bar_width, label="Macro Avg")
plt.bar(x + bar_width / 2, weighted_scores, bar_width, label="Weighted Avg")
plt.xticks(x, metrics_to_plot)
plt.ylim(0, 1)
plt.title("Evaluation Metrics")
plt.legend()
plt.tight_layout()
plt.savefig("data/metrics_barplot.png")
plt.show()

# ======= Dataset Size Info =======
unique, counts = np.unique(y_test, return_counts=True)
print(f"Dataset distribution in test set: {dict(zip(unique.astype(int), counts))}")

# Save model
model.save("model_13feat.keras")
