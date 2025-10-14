"""
main.py
--------
Entry point for training and evaluating the MNIST neural network.

- Loads MNIST data via MnistDataloader
- Builds model architecture (Sequential with Dense + ReLU layers)
- Either trains a new model or loads a saved one
- Displays performance metrics and sample predictions

Usage:
  Run and choose:
    "y" → Train and save a new model
    "n" → Load an existing saved model and test it
"""

from read_data import MnistDataloader, train_images_fp, train_labels_fp, test_labels_fp, test_images_fp
from classes import Dense, ReLU, CrossEntropyLoss, Sequential, DataLoader, MNISTDataset, Trainer
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load MNIST Data (images + labels)
# -------------------------------
loader = MnistDataloader(train_images_fp, train_labels_fp, test_images_fp, test_labels_fp)
(x_train, y_train), (x_test, y_test) = loader.load_data()

# -------------------------------
# Dataset preprocessing
# Normalize pixel values (0–255 → 0–1) and flatten 28×28 images → 784 features
# -------------------------------
train_ds = MNISTDataset(x_train, y_train, normalize=True, flatten=True)
test_ds  = MNISTDataset(x_test,  y_test,  normalize=True, flatten=True)

# -------------------------------
# Data loaders for batching
# -------------------------------
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# -------------------------------
# Define loss function
# -------------------------------
loss_fn = CrossEntropyLoss()

# -------------------------------
# Ask user whether to train or load
# -------------------------------
x = input("Train or Not (y/n): ")

# -------------------------------
# Define model architecture
# -------------------------------
model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10)
])

# ==========================================================
# TRAINING BRANCH
# ==========================================================
if x.lower() == "y":
    # Initialize trainer with model, loss, and learning rate
    trainer = Trainer(model, loss_fn, lr=0.005)

    # Train for 12 epochs
    results = trainer.fit(train_loader, test_loader, epochs=12)

    # Save trained model
    model.save("mnist_model_1.npz")

    # Round printed metrics for readability
    train_loss = np.round(results["train_losses"], 4)
    train_acc  = np.round(results["train_accuracies"], 4)
    test_loss  = np.round(results["test_losses"], 4)
    test_acc   = np.round(results["test_accuracies"], 4)

    # Display per-epoch results
    print("\nTrain Loss:", train_loss)
    print("Train Acc :", train_acc)
    print("Test Loss :", test_loss)
    print("Test Acc  :", test_acc)

    # Print final accuracies
    print(f"\nFinal Train Accuracy: {train_acc[-1]*100:.2f}%")
    print(f"Final Test Accuracy : {test_acc[-1]*100:.2f}%")

    # Use final-epoch predictions for visualization
    final_preds = results["predictions"][-1]

# ==========================================================
# TESTING / INFERENCE BRANCH
# ==========================================================
else:
    # Load pretrained model weights
    model.load("mnist_model_1.npz")

    # Forward pass on all test images
    logits = model.forward(test_ds.images)           # shape: (10000, 10)
    final_preds = np.argmax(logits, axis=1)          # predicted digits (0–9)

    # Compute test accuracy
    final_test_acc = (final_preds == y_test).mean() * 100
    print(f"Model accuracy on test set: {final_test_acc:.2f}%")

    # -------------------------------
    # Display 25 random predictions (5×5 grid)
    # -------------------------------
    rng = np.random.default_rng()
    idxs = rng.choice(len(y_test), size=25, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for ax, i in zip(axes.flat, idxs):
        ax.imshow(x_test[i], cmap='gray')
        pred = int(final_preds[i])
        true = int(y_test[i])
        ax.set_title(f"{i}\nP:{pred} | T:{true}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
