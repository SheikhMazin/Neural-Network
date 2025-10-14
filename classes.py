"""
classes.py
-----------
Core implementation of a fully-connected neural network built from scratch using NumPy.

Includes:
- Dense, ReLU, Sequential, and CrossEntropyLoss classes
- MNISTDataset and DataLoader for batching and normalization
- Trainer class for training, evaluation, and metric tracking

This file defines all key components used by main.py.
"""



import numpy as np

class Dense:
    """
    Fully-connected (linear) layer: y = x @ W + b
    Stores gradients for backprop and updates its own parameters.
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # He initialization for ReLU networks: keeps activations/gradients stable
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        # Bias starts at zero; shape (1, out_features) so broadcasting works during forward
        self.bias = np.zeros((1, out_features), dtype=np.float32)

        # Gradient buffers (same shapes as params)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        # Saved input for backward
        self.input = None

    def forward(self, x):
        """Compute linear transform and cache input for backward."""
        self.input = x                      # shape: (B, in_features)
        return np.dot(x, self.weights) + self.bias  # shape: (B, out_features)

    def backward(self, d_out):
        """
        Backprop through the linear layer.
        d_out: gradient from next layer, shape (B, out_features)
        Returns gradient to pass to previous layer, shape (B, in_features)
        """
        # Gradients w.r.t. parameters
        # dW = X^T @ d_out  -> (in, B) @ (B, out) = (in, out)
        self.grad_weights = np.dot(self.input.T, d_out)
        # db = sum over batch -> keepdims to match (1, out_features)
        self.grad_bias = np.sum(d_out, axis=0, keepdims=True)

        # Gradient to previous layer: dX = d_out @ W^T
        d_input = np.dot(d_out, self.weights.T)
        return d_input

    def update_parameters(self, learning_rate, weight_decay=0.0):
        """
        SGD update with optional L2 weight decay (λ * W).
        W <- W - lr * (dW + λW)
        b <- b - lr * db
        """
        self.weights -= learning_rate * (self.grad_weights + weight_decay * self.weights)
        self.bias    -= learning_rate * self.grad_bias

    def reset_grad(self):
        """Zero the stored gradients (called every batch)."""
        self.grad_weights.fill(0.0)
        self.grad_bias.fill(0.0)

    def parameters(self):
        """Return references to parameters (useful for inspection)."""
        return [self.weights, self.bias]


class ReLU:
    """Element-wise ReLU activation: max(0, x)"""
    def __init__(self):
        self.input = None  # cached for backward mask

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out):
        # Pass gradient only where input was positive
        return d_out * (self.input > 0)


class Sequential:
    """
    Container that chains layers.
    Calls forward/backward/update on each layer in order.
    """
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """Feed input through all layers in order."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        """Backpropagate gradients in reverse order."""
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def update_parameters(self, learning_rate, weight_decay=1e-4):
        """
        Update all layers that have parameters.
        NOTE: pass weight_decay down so L2 regularization actually applies.
        """
        for layer in self.layers:
            if hasattr(layer, "update_parameters"):
                layer.update_parameters(learning_rate, weight_decay=weight_decay)

    def reset_grad(self):
        """Reset grads on all layers that store them."""
        for layer in self.layers:
            if hasattr(layer, "reset_grad"):
                layer.reset_grad()

    def save(self, path="model.npz"):
        """Save weights/biases of all Dense layers into a single .npz file."""
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                data[f"W{i}"] = layer.weights
                data[f"b{i}"] = layer.bias
        np.savez(path, **data)
        print(f"Model saved to {path}")

    def load(self, path="model.npz"):
        """Load weights/biases from a .npz file into matching layers."""
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                layer.weights = data[f"W{i}"]
                layer.bias = data[f"b{i}"]
        print(f"Model loaded from {path}")


class CrossEntropyLoss:
    """
    Cross-entropy loss combined with softmax.
    - forward() returns scalar loss (mean over batch)
    - backward() returns gradient w.r.t. logits, shape (B, num_classes)
    """
    def __init__(self):
        self.predictions = None  # softmax probabilities
        self.targets = None      # integer class labels (B,)
        self.batch_size = None

    def forward(self, predictions, targets):
        """
        predictions: raw logits, shape (B, C)
        targets:     integer labels, shape (B,)
        """
        # Numerically stable softmax: subtract max per row
        exps = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        softmax = exps / np.sum(exps, axis=1, keepdims=True)

        # Cache for backward
        self.predictions = softmax
        self.targets = targets
        self.batch_size = predictions.shape[0]

        # Pick probability of the correct class for each sample
        correct_probs = self.predictions[np.arange(self.batch_size), targets]
        # Cross-entropy = -log(p_correct); add epsilon to avoid log(0)
        loss = -np.log(correct_probs + 1e-9)
        return np.mean(loss)

    def backward(self):
        """
        Gradient of cross-entropy(softmax(logits)) wrt logits:
        grad = softmax - one_hot(target)
        """
        grad = self.predictions.copy()
        rows = np.arange(self.batch_size)
        grad[rows, self.targets] -= 1.0
        grad /= self.batch_size  # mean over batch
        return grad


class MNISTDataset:
    """
    Simple dataset wrapper around MNIST arrays.
    - normalizes images to [0,1]
    - optionally flattens 28x28 -> 784
    """
    def __init__(self, images, labels, normalize=True, flatten=True):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

        if normalize:
            self.images = self.images / 255.0
        if flatten:
            self.images = self.images.reshape(self.images.shape[0], -1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DataLoader:
    """
    Minimal DataLoader:
    - shuffles indices each epoch (if shuffle=True)
    - yields mini-batches of (x, y)
    """
    def __init__(self, dataset, batch_size=64, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._indices = np.arange(len(dataset))  # index list we permute
        self._ptr = 0                            # pointer into indices

    def __len__(self):
        # Number of batches per epoch
        if self.drop_last:
            return len(self._indices) // self.batch_size
        return int(np.ceil(len(self._indices) / self.batch_size))

    def __iter__(self):
        # Start a new epoch: reset pointer and shuffle
        self._ptr = 0
        if self.shuffle:
            np.random.shuffle(self._indices)
        return self

    def __next__(self):
        # Stop when we've consumed the index list
        if self._ptr >= len(self._indices):
            raise StopIteration

        end = self._ptr + self.batch_size
        batch_idx = self._indices[self._ptr:end]
        self._ptr = end

        # Optionally drop the last small batch
        if self.drop_last and len(batch_idx) < self.batch_size:
            raise StopIteration

        x_batch = self.dataset.images[batch_idx]
        y_batch = self.dataset.labels[batch_idx]
        return x_batch, y_batch


class Trainer:
    """
    Handles the training/evaluation loops for a model + loss function.
    """
    def __init__(self, model, loss_fn, lr):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def train_epoch(self, dataloader):
        """Train for one full pass over the dataloader."""
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        num_batches = 0

        for x_batch, y_batch in dataloader:
            # 1) clear grads
            self.model.reset_grad()

            # 2) forward + loss
            logits = self.model.forward(x_batch)
            loss = self.loss_fn.forward(logits, y_batch)

            # 3) loss gradient and backprop through model
            grad = self.loss_fn.backward()
            self.model.backward(grad)

            # 4) parameter update
            self.model.update_parameters(self.lr)

            # 5) metrics
            preds = np.argmax(logits, axis=1)
            total_correct += (preds == y_batch).sum()
            total_seen += y_batch.shape[0]
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = total_correct / max(1, total_seen)
        return avg_loss, accuracy

    def evaluate(self, dataloader):
        """Evaluate (no gradient updates) over the dataloader."""
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        num_batches = 0
        pred_list = []

        for x_batch, y_batch in dataloader:
            logits = self.model.forward(x_batch)
            loss = self.loss_fn.forward(logits, y_batch)

            preds = np.argmax(logits, axis=1)
            pred_list.append(preds)

            total_loss += loss
            total_correct += (preds == y_batch).sum()
            total_seen += y_batch.shape[0]
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = total_correct / max(1, total_seen)
        all_preds = np.concatenate(pred_list, axis=0)
        return avg_loss, accuracy, all_preds

    def fit(self, train_loader, test_loader, epochs):
        """Run multiple epochs of training + evaluation; collect metrics."""
        total_train_loss = []
        total_train_acc = []
        total_test_loss = []
        total_test_acc = []
        total_preds = []

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc, preds = self.evaluate(test_loader)

            total_train_loss.append(train_loss)
            total_train_acc.append(train_acc)
            total_test_loss.append(test_loss)
            total_test_acc.append(test_acc)
            total_preds.append(preds)

        return {
            "train_losses": total_train_loss,
            "train_accuracies": total_train_acc,
            "test_losses": total_test_loss,
            "test_accuracies": total_test_acc,
            "predictions": total_preds,
        }
