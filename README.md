# 🧠 MNIST Neural Network (from Scratch)
A fully-connected neural network built **entirely from scratch using NumPy** — no TensorFlow, PyTorch, or Keras.  
This project trains a simple feedforward model to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## 🚀 Overview
This project demonstrates the core mechanics of a neural network implemented manually:
- Matrix-based **forward** and **backward propagation**
- **ReLU** activation and **Cross-Entropy Loss**
- Mini-batch training with a custom **DataLoader**
- Weight initialization, gradient updates, and learning rate tuning
- Model saving/loading and prediction visualization with Matplotlib  
It was developed as a learning exercise to understand how deep learning works under the hood.


## ⚙️ Requirements
To install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dataset
This project uses the **MNIST** dataset from Kaggle:  
👉 [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
Download the `.idx` files from there and place them in the `data/` folder.

Place the following files inside the **data/** folder:
- train-images.idx3-ubyte
- train-labels.idx1-ubyte
- t10k-images.idx3-ubyte
- t10k-labels.idx1-ubyte

## 🧠 Model Architecture
The default network used in main.py:

| Layer                                                                        | Type  | Input → Output |
| ---------------------------------------------------------------------------- | ----- | -------------- |
| 1                                                                            | Dense | 784 → 256      |
| 2                                                                            | ReLU  | -              |
| 3                                                                            | Dense | 256 → 128      |
| 4                                                                            | ReLU  | -              |
| 5                                                                            | Dense | 128 → 10       |

Each neuron learns via stochastic gradient descent using cross-entropy loss. 



## 🏋️‍♂️ Training
To train a new model:
```bash
python main.py
```

Then choose:
```bash
Train or Not (y/n): y
```
- Trains the model for 12 epochs (by default)
- Saves weights as **mnist_model_1.npz**
- Prints per-epoch losses and accuracies

Example output:
```bash
Final Train Accuracy: 91.87%
Final Test Accuracy : 90.42%
```

## 🔍 Testing / Evaluation
To test a pre-trained model:
```bash
python main.py
```

Then choose:
```bash
Train or Not (y/n): n
```
This loads your saved weights, evaluates accuracy on the MNIST test set, and displays 25 random digit predictions (5×5 grid) with predicted and true labels.

## 📊 Results
With basic tuning (**lr=0.005**, **batch_size=128**, **epochs=12**):
- Training Accuracy: ~92%
- Test Accuracy: ~90%
Each run may vary slightly due to random initialization.

## 🙌 Acknowledgements
- MNIST dataset: Yann LeCun (http://yann.lecun.com/exdb/mnist/) et al.
- Data loader: adapted from Hojjat Khodabakhsh’s Kaggle notebook (https://www.kaggle.com/code/hojjatk/read-mnist-dataset)
- Created for personal learning and educational purposes.
