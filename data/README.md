# 📂 Data Folder

This folder is used to store the **MNIST dataset files** required for training and testing the neural network.

## 🧠 Dataset Overview
The project uses the classic **MNIST handwritten digits** dataset, which contains:
- **60,000 training images** (`train-images.idx3-ubyte`)
- **60,000 training labels** (`train-labels.idx1-ubyte`)
- **10,000 test images** (`t10k-images.idx3-ubyte`)
- **10,000 test labels** (`t10k-labels.idx1-ubyte`)

Each image is a **28×28 grayscale digit (0–9)**.

## 📥 Download Instructions
The dataset is **not included** in this repository to keep the size small.  
You can download the files directly from the Kaggle page:

🔗 [https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

After downloading, place the following four files in this folder

## ⚙️ Usage
When you run `main.py`, the `read_data.py` script automatically loads these files:
```python
loader = MnistDataloader(train_images_fp, train_labels_fp, test_images_fp, test_labels_fp)
(x_train, y_train), (x_test, y_test) = loader.load_data()
```

Make sure the filenames match exactly, otherwise the loader will raise a FileNotFoundError.

Note:
These dataset files are provided by Yann LeCun et al. and are publicly available for research and educational purposes.