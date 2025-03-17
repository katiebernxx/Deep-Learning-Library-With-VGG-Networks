# Deep Learning Library with VGG Networks

## Overview

This project focuses on training deep neural networks to achieve high classification accuracy on the CIFAR-10 dataset using VGG-style convolutional networks. It involves implementing various deep learning techniques, leveraging GPU acceleration on cloud compute servers, and developing a custom deep learning library using low-level TensorFlow. The project also explores the scalability and effectiveness of different VGG architectures.
Acheivement: >87% accuracy on CIFAR dataset

## Objectives

* Implement a deep learning library from scratch using TensorFlow's low-level API.

* Explore and train different VGG architectures (VGG4, VGG6, VGG8, VGG15, VGG16, and variations with batch normalization and dropout).

* Optimize training with techniques such as weight initialization, early stopping, regularization, dropout, batch normalization, and learning rate decay.

* Achieve high classification accuracy on the CIFAR-10 dataset.

* Utilize GPU-based cloud computing to train large neural networks efficiently.

* Compare performance across varying VGG depths and configurations.

* Maintain and enhance a large software project over the course of a semester.

## Implementation Details

### 1. Deep Learning Library

The project includes a deep learning library that implements:

* Layer Abstractions (e.g., Dense, Conv2D, MaxPool2D, Dropout, Flatten)

* Block Structures for modularizing convolutional and dense layers (e.g., VGGConvBlock, VGGDenseBlock)

* Neural Network Architecture Management via DeepNetwork base class

* Training and Evaluation Functions including backpropagation, loss computation, optimizer handling, and early stopping

* Dataset Handling with preprocessing and standardization functions

### 2. VGG Architectures

The following VGG networks were implemented:

* VGG4: A small architecture with two convolutional layers and a dense classifier.

* VGG6: A deeper variant with two convolutional blocks.

* VGG8: A three-block structure adding complexity.

* VGG15 & VGG16: Standard VGG architectures with multiple convolutional blocks and fully connected layers.

* VGG4Plus, VGG15Plus, VGG15PlusPlus: Enhanced versions with batch normalization and dropout for improved regularization.

### 3. Training and Evaluation

The networks were trained on CIFAR-10 with the AdamW optimizer and batch normalization to stabilize training. Various configurations were tested to analyze model performance across different depths.

## Key Features

* Custom Neural Network API: Provides reusable blocks for building deep networks.

* VGG Family Exploration: Implements and benchmarks various VGG architectures.

* Optimization Techniques: Uses weight initialization, dropout, batch normalization, and early stopping.

* Cloud Compute with GPU Acceleration: Trains models efficiently on cloud-based GPUs.

* Scalability: Allows for easy extension of architectures and hyperparameter tuning.

## Results

The project compares different VGG architectures on CIFAR-10, analyzing accuracy and loss trends. Training logs, loss curves, and validation accuracy trends are used to determine the best-performing models.
Achieved > 87% accuracy on CIFAR

## Running the Project

Dependencies

Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib

### Installation

Clone the repository and install dependencies:

pip install -r requirements.txt

Training a Model

To train a model (e.g., VGG15Plus) on CIFAR-10:

```python
from datasets import get_dataset
from vgg_nets import VGG15Plus

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test, classnames = get_dataset('cifar10')

# Initialize and compile model
model = VGG15Plus(C=10, input_feats_shape=(32, 32, 3))
model.compile(optimizer='adamw')

# Train model
train_loss_hist, val_loss_hist, val_acc_hist, epochs_run = model.fit(x_train, y_train, x_val, y_val, max_epochs=50)

# Evaluate on test set
test_acc, test_loss = model.evaluate(x_test, y_test)
print(f"VGG15Plus CIFAR-10 Test accuracy: {test_acc.numpy()*100:.2f}%")
```

## Performance Analysis

To compare different architectures:
```python
from vgg_nets import VGG4Plus, VGG15, VGG15Plus, VGG15PlusPlus

models = [(VGG4Plus, 'VGG4Plus'), (VGG15, 'VGG15'), (VGG15Plus, 'VGG15Plus'), (VGG15PlusPlus, 'VGG15PlusPlus')]

for model_class, model_name in models:
    model = model_class(C=10, input_feats_shape=(32, 32, 3))
    model.compile(optimizer='adamw')
    train_loss_hist, val_loss_hist, val_acc_hist, epochs_run = model.fit(x_train, y_train, x_val, y_val, patience=4)
    test_acc, test_loss = model.evaluate(x_test, y_test)
    print(f"{model_name} CIFAR-10 Test accuracy: {test_acc.numpy()*100:.2f}%")
```
## Conclusion

This project successfully developed a deep learning library and trained multiple VGG architectures on CIFAR-10. The modular library structure facilitates further research into deep learning techniques and optimizations.

