import numpy as np
import torch
import torch.nn as nn
import tensorflow.keras.datasets.mnist as MNIST

# set random seed for reproducability
torch.manual_seed(4460)
np.random.seed(4460)

# load and display MNIST
(image_MNIST_train_set, label_MNIST_train_set), (image_MNIST_test_set, label_MNIST_test_set) = MNIST.load_data()
print(label_MNIST_test_set.size())