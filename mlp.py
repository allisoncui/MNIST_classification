import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

import tensorflow.keras.datasets.mnist as MNIST

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

# set random seed for reproducability
torch.manual_seed(4460)
np.random.seed(4460)

# load and display MNIST
(image_MNIST_train_set, label_MNIST_train_set), (image_MNIST_test_set, label_MNIST_test_set) = MNIST.load_data() # all images are represented by a 2d array

# display sample images and labels (optional)
row = 4
col = 4
fig = plt.figure(figsize = (12,15)) # creates new figure from matplotlib, figsize specifies size of image. figure object can plot multiple subplots
for img_index in range(1, row*col+1): # loop iterates from 1 all the way to row * col (last index is excluded)
    image = image_MNIST_test_set[img_index, :, :] # first dimension is indexed with img_index, : indicates we want to include all elements along remaining two dimensions (rows and columns)
    ax = fig.add_subplot(row, col, img_index) # creates subplot within figure 'fig', row and col indicate number of rows and columns of subplots in the figure, img_index determines position of current subplot within grid of subplots
    ax.set_xticks([]) # remove ticks on subplots
    ax.set_yticks([])
    ax.title.set_text("Label: " + str(label_MNIST_test_set[img_index])) # sets title of current subplot 
    plt.imshow(image, cmap='gray') # displays image on current subplot
plt.show() # display all active figures

# split into training, validation, test using train_test_split from scikit-learn (using what MNIST originally allocated as the 10k test set as our entire dataset)
indices_train, indices_else = train_test_split(range(len(image_MNIST_test_set)), test_size=0.2) # 80% train, 20% validation and test

