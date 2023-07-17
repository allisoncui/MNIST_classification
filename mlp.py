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

torch.manual_seed(4460)
np.random.seed(4460)


# LOAD AND DISPLAY MNIST
(image_MNIST_train_set, label_MNIST_train_set), (image_MNIST_test_set, label_MNIST_test_set) = MNIST.load_data() # images are represented by a 2d array


# DATA PREPROCESSING
# split into training, validation, test using train_test_split
indices_train, indices_else = train_test_split(range(len(image_MNIST_test_set)), test_size=0.2) # 80% train, 20% validation and test

image_train = image_MNIST_test_set[indices_train, :, :] # index, row, column
label_train = label_MNIST_test_set[indices_train]
image_else = image_MNIST_test_set[indices_else, :, :]
label_else = label_MNIST_test_set[indices_else]

indices_validation, indices_test = train_test_split(range(len(image_else)), test_size=0.5)
image_validation = image_else[indices_validation, :, :]
label_validation = label_else[indices_validation]
image_test = image_else[indices_test, :, :]
label_test = label_else[indices_test]

# reformat images and labels so they can be fed into data loader, convert numpy array to pytorch tensor + reshape 
image_train_torch = torch.from_numpy(image_train).type(torch.FloatTensor).view(-1,1,28,28) # -1 means size of that dimension will be automatically inferred, 1 along second dim (single channel image), height and width of 28x28 pixels
label_train_torch = torch.from_numpy(label_train).type(torch.LongTensor)

image_validation_torch = torch.from_numpy(image_validation).type(torch.FloatTensor).view(-1,1,28,28) # 1x28x28
label_validation_torch = torch.from_numpy(label_validation).type(torch.LongTensor)

image_test_torch = torch.from_numpy(image_test).type(torch.FloatTensor).view(-1,1,28,28) 
label_test_torch = torch.from_numpy(label_test).type(torch.LongTensor)

# dataset and dataloader
train_dataset = TensorDataset(image_train_torch, label_train_torch)   # TensorDataset (input data + corresponding labels)
train_dataloader = DataLoader(train_dataset, batch_size=1000)         # DataLoader for batching data

val_dataset = TensorDataset(image_validation_torch, label_validation_torch)
val_dataloader = DataLoader(val_dataset, batch_size=1000)

test_dataset = TensorDataset(image_test_torch, label_test_torch)
test_dataloader = DataLoader(test_dataset, batch_size=1000)


# MLP MODEL IMPLEMENTATION
'''
sets up basic MLP model with 3 FC layers and sigmoid activation layers between each layer, culminates in log softmax activation for classification
reshape -> FC -> sigmoid -> FC -> sigmoid -> FC -> softmax -> cross entropy
'''
class MLPModel(nn.Module):
    def __init__(self):                     # models' modules
        super(MLPModel, self).__init__()    # base class in pytorch for defining neural network models
        self.fc1 = nn.Linear(28*28, 128)    # (input features, output features), linear transformation layer in nn
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):       # model architecture
        x = x.view(-1, 28*28)   # reshape input tensor x
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)
    
our_MLP = MLPModel()

optimizer = SGD(our_MLP.parameters(), lr = 0.1)


# TRAIN MLP MODEL FOR CLASSIFICATION
'''
1. define number of epochs 
2. define lists to keep train and validation losses at end of each epoch
3. iterate over each epoch for training the model
4. define list to keep track of loss for every epoch iteration
5. set model to train mode so parameters can be updated
6. training loop -- iterate over batches of training data using train_loader
7. use model to predict labels for input data
8. compute loss
'''

EPOCHS = 400
train_epoch_loss = []
val_epoch_loss = []

for epoch in range(EPOCHS):
    train_loss = []
    
    # TRAINING
    our_MLP.train()
    
    for batch_index, (train_image, train_label) in enumerate(train_dataloader):
        train_label_predicted = our_MLP(train_image)

        loss = F.cross_entropy(train_label_predicted, train_label)
        train_loss.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_epoch_loss.append(np.mean(train_loss))

    # VALIDATION
    our_MLP.eval()
    
    for batch_index, (val_image, val_label) in enumerate(val_dataloader):
        val_label_predicted = our_MLP(val_image)
        
        loss = F.cross_entropy(val_label_predicted, val_label)
        val_epoch_loss.append(loss.data.item())
        
    torch.save(our_MLP.state_dict(), './saved_MLP_models/checkpoint_epoch_%s.pth' % (epoch)) 
        

# VALIDATE DATA
for batch_index, (train_image, train_label) in enumerate(train_dataloader):
    train_label_predicted = our_MLP(train_image)

# check which epoch model performs best on validation set, use for inference on test set
best_epoch = np.argmin(val_epoch_loss)
print(best_epoch)
state_dict = torch.load('./saved_MLP_models/checkpoint_epoch_%s.pth' % (best_epoch))
our_MLP.load_state_dict(state_dict)


# PREDICT LABELS ON TEST SET
def predict_with_pytorch(model, input_data):
    model.eval()
    label_predicted_all = []

    label_predicted_one_hot = model(input_data)
    label_predicted_probability, label_predicted_index = torch.max(label_predicted_one_hot.data, 1)
    
    for current_prediction in label_predicted_index:
        label_predicted_all.append(current_prediction.detach().cpu().numpy().item())

    return label_predicted_all

test_label_predicted = predict_with_pytorch(our_MLP, image_test_torch)
accuracy = accuracy_score(label_test, test_label_predicted)
print("Accuracy:", accuracy * 100, "%")