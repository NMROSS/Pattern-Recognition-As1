# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as torchvision
from torch.optim import *
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale

BATCH_SIZE = 100
NUM_ITERS = 2500
LEARNING_RATE = 0.001
csv = False # True for csv data, False for permutated png data
permutated = True # Only relevant if csv = True. If True, the permutated data is loaded, otherwise the normal data

if csv :
    # Load train dataset
    dataset_train = pd.read_csv('mnist_train.csv', delimiter=',', header=None)
    data_train = dataset_train.values
    images_train, labels_train = data_train[:,1:], data_train[:, 0]

    # Load test dataset
    dataset_test = pd.read_csv('mnist_test.csv', delimiter=',', header=None)
    data_test = dataset_test.values
    images_test, labels_test = data_test[:,1:], data_test[:, 0]

    # Formatting of train dataset
    images_train = images_train.reshape(60000, 1, 28, 28)
    images_train = torch.from_numpy(images_train).float()
    labels_train = torch.from_numpy(np.array(labels_train))

    # Formatting of test dataset
    images_test = images_test.reshape(10000, 1, 28, 28)
    images_test = torch.from_numpy(images_test).float()
    labels_test = torch.from_numpy(np.array(labels_test))

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(images_train, labels_train)
    test = torch.utils.data.TensorDataset(images_test, labels_test)

    # Pytorh data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

else:
    # choose normal or permutated data set
    if permutated:
        train_set = 'mnist-png-format-permutated/train'
        test_set = 'mnist-png-format-permutated/test'
    else:
        train_set = 'mnist-png-format/train'
        test_set = 'mnist-png-format/test'

    workers = 0

    # load images
    transforms = Compose([
        Grayscale(num_output_channels=1),  # PNG file is RGB 3 layer convert to greyscal 1 layer
        ToTensor(),
        Normalize(mean=(0.5), std=(0.5)),
    ])

    train_dataset = ImageFolder(train_set, transform=transforms)
    test_dataset = ImageFolder(test_set, transform=transforms)

    # Split Train dataset set into two (Train=85/Validation=15% split)
#    train_size = int(len(train_dataset) * 0.85)
#    validation_size = (len(train_dataset) - train_size)
#    train, validation = random_split(train_dataset, [train_size, validation_size])

    # load data into usable format, mix/shuffle data so data is not in order
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
#    val_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=workers, pin_memory=True)

# Simple feed forward convolutional neural network
class PR_CNN(nn.Module):

    def __init__(self, **kwargs):

        # Creates an CNN_basic model from the scratch
        super(PR_CNN, self).__init__()
        # Set 1
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu_1 = nn.LeakyReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        # Set 2
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu_2 = nn.LeakyReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected
        self.fc = nn.Linear(32 * 5 * 5, 10)

    # Computes forward pass on the network
    def forward(self, x):

        # Set 1
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.pool_1(out)
        # Set 2
        out = self.conv_2(out)
        out = self.relu_2(out)
        out = self.pool_2(out)
        # Flatten
        out = out.view(out.size(0), -1)
        # Fully connected
        out = self.fc(out)

        return out


# Create model
model = PR_CNN()
# Cross entropy loss
error = nn.CrossEntropyLoss()
# SGH optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
training_size = len(images_train) if csv else len(train_dataset)
epochs = int(NUM_ITERS / (training_size / BATCH_SIZE))

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and entropy loss
        loss = error(outputs, labels)
        # Calculate gradients
        loss.backward()
        # Update parameters
        optimizer.step()

        count += 1
        if count % 50 == 0:
            # Calculate accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                test = Variable(images.view(100,1,28,28))
                # Forward propagation
                outputs = model(test)
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()
            accuracy = 100 * correct / float(total)
            # Store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

total = 0
correct = 0
for images, labels in train_loader:
    train = Variable(images.view(100,1,28,28))
    outputs = model(train)
    predicted = torch.max(outputs.data, 1)[1]
    total += len(labels)
    correct += (predicted == labels).sum()
    accuracy_train = 100 * correct / float(total)
print('Training accuracy: {}, Test accuracy: {}'.format(accuracy_train, accuracy))

# Visualization loss
fig = plt.figure(figsize=(10,5))
plt.figure(1)
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of Iterations")
fig.savefig('Loss.jpg', bbox_inches='tight', dpi=150)

# Visualization accuracy
fig = plt.figure(figsize=(10,5))
plt.figure(2)
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of Iterations")
fig.savefig('Accuracy.jpg', bbox_inches='tight', dpi=150)
plt.show()