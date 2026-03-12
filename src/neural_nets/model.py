import numpy as np
from modules import *
from optim import *
from loss import *
from dataloaders import DataLoader

# Initialize random num generator
rng = np.random.default_rng()

# For getting mnist model
import tensorflow as tf

num_classes = 10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Flatten the data + make labels one-hot
train_images: np.ndarray = train_images.reshape(60000, -1)
test_images: np.ndarray = test_images.reshape(10000, -1)

# Print the shape of the data to verify
print("Training data shape:", train_images.shape)
print("Testing data shape:", test_images.shape)

train_labels = np.eye(num_classes)[train_labels].astype(np.int8)
test_labels = np.eye(num_classes)[test_labels].astype(np.int8)

# Declare hyperparameters
epochs = 20
learning_rate = 0.001
input_size = 784
output_size = 10
batch_size = 256

# Initialize dataloaders
train_loader = DataLoader(train_images, train_labels, batch_size)

# Define the model, optimizer, loss
modules = [Linear(input_size, 128), ReLU(), Linear(128, output_size)]
loss_fn = CrossEntropyLoss(modules)
optim = Adam(modules, learning_rate=learning_rate, weight_decay=0.001)

# Train the model
for epoch in range(epochs):
    print(f'Training epoch {epoch + 1}: ', end="")
    train_loss = 0.0

    # Load the batch
    while train_loader.next():
        data = train_loader.data
        labels = train_loader.labels

        # Forward propogate
        for module in modules:
            data = module.forward(data)

        loss = loss_fn.compute(data, labels)

        # Backpropogation
        optim.zero_grad()
        loss_fn.backward()
        optim.step()

        # Keep track of total loss
        train_loss += loss.sum()

    # Report loss & reset batch
    print(f'Total Loss: {train_loss}')
    train_loader.reset()

# Check accuracy of model using ENTIRE test dataset
preds = test_images[:]
for module in modules:
    preds = module.predict(preds)

# Get predictions (in one_hot form)
pred_indices = np.argmax(preds, axis=1)
true_indices = np.argmax(test_labels, axis=1)

accuracy = np.mean(pred_indices == true_indices)

print(f'Test Accuracy: {accuracy:.2%}')
