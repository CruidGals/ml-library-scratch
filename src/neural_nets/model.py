import numpy as np
import time
from modules import *
from optim import *
from loss import *
from dataloaders import DataLoader
from util import *

# Initialize random num generator
rng = np.random.default_rng()

# For getting mnist model
import tensorflow as tf

num_classes = 10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Scale pixel values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape to (batch_size, channels, height, width) for Conv2D
train_images: np.ndarray = train_images.reshape(60000, 1, 28, 28)
test_images: np.ndarray = test_images.reshape(10000, 1, 28, 28)

# Print the shape of the data to verify
print("Training data shape:", train_images.shape)
print("Testing data shape:", test_images.shape)

train_labels = np.eye(num_classes)[train_labels].astype(np.int8)
test_labels = np.eye(num_classes)[test_labels].astype(np.int8)

# Declare hyperparameters
epochs = 100
learning_rate = 0.0005
stop_factor_lr = 0.000001
output_size = 10
batch_size = 256

# Initialize dataloaders
train_loader = DataLoader(train_images, train_labels, batch_size)

# Define the LeNet-style model
# Input: 28x28x1 
# Conv2D(1->6, 5x5) -> ReLU -> MaxPool2D(2x2) = 12x12x6
# Conv2D(6->16, 5x5) -> ReLU -> MaxPool2D(2x2) = 4x4x16 (256 features)
# Flatten -> Linear(256, 120) -> ReLU -> Linear(120, 84) -> ReLU -> Linear(84, 10)
modules = [
    Conv2D(np.array([28, 28]), 1, 6, np.array([5, 5]), np.array([1, 1]), np.array([0, 0])),
    ReLU(),
    MaxPool2D(np.array([2, 2]), np.array([2, 2]), np.array([0, 0])),
    Conv2D(np.array([12, 12]), 6, 16, np.array([5, 5]), np.array([1, 1]), np.array([0, 0])),
    ReLU(),
    MaxPool2D(np.array([2, 2]), np.array([2, 2]), np.array([0, 0])),
    Flatten(),
    Linear(256, 120),
    ReLU(),
    Linear(120, 84),
    ReLU(),
    Linear(84, output_size)
]
loss_fn = CrossEntropyLoss(modules)
optim = Adam(modules, learning_rate=learning_rate, weight_decay=0.0001)

# Define learning rate scheduler
def factor_scheduler(factor):
    optim.lr = max(stop_factor_lr, optim.lr * factor)

# Train the model
for epoch in range(epochs):
    epoch_start = time.time()
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
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    factor_scheduler(0.95)
    print(f'Total Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s')
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

# Save the model
save_state_dict(modules, "saved/model.npz")
get_modules_info(modules, "saved/model.yaml")