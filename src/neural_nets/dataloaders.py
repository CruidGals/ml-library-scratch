import numpy as np

class DataLoader:
    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int):
        # Store data and labels
        self.all_data = data
        self.all_labels = labels

        self.data_size = len(data)
        self.batch_size = batch_size

        # Current data + label batch
        self.data = None
        self.labels = None

        # Initialize the rng
        self.rng = np.random.default_rng()

        # Reset the data loader
        self.reset()

    def reset(self):
        """ Initializes the pointer for efficient batch selection and indices of each batch """
        self.pointer = 0
        self.indices = self.rng.choice(self.data_size, self.data_size)

        self.data = None
        self.labels = None
    
    def next(self):
        """ Gets next batch of data and labels """

        # if pointer is equal to data size, then stop
        if self.pointer >= self.data_size:
            return False

        # Prevent from going over
        if self.pointer + self.batch_size > self.data_size:
            end_idx = self.data_size
        else:
            end_idx = self.pointer + self.batch_size

        batch_indices = self.indices[self.pointer:end_idx]
        self.data = self.all_data[batch_indices]
        self.labels = self.all_labels[batch_indices]

        self.pointer = end_idx
        return True