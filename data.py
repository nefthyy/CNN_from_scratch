import numpy as np
import pandas as pd

# number of possible outputs
num_classes = 10

def preprocess_data(x, Y, limit=None):
    # Puts limit on data if specified
    if limit is not None:
        x = x[:limit]
        Y = Y[:limit]
        m = limit
    else:
        m = x.shape[0]
    # 1x28x28
    x = x.reshape(m, 1, 28, 28)
    # Normalize pixel values
    x = x.astype("float32") / 255.0 

    # one hot encode y
    y_hot = np.zeros((m, num_classes))
    y_hot[np.arange(m), Y] = 1
    y_hot = y_hot.reshape(m, num_classes, 1)
    return x, y_hot

        
# Load data and convert into an array
data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Dev data to check results
data_dev = data[0:1000]
Y_dev = data_dev[:, 0]
X_dev = data_dev[:, 1:]

# y = actuals numbers and X = pixel values
data_train = data[1000:]
y_train = data_train[:, 0]
X_train = data_train[:, 1:]

# Process data
x_train, y_train = preprocess_data(X_train, y_train, 41000)
x_dev, y_dev = preprocess_data(X_dev, Y_dev, 1000)
# to be used in test.py
x_d = x_dev
y_d = y_dev
