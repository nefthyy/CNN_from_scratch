from data import x_train, y_train, x_dev, y_dev
import csv
from helper import Sigmoid, binary_cross_entropy, binary_cross_entropy_prime, Reshape, Convolutional, Dense, Layer, Activation, MaxPooling, Softmax, catagorical_cross_entropy
import numpy as np
import pandas as pd
from utilities import save_model
import time

start_time = time.time()

# Create the layers
num_classes = 10 # number of outputs

conv1 = Convolutional((1, 28, 28), 3, 16) # gets 16 filters of size 3x3 and initializes shapes
relu1 = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))

conv2 = Convolutional((16, 26, 26), 3, 32)
relu2 = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))

pool1 = MaxPooling(size = 2, stride = 2)  # Output shape now: (32, 12, 12)


# {[TEST]}
convT = Convolutional((32, 12, 12), 2, 64) # lose 2 width and height
reluT = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
poolT = MaxPooling(size = 2, stride = 2)  # Output shape now: (64, 5, 5)


reshape = Reshape((64, 5, 5), (64*5*5, 1))
dense1 = Dense(64*5*5, 128)
relu3 = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))


# {[TEST]}
dense2 = Dense(128, 32)
relu4 = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))


dense3 = Dense(32, num_classes)
# sigmoid2 = Sigmoid()
soft1 = Softmax()


layers = [conv1, relu1, conv2, relu2, pool1, convT, reluT, poolT, reshape, dense1, relu3, dense2, relu4, dense3, soft1]



epochs = 1
learning_rate = .001
count = 0
epoch=0


print("made to loop")


for e in range(epochs):
    total_loss = 0
    cor = 0
    total_count = 0
    '''
    if e > 4:
        learning_rate = .0001
    '''
    # combines x_train with y hot encoded (y_train) (3,1,28,28) and (3,10,1) to (1, 28, 28) (10, 1) | (1, 28, 28) (10, 1) | (1, 28, 28) (10, 1)
    for x_in, y_in in zip(x_train, y_train):
        total_count += 1
        # Forward pass
        # creates the input for the neural network
        out = x_in
        for layer in layers:
            out = layer.forward(out)


        # compute loss
        total_loss += catagorical_cross_entropy(y_in, out)

        # backward pass
        grad = out - y_in  # softmax and cross entropy gradient
        for layer in reversed(layers[:-1]):             # -1 to exclude softmax from backwards loop
            grad = layer.backward(grad, learning_rate)


        # count accuracy
        guess = np.argmax(out)
        actual = np.argmax(y_in)
        if guess == actual:
            cor += 1
        if total_count % 1 == 0:
            ccuracy = cor/total_count
            print("Number ",total_count, " Accuracy: ", ccuracy)
        
    accuracy = cor/total_count

    avg_loss = total_loss / len(x_train)


    print(f"Accuracy: {accuracy} | Loss: {avg_loss:.6f}")

   

save_model(layers)


end_time = time.time()
elapsed_time = end_time - start_time
print("Time: ", elapsed_time/60, " minutes")


# test accuracy
correct = 0

for x_d, y_d in zip(x_dev, y_dev):
    # Forward pass
    out = x_d
    for layer in layers:
        out = layer.forward(out)

    predicted = np.argmax(out)
    actual = np.argmax(y_d)
    if predicted == actual:
        correct += 1

accuracy = correct / len(x_dev)
print(f"Test Accuracy: {accuracy:.4f}")


