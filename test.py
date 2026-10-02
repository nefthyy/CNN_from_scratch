from data import x_d, y_d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utilities import load_model
from helper import Convolutional, Activation, MaxPooling, Reshape, Dense, Sigmoid, Softmax


'''
# Create the layers
num_classes = 10 # number of outputs

conv1 = Convolutional((1, 28, 28), 3, 16) # gets 16 filters of size 3x3 and initializes shapes
relu1 = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))

conv2 = Convolutional((16, 26, 26), 3, 32)
relu2 = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))

pool1 = MaxPooling(size = 2, stride = 2)  # Output shape now: (32, 12, 12)

reshape = Reshape((32, 12, 12), (32*12*12, 1))
dense1 = Dense(32*12*12, 64)
relu3 = Activation(lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float))
dense2 = Dense(64, num_classes)
# sigmoid2 = Sigmoid()
soft1 = Softmax()

# --- Combine into a list for cleaner loop later (optional) ---
layers = [conv1, relu1, conv2, relu2, pool1, reshape, dense1, relu3, dense2, soft1]

'''


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



# load correct biases, weigths, and kernels
# try:
#     load_model(layers, "model2.pkl")
# except FileNotFoundError:
#     print("No Model Found")


def test_index(index, layers, x, y):
    x_data = x[index, :, :, :]
    y_data = y[index, :, :]
    out = x_data
    for layer in layers:
        out = layer.forward(out)
    return np.argmax(out), np.argmax(y_data)




while True:
    i = input("Enter index to compare (-1 to exit): ")
    i = int(i)
    if (i == -1):
        break
    else:
        guess, actual = test_index(i, layers, x_d, y_d)
        print("Guess: ", guess, " | Actaul: ", actual)

        x_data = x_d[i, :, :, :]
        x_data = x_data.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(x_data, interpolation='nearest')
        plt.show()

'''
w=-1
for x_d, y_d in zip(x_d, y_d):
    w += 1
    if (np.argmax(y_d) == 7):
'''  
        


while True:
    t = input("Enter the file of your number (na to exit): ")
    if (t == "na"):
        break
    try:
        # Convert png to correct data pixel structure
        img = Image.open(t).convert("L")
        if (img.size != (28,28)):
            img = img.resize((28,28))
        img_array = np.array(img) 
        X = img_array.flatten()/255
        X_data = X.reshape(1,28,28)
        # Show Prediction
        out = X_data
        for layer in layers:
            out = layer.forward(out)
        guess = np.argmax(out)
        print("Prediction --> ", guess)

        plt.gray()
        plt.imshow(img, interpolation='nearest')
        plt.show()

    except FileNotFoundError:
        print("Error, file not found try again ")
        
