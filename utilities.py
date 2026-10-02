# store data from training
import pickle
from helper import Convolutional, Dense


def save_model(layers, filename = "model3.pkl"):
    weights = []
    for layer in layers:
        if isinstance(layer, Dense):
            weights.append({
                "type": "Dense",
                "weights": layer.weights,
                "bias": layer.bias
            })
        elif isinstance(layer, Convolutional):
            weights.append ({
                "type": "Convolutional",
                "kernels": layer.kernels,
                "biases": layer.biases
            })
        else:
            weights.append(None)

    with open(filename, "wb") as f:
        pickle.dump(weights, f)
    
    print("model saved")



def load_model(layers, filename="model3.pkl"):
    with open(filename, "rb") as f:
        weights = pickle.load(f)
    for i, layer in enumerate(layers):
        layer_data = weights[i]
        if layer_data is None:
            continue
        if isinstance(layer, Dense) and layer_data["type"] == "Dense":
            layer.weights = layer_data["weights"]
            layer.bias = layer_data["bias"]
        elif isinstance(layer, Convolutional) and layer_data["type"] == "Convolutional":
            layer.kernels = layer_data["kernels"]
            layer.biases = layer_data["biases"]

    print("model loaded successfully")
