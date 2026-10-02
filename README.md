# CNN from Scratch

A convolutional neural network for handwritten digit recognition (MNIST), built using only NumPy and SciPy. No PyTorch or TensorFlow. Every layer, forward pass, backward pass, and weight update is written by hand.

## Architecture

| # | Layer | Output shape |
|---|-------|--------------|
| 1 | Conv 3×3, 16 filters + ReLU | 16 × 26 × 26 |
| 2 | Conv 3×3, 32 filters + ReLU | 32 × 24 × 24 |
| 3 | Max pool 2×2, stride 2 | 32 × 12 × 12 |
| 4 | Conv 2×2, 64 filters + ReLU | 64 × 11 × 11 |
| 5 | Max pool 2×2, stride 2 | 64 × 5 × 5 |
| 6 | Flatten | 1600 × 1 |
| 7 | Dense 1600 → 128 + ReLU | 128 × 1 |
| 8 | Dense 128 → 32 + ReLU | 32 × 1 |
| 9 | Dense 32 → 10 + Softmax | 10 × 1 |

Training uses categorical cross-entropy loss and plain SGD (one sample at a time, learning rate 0.001).

## Files

- `helper.py`: layer implementations (`Convolutional`, `MaxPooling`, `Dense`, `Reshape`, `Activation`, `Sigmoid`, `Softmax`) and loss functions
- `data.py`: loads `train.csv`, shuffles it, normalizes pixels, one-hot encodes labels, and splits off 1,000 samples as a dev set
- `network.py`: builds the network, trains it, saves the weights, and reports dev-set accuracy
- `test.py`: loads saved weights and lets you check predictions interactively, either by dev-set index or on your own 28×28 PNG images
- `utilities.py`: `save_model` / `load_model` helpers (pickle)
- `model*.pkl`: saved weights from previous training runs
- `numbers/`: sample hand-drawn digit images for testing

## Setup

```bash
pip install numpy scipy pandas matplotlib pillow
```

Download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and place it in the project root. It isn't included in this repo because of its size.

## Usage

Train the network (weights are saved to `model3.pkl`):

```bash
python network.py
```

Test predictions:

```bash
python test.py
```

You'll be asked for a dev-set index to compare the prediction with the true label (enter `-1` to move on). Then you'll be asked for a path to an image such as `numbers/Niko2.png` (enter `na` to exit). Images are converted to grayscale and resized to 28×28. Draw a white digit on a black background to match MNIST.

> Note: the `load_model(...)` call in `test.py` is commented out. Uncomment it and point it at a weights file saved from the same architecture (e.g. `model3.pkl`) to use trained weights instead of random ones.
