# ============= NEURAL NETWORK PARAMETERS =============
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

DROPOUT_RATE = 0.2
CONVOLUTION_FILTERS = [32, 64, 128, 256]

# neuron number in dense layers
DENSE_LAYER_NEURON_NUMBER = 256

# Input shape of the model
n = 128
IMAGE_SIZE = (n, n)
INPUT_SHAPE = (n, n, 3)

NUM_CLASSES = 3

RANDOM_SEED = 42


# model path
MODEL_PATH = './code/models/model.keras'

# callbacks
CALLBACKS = [
    EarlyStopping(patience=10),
    ModelCheckpoint(filepath=MODEL_PATH,
                    save_best_only=True,
                    monitor='val_loss'),
]
