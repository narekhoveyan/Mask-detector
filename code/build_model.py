from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.src.layers import MaxPooling2D, AveragePooling2D
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import config


class ModelBuilder:

    def __init__(self, input_shape: tuple, num_classes: int):
        self.convolution_filters = config.CONVOLUTION_FILTERS
        self.input_shape = input_shape
        self.dropout_rate = config.DROPOUT_RATE
        # num of output classes
        self.num_classes = num_classes
        # number of neuron
        self.nnumber = config.DENSE_LAYER_NEURON_NUMBER

    def build_model(self):
        input_layer = Input(shape=self.input_shape, name='main_input')
        # data augmentation
        layer = RandomFlip(mode='horizontal_and_vertical')(input_layer)
        layer = RandomRotation(0.5)(layer)

        # creating convolutional layers and adding pooling
        for i in range(1, len(self.convolution_filters)):
            layer = Conv2D(filters=self.convolution_filters[i], kernel_size=(3, 3), strides=(
                1, 1), data_format='channels_first', padding='same', activation='relu')(layer)
            layer = AveragePooling2D(pool_size=(2, 2))(layer)

        # flattening and adding FC layers
        layer = Flatten()(layer)
        layer = Dropout(self.dropout_rate)(layer)
        for i in range(len(self.convolution_filters) - 1):
            layer = Dense(self.nnumber, activation=None)(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
        layer = Dense(self.num_classes, activation='softmax')(layer)

        # creating and compiling the model
        model = Model(inputs=input_layer, outputs=layer)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate=config.LEARNING_RATE),
                      metrics=['sparse_categorical_accuracy'])

        # return the compiled model
        return model

# %%
