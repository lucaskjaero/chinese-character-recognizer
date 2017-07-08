from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from preprocessing import TARGET_HEIGHT, TARGET_WIDTH

__author__ = 'Lucas Kjaero'

# The extra one is to let type inference know that the image is black and white.
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, 1)


def alex_net(output_dimensions):
    """
    Returns an AlexNet model.
    :param output_dimensions: The number of output classes.
    :return: The model.
    """
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11), strides=4, padding="same", activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"))
    model.add(Conv2D(256, kernel_size=(5, 5), strides=1, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"))
    model.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_dimensions, activation='softmax'))
    print("Built model")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Compiled model")

    return model
