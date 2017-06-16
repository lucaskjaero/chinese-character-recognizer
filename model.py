from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D

from preprocessing import TARGET_HEIGHT, TARGET_WIDTH

__author__ = 'Lucas Kjaero'

FILTERS = 75
# TODO Make sure to keep this updated.

# The extra one is to let type inference know that the image is black and white.
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, 1)


def alex_net(output_dimensions):
    """
    Returns an AlexNet model. 
    :param output_dimensions: The number of output classes. 
    :return: The model. 
    """
    # TODO complete
    model = Sequential()
    model.add(Conv2D(FILTERS, kernel_size=(11, 11), strides=4, padding="same", activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"))
    model.add(Conv2D(FILTERS, kernel_size=(5, 5), strides=1, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding="valid"))
    model.add(Conv2D(FILTERS, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Conv2D(FILTERS, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Conv2D(FILTERS, kernel_size=(3, 3), strides=1, padding="same", activation='relu'))
    model.add(Dense(output_dimensions, activation='relu'))
    model.add(Dense(output_dimensions, activation='relu'))
    model.add(Dense(output_dimensions, activation='softmax'))
    print("Built model")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Compiled model")

    return model


"""
from sklearn.metrics import fbeta_score

p_valid = model.predict(x_valid, batch_size=128)
print(y_valid)
print(p_valid)
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
"""
