from math import ceil, floor

import numpy as np
from PIL import Image

__author__ = 'Lucas Kjaero'

TARGET_HEIGHT = 32
TARGET_WIDTH = 32
# Note, in numpy the indexing is opposite cartesian, at (y, x)
TARGET_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH)


def no_processing(image):
    return np.array(image)


def view_resize_and_letterbox(image):
    image_array = resize_and_letterbox(image)
    return Image.fromarray(image_array)


def resize_and_letterbox(image):

    # Default values
    width = image.width
    height = image.height

    x1 = 0
    y1 = 0
    x2 = width
    y2 = height

    # Calculate new dimensions and letterboxing positions
    if width == height:
        new_width, new_height = TARGET_WIDTH, TARGET_HEIGHT
    elif width > height:
        new_height = ceil(TARGET_WIDTH * (height / width))
        new_width = TARGET_WIDTH

        # prepare for letterboxing
        buffer = TARGET_HEIGHT - new_height
        # If there's an extra pixel, add it to the top
        y1 = ceil(buffer / 2)
        y2 = y1 + new_height
    elif width < height:
        new_width = ceil(TARGET_HEIGHT * (width / height))
        new_height = TARGET_HEIGHT

        # prepare for letterboxing
        buffer = TARGET_WIDTH - new_width
        # If there's an extra pixel, add it to the right side.
        x1 = floor(buffer / 2)
        x2 = x1 + new_width

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a blank background image to paste into
    letterboxed_image = np.zeros(TARGET_SHAPE)
    letterboxed_image.fill(255)

    # Letterbox the image
    image_array = np.array(resized_image)
    letterboxed_image[y1:y2, x1:x2] = image_array

    return letterboxed_image


def process_image(image):
    image_array = image.resize(32, 32)

    # Normalize between -1 and 1
    normalized_image_array = image_array / 256

    reshaped_image_array = normalized_image_array.reshape((TARGET_HEIGHT, TARGET_WIDTH, 1))

    return reshaped_image_array
