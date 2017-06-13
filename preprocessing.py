import numpy as np

__author__ = 'Lucas Kjaero'

SCALED_HEIGHT = 75
SCALED_WIDTH = 75

def process_image(image):
    # Resize image to average size
    resized_image = image.resize((SCALED_WIDTH, SCALED_HEIGHT))

    # Convert image to array and then normalize between -1 and 1
    image_array = np.array(resized_image)
    normalized_image_array = (2 * (image_array / 256)) - 1

    return normalized_image_array
