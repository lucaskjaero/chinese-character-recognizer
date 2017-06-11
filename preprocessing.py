__author__ = 'Lucas Kjaero'


def process_image(image):
    # image = (image.astype(float) / 256) - 0.5  # normalize to [-0.5,0.5] to avoid saturation
    # TODO: should also invert image so convolutional zero-padding doesn't add a "border"?
    yield image
