import glob
from IPython.display import display, Image
from io import BytesIO
from PIL import Image as Pil

__author__ = 'Lucas Kjaero'

EXPLORATION_DIR = "raw/competition-gnt/"


def identity_function(image):
    """
    Default image transformation function. Does nothing. 
    :param image: The image to transform.
    :return: The original image. 
    """
    return image


def display_label(label, transform=identity_function, directory=EXPLORATION_DIR):
    """
    Displays all the images with a given label. 
    :param label: The label to look at.
    :param transform: A function to process the images.
    :param directory: The path to the dataset to display. Default is "competition-gnt"
    :return: Nothing
    """
    base = directory + "%s/" % label
    for path in glob.glob(base + "*.jpg"):
        pil_image = transform(Pil.open(path))

        image_bytes = BytesIO()
        pil_image.save(image_bytes, format="jpeg")

        ip_image = Image(data=image_bytes.getvalue(), format="jpeg")
        display(ip_image)
