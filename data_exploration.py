import glob
from IPython.display import display, Image
from io import BytesIO
from PIL import Image as Pil

__author__ = 'Lucas Kjaero'


def identity_function(image):
    return image


def display_label(label, transform=identity_function):
    base = "raw/%s/" % label
    for path in glob.glob(base + "*.jpg"):
        pil_image = transform(Pil.open(path))

        image_bytes = BytesIO()
        pil_image.save(image_bytes, format="jpeg")

        ip_image = Image(data=image_bytes.getvalue(), format="jpeg")
        display(ip_image)