import glob
import struct
import zipfile

from codecs import decode
from os import makedirs, remove
from os.path import isfile, isdir
from time import clock
from urllib.request import urlretrieve

import numpy as np

from scipy.misc import toimage

from tqdm import tqdm

__author__ = 'Lucas Kjaero'

DATASETS = {
        "competition-gnt": {
                            "url": "http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-gnt.zip",
                            "purpose": "train"},
        "HWDB1.1trn_gnt_P1": {
                            "url": "http://www.nlpr.ia.ac.cn/databases/Download/feature_data/HWDB1.1trn_gnt_P1.zip",
                            "purpose": "train"},
        "HWDB1.1trn_gnt_P2": {
                            "url": "http://www.nlpr.ia.ac.cn/databases/Download/feature_data/HWDB1.1trn_gnt_P2.zip",
                            "purpose": "train"},
        "HWDB1.1tst_gnt": {
                            "url": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip",
                            "purpose": "test"}
}


class DLProgress(tqdm):
    """ Class to show progress on dataset download """
    # Progress bar code adapted from a Udacity machine learning project.
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def get_datasets():
    """
    Make sure the datasets are present. If not, downloads and extracts them.
    Attempts the download five times because the file hosting is unreliable.
    :return:
    """
    success = True

    for dataset in DATASETS:
        # If the dataset is present, no need to download anything.
        if not isdir(dataset):

            # Try 5 times to download. The download page is unreliable, so we need a few tries.
            was_error = False
            for iteration in range(5):
                if iteration > 0 or was_error is True:
                    was_error = get_dataset(dataset)

            if was_error:
                print("\nThis recognizer is trained by the CASIA handwriting database.")
                print("If the download doesn't work, you can get the files at %s" % DATASETS[dataset]["url"])
                print("If you have GFW problems, wget may be effective at downloading.")
                success = False

    return success


def get_dataset(dataset):
    """
    Checks to see if the dataset is present. If not, it downloads and unzips it.
    """
    was_error = False
    zip_path = dataset + ".zip"

    # Download zip files if they're not there
    if not isfile(zip_path):
        try:
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset) as pbar:
                urlretrieve(DATASETS[dataset]["url"], zip_path, pbar.hook)
        except Exception as ex:
            print("Error downloading %s: %s" % (dataset, ex))
            was_error = True

    # Unzip the data files
    if not isdir(dataset):
        try:
            with zipfile.ZipFile(zip_path) as zip_archive:
                zip_archive.extractall(path=dataset)
                zip_archive.close()
        except Exception as ex:
            print("Error unzipping %s: %s" % (zip_path, ex))
            # Usually the error is caused by a bad zip file. Delete it so the program will try to download it again.
            remove(zip_path)
            was_error = True

    return was_error


def load_datasets(purpose="train"):
    """
    Generator loading all images in the dataset for the given purpose. Uses exploration data if nothing specified.
    :param purpose: Data purpose. Options are "train" and "test". Default is train.
    :return: Yields (label, image) tuples.
    """
    # Just make sure the data is there. If not, this will download them.
    assert get_datasets() is True, "Datasets aren't properly loaded, rerun to try again or download datasets manually."

    paths = [path for path in DATASETS if DATASETS[path]["purpose"] == purpose]

    for path in paths:
        for label, image in load_gnt_dir(path):
            yield label, image


def load_gnt_dir(dataset_path):
    """
    Load a directory of gnt files. Yields the image and label in tuples.
    :param dataset_path: The directory to search in.
    :return:  Yields (label, image) pairs
    """
    for path in glob.glob(dataset_path + "/*.gnt"):
        for label, image in load_gnt_file(path):
            yield label, image


def load_gnt_file(filename):
    """
    Generator yielding all characters and images from a given GNT file.
    :param filename: The file path to load.
    :return: (character, image) tuples
    """
    print("Loading file: %s" % filename)

    # Thanks to nhatch for the code to read the GNT file, available at https://github.com/nhatch/casia
    with open(filename, "rb") as f:
        while True:
            packed_length = f.read(4)
            if packed_length == b'':
                break

            length = struct.unpack("<I", packed_length)[0]
            raw_label = struct.unpack(">cc", f.read(2))
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]
            photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))

            # Comes out as a tuple of chars. Need to be combined. Encoded as gb2312, gotta convert to unicode.
            label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
            # Create an array of bytes for the image, match it to the proper dimensions, and turn it into a PIL image.
            image = toimage(np.array(photo_bytes).reshape(height, width))

            yield (label, image)


def get_all_raw():
    """
    Used to create easily introspectable image directories of all the data.
    :return:
    """
    for dataset in DATASETS:
        get_raw(dataset)


def get_raw(path):
    """
    Creates an easily introspectable image directory for a given dataset.
    :param path: The dataset to get.
    :return: None
    """
    for label, image in load_gnt_dir(path):
        output_image(path, label, image)


def output_image(prefix, label, image):
    """
    Exports images into files. Organized by dataset / label / image.
    Stored in the raw directory.
    Saves the image with a name of the current time in seconds. This is to prevent two filenames from being the same.
    :param prefix: The name of the dataset to save the images under.
    :param label: The character the image represents/
    :param image: The image file.
    :return: nothing.
    """
    prefix_path = "raw/" + prefix
    if not isdir(prefix_path):
        makedirs(prefix_path)

    label_path = prefix_path + "/" + label
    if not isdir(label_path):
        makedirs(label_path)

    image.save(label_path + "/%s.jpg" % clock())
