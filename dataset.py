import glob
import struct
import zipfile

from codecs import decode
from os.path import isfile, isdir
from os import remove
from urllib.request import urlretrieve

import numpy as np

from scipy.misc import toimage

from tqdm import tqdm

from preprocessing import process_image

DATASETS = {
        "HWDB1.0trn": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0trn.zip",
        "HWDB1.0tst": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0tst.zip",
        "competition-gnt": "http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-gnt.zip",
        "competition-dgr": "http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-dgr.zip"
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
    for dataset in DATASETS:
        # If the dataset is present, no need to download anything.
        if not isdir(dataset):

            # Try 5 times to download. The download page is unreliable, so we need a few tries.
            was_error = False
            for iteration in range(5):
                if iteration > 0 or was_error is True:
                    was_error = get_dataset(dataset)

            if was_error:
                print("\nThis recognizer is trained by the CASIA handwriting database. " +
                      "If the download doesn't work, you can get the files at nlpr.ia.ac.cn")


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
                urlretrieve(DATASETS[dataset], zip_path, pbar.hook)
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


def load_datasets():
    # full_data = defaultdict(lambda: [])
    keys = []
    for label, image in load_gnt_dir("competition-gnt"):
        keys.append(label)
        # Image is PIL.Image.Image

    labels = set(keys)
    print("%s unique labels:" % len(labels))
    print(labels)


def load_gnt_dir(dataset_path):
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
            bytes = struct.unpack("{}B".format(height * width), f.read(height * width))

            # Comes out as a tuple of chars. Need to be combined. Encoded as gb2312, gotta convert to unicode.
            label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
            # Create an array of bytes for the image, match it to the proper dimensions, and turn it into a PIL image.
            image = toimage(np.array(bytes).reshape(height, width))

            yield (label, image)
