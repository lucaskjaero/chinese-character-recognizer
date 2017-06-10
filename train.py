from collections import defaultdict
from codecs import decode, encode
import glob
from os.path import isfile, isdir
from os import remove
import struct
from urllib.request import urlretrieve
import zipfile

import numpy as np
import scipy

from scipy.misc import toimage

from tqdm import tqdm

__author__ = 'Lucas Kjaero'

DATASETS = {
        "HWDB1.0trn": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0trn.zip",
        "HWDB1.0tst": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0tst.zip",
        "competition-gnt": "http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-gnt.zip",
        "competition-dgr": "http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-dgr.zip"
    }

SIDE = 224


class DLProgress(tqdm):
    """ Class to show progress on dataset download """
    # Code adapted from a Udacity machine learning project.
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


def load_gnt_file(filename):
    # Keys are in GB2312

    print("Loading file: %s" % filename)

    f = open(filename, "rb")

    full_data = defaultdict(lambda: [])

    num_classes = 8

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

        existing_labels = full_data.keys()
        if (label in existing_labels) or (len(existing_labels) < num_classes):
            image = np.array(bytes).reshape(height, width)
            #image = scipy.misc.imresize(image, (SIDE, SIDE))
            #image = (image.astype(float) / 256) - 0.5  # normalize to [-0.5,0.5] to avoid saturation
            # TODO: should also invert image so convolutional zero-padding doesn't add a "border"?
            full_data[label].append(image)

    f.close()

    return full_data


def load_datasets():
    path = "competition-gnt/C001-f-f.gnt"
    data = load_gnt_file(path)
    for key in data.keys():
        image = data[key][0]
        print(key)
        toimage(image).show()

    return data


def main():
    #get_datasets()
    load_datasets()

if __name__ == '__main__':
    main()
