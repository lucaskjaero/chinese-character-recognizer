import glob
import struct
import zipfile

from codecs import decode
from math import ceil
from os import listdir, makedirs, remove
from os.path import isdir, isfile
from time import clock
from urllib.request import urlretrieve

import numpy as np

from scipy.misc import toimage

from tqdm import tqdm

from preprocessing import no_processing, process_image

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

TRAINING_SETS = [dataset for dataset in DATASETS if DATASETS[dataset]["purpose"] == "train"]
TESTING_SETS = [dataset for dataset in DATASETS if DATASETS[dataset]["purpose"] == "test"]

"""
DATASET DOWNLOADING
"""


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
                print("If you have download problems, "
                      "wget may be effective at downloading because of download resuming.")
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
    Generator loading all images in the dataset for the given purpose. Uses training data if nothing specified.
    :param purpose: Data purpose. Options are "train" and "test". Default is train.
    :return: Yields (image, label) tuples. Pillow.Image.Image
    """
    # Just make sure the data is there. If not, this will download them.
    assert get_datasets() is True, "Datasets aren't properly loaded, rerun to try again or download datasets manually."

    paths = [path for path in DATASETS if DATASETS[path]["purpose"] == purpose]

    for path in paths:
        for image, label in load_gnt_dir(path):
            yield image, label

"""
GNT FILE READERS
"""


def load_gnt_dir(dataset_path, preprocess=no_processing):
    """
    Load a directory of gnt files. Yields the image and label in tuples.
    :param dataset_path: The directory to search in.
    :param preprocess: A preprocessing function. Default does nothing. 
    :return:  Yields (image, label) pairs
    """
    for path in glob.glob(dataset_path + "/*.gnt"):
        for image, label in load_gnt_file(path, preprocess=preprocess):
            yield image, label


def load_gnt_file(filename, preprocess=no_processing):
    """
    Load characters and images from a given GNT file.
    :param filename: The file path to load.
    :param preprocess: A function to do any preprocessing. Default is no processing.
    :return: (image: np.array, character) tuples
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
            processed_image = preprocess(image)

            yield processed_image, label

"""
FILE COUNTERS
"""


def files_for_purpose(purpose="train"):
    """
    Get the file count for a particular set of datasets. Used for a test-train split.
    :param purpose: Which set to use
    """
    count = 0
    for dataset in [dataset for dataset in DATASETS if DATASETS[dataset]["purpose"] == purpose]:
        count += files_in_dataset(dataset)
    return count


def files_in_dataset(dataset):
    """
    Gets the count of files in an individual dataset.
    :param dataset: The path of the dataset.
    :return: The count
    """
    assert get_datasets() is True, "Datasets aren't properly loaded, rerun to try again or download datasets manually."
    path = dataset + "/"
    return len([name for name in listdir(path)])

"""
SET GENERATORS
"""


def test_train_split(split_percentage=0.2):
    """
    Partition the files in the dataset into train and test.
    :param split_percentage: The percentage to split on.
    :return: Two lists of filenames containing the set. Train, test.
    """
    train_files = []
    test_files = []

    # TODO implement k-fold cross-validation

    for dataset in TRAINING_SETS:
        count = files_in_dataset(dataset)
        training_count = ceil((1 - split_percentage) * count)
        testing_count = count - training_count

        paths = [dataset + "/" + path for path in listdir(dataset)]

        train = paths[0:training_count]
        test = paths[training_count:]

        assert len(train) == training_count
        assert len(test) == testing_count

        train_files.extend(train)
        test_files.extend(test)

    return train_files, test_files


def train_set_counts(split_percentage=0.2):
    train, test = test_train_split(split_percentage)

    sample_count = 0
    labels = []

    for file in train:
        for image, label in load_gnt_file(file):
            sample_count += 1
            labels.append(label)

    return sample_count, set(labels)


def train_set(split_percentage=0.2, infinite=True):
    """
    A generator to load all the images in the training set. Loads data infinitely.
    :param split_percentage: The percentage of data to use for testing.
    :param infinite: Whether to load data infinitely. If false, loads each image once and then stops.
    :return: Yields (image, label) tuples of String, Pillow.Image.Image
    """
    assert get_datasets() is True, "Datasets aren't properly loaded, rerun to try again or download datasets manually."

    train, test = test_train_split(split_percentage)

    if infinite:
        while True:
            for file in train:
                for image, label in load_gnt_file(file, preprocess=process_image):
                    yield image, label
    else:
        for file in train:
            for image, label in load_gnt_file(file, preprocess=process_image):
                yield image, label


def test_set(split_percentage=0.2):
    """
        A generator to load all the images in the testing set. Use this during training to verify models.
        :param split_percentage: The percentage of data to use for testing.
        :return: Yields (image, label) tuples of String, Pillow.Image.Image
        """
    assert get_datasets() is True, "Datasets aren't properly loaded, rerun to try again or download datasets manually."

    train, test = test_train_split(split_percentage)

    for file in test:
        for image, label in load_gnt_file(file, preprocess=process_image):
            yield image, label


def validation_set():
    """
    A generator to load all the images in the validation set.
    Don't touch this until you are evaluating the final model.
    If you use this to inform your model building, you will have a much higher chance of overfitting.
    :return: Yields (image, label) tuples of String, Pillow.Image.Image
    """
    assert get_datasets() is True, "Datasets aren't properly loaded, rerun to try again or download datasets manually."

    for dataset in TESTING_SETS:
        for image, label in load_gnt_dir(dataset, preprocess=process_image):
            yield image, label

"""
RAW IMAGE OUTPUTS
"""


def get_all_raw():
    """
    Used to create easily introspectable image directories of all the data.
    :return:
    """
    assert get_datasets() is True, "Datasets aren't properly loaded, rerun to try again or download datasets manually."

    for dataset in DATASETS:
        get_raw(dataset)


def get_raw(path):
    """
    Creates an easily introspectable image directory for a given dataset.
    :param path: The dataset to get.
    :return: None
    """
    for image, label in load_gnt_dir(path):
        output_image(path, image, label)


def output_image(prefix, image, label):
    """
    Exports images into files. Organized by dataset / label / image.
    Stored in the raw directory.
    Saves the image with a name of the current time in seconds. This is to prevent two filenames from being the same.
    :param prefix: The name of the dataset to save the images under.
    :param label: The character the image represents/
    :param image: The image file.
    :return: nothing.
    """
    assert type(image) == "PIL.Image.Image", "image is not the correct type. "

    prefix_path = "raw/" + prefix
    if not isdir(prefix_path):
        makedirs(prefix_path)

    label_path = prefix_path + "/" + label
    if not isdir(label_path):
        makedirs(label_path)

    image.save(label_path + "/%s.jpg" % clock())
