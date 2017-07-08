from math import ceil
from os import listdir
from os.path import expanduser, isfile

import numpy as np

from tqdm import tqdm

from pycasia.CASIA import CASIA

from preprocessing import no_processing

__author__ = 'Lucas Kjaero'

LABELS_FILE = "labels.txt"
SAMPLES_FILE = "samples.txt"


class CasiaML(CASIA):
    """
    Class to read the dataset for all learning functions.
    """
    def __init__(self, split_percentage=0.2, pre_processing_function=no_processing):
        super().__init__()
        print("Checking for datasets")
        assert self.get_all_datasets() is True, "Datasets aren't properly loaded, " \
                                                "rerun to try again or download datasets manually."

        # Give purposes to the dataset for our own use.
        self.datasets["competition-gnt"]["purpose"] = "train"
        self.datasets["HWDB1.1trn_gnt_P1"]["purpose"] = "train"
        self.datasets["HWDB1.1trn_gnt_P2"]["purpose"] = "train"
        self.datasets["HWDB1.1tst_gnt"]["purpose"] = "test"

        self.training_sets = [dataset for dataset in self.datasets if self.datasets[dataset]["purpose"] == "train"]
        self.testing_sets = [dataset for dataset in self.datasets if self.datasets[dataset]["purpose"] == "test"]

        self.pre_processing_function = pre_processing_function

        # Generate train-test split
        self.split_percentage = split_percentage
        self.train_files = []
        self.test_files = []

        for dataset in self.training_sets:
            dataset_path = expanduser(self.base_dataset_path + dataset)
            count = len([name for name in listdir(dataset_path)])
            training_count = ceil((1 - split_percentage) * count)
            testing_count = count - training_count

            paths = [dataset_path + "/" + path for path in listdir(dataset_path)]

            train = paths[0:training_count]
            test = paths[training_count:]

            assert len(train) == training_count
            assert len(test) == testing_count

            self.train_files.extend(train)
            self.test_files.extend(test)

        # Do one time preprocessing of the dataset's labels and samples
        if not isfile(LABELS_FILE) or not isfile(SAMPLES_FILE):
            labels = []

            print("Generating first-run training set information")
            train_samples_count = 0
            for file in tqdm(self.train_files):
                for image, label in self.load_gnt_file(file):
                    train_samples_count = train_samples_count + 1
                    labels.append(label)

            print("Generating first-run testing set information")
            test_samples_count = 0
            for file in tqdm(self.test_files):
                for image, label in self.load_gnt_file(file):
                    test_samples_count = test_samples_count + 1
                    labels.append(label)

            print("Generating first-run validation set information")
            validation_samples_count = 0
            for dataset in self.testing_sets:
                for image, label in self.load_dataset(dataset):
                    validation_samples_count = validation_samples_count + 1
                    labels.append(label)

            if not isfile(LABELS_FILE):
                labels_string = "".join(set(labels))
                with open(LABELS_FILE, "w") as labels_file:
                    labels_file.write(labels_string)

            if not isfile(SAMPLES_FILE):
                with open(SAMPLES_FILE, "w") as samples_file:
                    samples_file.write("train:%s\n" % train_samples_count)
                    samples_file.write("test:%s\n" % test_samples_count)
                    samples_file.write("validation:%s\n" % validation_samples_count)

        # Label count is important for one-hot encoding.
        with open(LABELS_FILE, "r") as labels_file:
            self.classes = labels_file.read()
            self.class_count = len(self.classes)

        # Sample count is needed to know how many samples per epoch
        with open(SAMPLES_FILE, "r") as samples_file:
            samples_data = samples_file.read().split("\n")
            self.training_sample_count = int(samples_data[0].split(":")[1])
            self.testing_sample_count = int(samples_data[1].split(":")[1])
            self.validation_sample_count = int(samples_data[2].split(":")[1])
            self.total_sample_count = self.training_sample_count + self.testing_sample_count + self.validation_sample_count

        print("Data loaded with %s classes and %s samples" % (self.class_count, self.total_sample_count))
        print("Train: %s, Test: %s, Validation: %s" % (self.training_sample_count, self.testing_sample_count, self.validation_sample_count))

    def train_set(self, batch_size=100):
        """
        A generator to load all the images in the training set. Loads data infinitely.
        :param batch_size: The number of samples to return at once.
        :return: Yields batches of (image, label) tuples of String, Pillow.Image.Image
        """
        batch_x = []
        batch_y = []

        while True:
            for file in self.train_files:
                for image, label in self.load_gnt_file(file):
                    processed_image = self.pre_processing_function(image)
                    one_hot_label = self.one_hot(label)

                    batch_x.append(processed_image)
                    batch_y.append(one_hot_label)

                    if len(batch_x) >= batch_size:
                        yield np.asarray(batch_x), np.asarray(batch_y)
                        batch_x = []
                        batch_y = []

    def test_set(self, batch_size=100):
        """
        A generator to load all the images in the testing set. Use this during training to verify models.
        :return: Yields (image, label) tuples of String, Pillow.Image.Image
        """
        batch_x = []
        batch_y = []

        while True:
            for file in self.test_files:
                for image, label in self.load_gnt_file(file):
                    processed_image = self.pre_processing_function(image)
                    one_hot_label = self.one_hot(label)

                    batch_x.append(processed_image)
                    batch_y.append(one_hot_label)

                    if len(batch_x) >= batch_size:
                        yield np.asarray(batch_x), np.asarray(batch_y)
                        batch_x = []
                        batch_y = []

    def validation_set(self, batch_size=100):
        """
        A generator to load all the images in the validation set.
        Don't touch this until you are evaluating the final model.
        If you use this to inform your model building, you will have a much higher chance of overfitting.
        :return: Yields (image, label) tuples of String, Pillow.Image.Image
        """
        batch_x = []
        batch_y = []

        while True:
            for dataset in self.testing_sets:
                for image, label in self.load_dataset(dataset):
                    processed_image = self.pre_processing_function(image)
                    one_hot_label = self.one_hot(label)

                    batch_x.append(processed_image)
                    batch_y.append(one_hot_label)

                    if len(batch_x) >= batch_size:
                        yield np.asarray(batch_x), np.asarray(batch_y)
                        batch_x = []
                        batch_y = []

    def one_hot(self, character):
        one_hot_vector = np.zeros(self.class_count)
        one_hot_vector[self.classes.index(character)] = 1
        return one_hot_vector
