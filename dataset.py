from math import ceil
from os import listdir
from os.path import expanduser

import numpy as np

from pycasia.CASIA import CASIA

from preprocessing import no_processing

__author__ = 'Lucas Kjaero'


class CasiaML(CASIA):
    """
    Class to read the dataset for all learning functions.
    """
    def __init__(self, split_percentage=0.2, pre_processing_function=no_processing):
        super().__init__()
        assert self.get_all_datasets() is True, "Datasets aren't properly loaded, " \
                                                "rerun to try again or download datasets manually."

        # Give purposes to the dataset for our own use.
        self.datasets["competition-gnt"]["purpose"] = "train"
        self.datasets["HWDB1.1trn_gnt_P1"]["purpose"] = "train"
        self.datasets["HWDB1.1trn_gnt_P2"]["purpose"] = "train"
        self.datasets["HWDB1.1tst_gnt"]["purpose"] = "test"

        self.training_sets = [dataset for dataset in self.datasets if self.datasets[dataset]["purpose"] == "train"]
        self.testing_sets = [dataset for dataset in self.datasets if self.datasets[dataset]["purpose"] == "test"]

        # Sample count is needed to know how many samples per epoch
        sample_count = 0
        # Label count is important for one-hot encoding.
        labels = []

        """
        for dataset in self.datasets:
            for image, label in self.load_dataset(dataset):
                sample_count += 1
                labels.append(label)

        self.classes = set(labels)
        self.class_count = len(self.classes)
        self.sample_count = sample_count
        print("Data loaded with %s samples and %s classes" % (self.sample_count, self.class_count))
        """
        # TODO remove after getting a proper model.
        self.sample_count = 1346168
        self.class_count = 3755

        self.split_percentage = split_percentage
        self.train_files = []
        self.test_files = []

        # Generate train-test split
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

        self.pre_processing_function = pre_processing_function

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

                    batch_x.append(processed_image)
                    batch_y.append(label)

                    if len(batch_x) >= batch_size:
                        yield np.asarray(batch_x), np.asarray(batch_y)
                        batch_x = []
                        batch_y = []

    def test_set(self):
        """
        A generator to load all the images in the testing set. Use this during training to verify models.
        :return: Yields (image, label) tuples of String, Pillow.Image.Image
        """
        for file in self.test_files:
            for image, label in self.load_gnt_file(file):
                processed_image = self.pre_processing_function(image)
                yield processed_image, label

    def validation_set(self):
        """
        A generator to load all the images in the validation set.
        Don't touch this until you are evaluating the final model.
        If you use this to inform your model building, you will have a much higher chance of overfitting.
        :return: Yields (image, label) tuples of String, Pillow.Image.Image
        """
        for dataset in self.testing_sets:
            for image, label in self.load_dataset(dataset):
                processed_image = self.pre_processing_function(image)
                yield processed_image, label
