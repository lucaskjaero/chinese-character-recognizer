from urllib.request import urlretrieve
from os.path import isfile, isdir
from os import remove
import zipfile

from tqdm import tqdm

__author__ = 'Lucas Kjaero'

datasets = {
        "HWDB1.0trn": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0trn.zip",
        "HWDB1.0tst": "http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.0tst.zip",
        "competition-gnt": "http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-gnt.zip",
        "competition-dgr": "http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-dgr.zip"
    }


class DLProgress(tqdm):
    """ Class to show progress on dataset download """
    # Code adapted from a Udacity machine learning project.
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def get_datasets():
    for dataset in datasets:
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
                urlretrieve(datasets[dataset], zip_path, pbar.hook)
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
    pass


def main():
    get_datasets()

if __name__ == '__main__':
    main()
