from urllib.request import urlretrieve
from os.path import isfile, isdir
import zipfile

from tqdm import tqdm

__author__ = 'Lucas Kjaero'


class DLProgress(tqdm):
    """ Class to show progress on dataset download """
    # Code adapted from a Udacity machine learning project.
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_datasets():
    """
    Checks to see if all the datasets are present. If not, it downloads and unzips them. 
    """
    datasets = {
        "HWDB1.0trn": "url",
        "HWDB1.0tst": "url",
        "OLHWDB1.0trn": "url",
        "OLHWDB1.0tst": "url"
    }

    was_error = False
    for dataset in datasets:
        # Make sure the zip files are there
        zip_path = dataset + ".zip"
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
                    zip_archive.extractall()
                    zip_archive.close()
            except Exception as ex:
                print("Error unzipping %s: %s" % (zip_path, ex))
                was_error = True

    if was_error:
        print("\nThis recognizer is trained by the CASIA handwriting database. "
              "If the download doesn't work, you can get the files at nlpr.ia.ac.cn")


def main():
    download_datasets()

if __name__ == '__main__':
    main()
