from dataset import load_datasets

__author__ = 'Lucas Kjaero'


def main():
    for label, image in load_datasets(purpose="train"):
        pass

if __name__ == '__main__':
    main()
