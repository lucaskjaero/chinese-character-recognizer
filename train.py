from dataset import train_set, train_set_sample_count, test_set, validation_set
from model import alex_net

import numpy as np

__author__ = 'Lucas Kjaero'


def train_model(model):
    steps_per_epoch = train_set_sample_count()
    return model.fit_generator(train_set, steps_per_epoch, epochs=4)


def test_model(model):
    pass


def validate_model(model):
    pass


def main():
    model = alex_net()
    history = train_model(model)


if __name__ == '__main__':
    main()
