from dataset import train_set, train_set_counts
from model import alex_net

__author__ = 'Lucas Kjaero'

TOTAL_SAMPLES, NUM_CLASSES = train_set_counts()


def train_model(model):
    """
    Trains the given model. Returns a history object.
    :param model: The model to train.
    :return: A history object.
    """
    steps_per_epoch = TOTAL_SAMPLES
    return model.fit_generator(train_set, steps_per_epoch, epochs=4)


def main():
    model = alex_net(output_dimensions=NUM_CLASSES)
    train_model(model)


if __name__ == '__main__':
    main()
