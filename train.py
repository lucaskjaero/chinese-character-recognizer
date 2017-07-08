from dataset import CasiaML
from model import alex_net

from preprocessing import process_image

__author__ = 'Lucas Kjaero'

BATCH_SIZE = 100


def main():
    dataset = CasiaML(pre_processing_function=process_image)

    class_count = dataset.class_count
    sample_count = dataset.sample_count

    model = alex_net(output_dimensions=class_count)

    steps_per_epoch = sample_count / BATCH_SIZE
    history = model.fit_generator(dataset.train_set(batch_size=BATCH_SIZE), steps_per_epoch, epochs=4)


if __name__ == '__main__':
    main()
