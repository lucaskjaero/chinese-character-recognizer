from dataset import CasiaML
from model import alex_net
from preprocessing import process_image

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

__author__ = 'Lucas Kjaero'

BATCH_SIZE = 100


def main():
    dataset = CasiaML(pre_processing_function=process_image)

    model = alex_net(output_dimensions=dataset.class_count)

    steps_per_epoch = dataset.training_sample_count / BATCH_SIZE
    validation_steps = dataset.testing_sample_count / BATCH_SIZE

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    checkpointing = ModelCheckpoint("Model.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=False, mode='auto', period=1)

    model.fit_generator(dataset.train_set(batch_size=BATCH_SIZE), steps_per_epoch, epochs=50, callbacks=[checkpointing, early_stopping, tensorboard], validation_data=dataset.test_set(batch_size=BATCH_SIZE), validation_steps=validation_steps)

    model.save("FINAL_MODEL.h5")


if __name__ == '__main__':
    main()
