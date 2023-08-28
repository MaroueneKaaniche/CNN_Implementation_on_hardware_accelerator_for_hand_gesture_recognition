#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Train a Keras sequential model
"""
import argparse
import fnmatch
import os
import random as rn
import sys
from datetime import datetime
from pydoc import locate
from random import randint

import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold


# following piece it to init seed to make reproducible results
os.environ['PYTHONHASHSEED'] = '0'
# setup deterministic TF2 operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(10)
rn.seed(100)
tf.random.set_seed(7)

VERBOSE = 2
sys.path.append('/home/maroueneubuntu/Desktop/NewMaxProject/Gesture-recognition-training/TensorFlow/datasets')
# command parser
parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '--epochs',
    action='store',
    dest='epochs',
    type=int,
    default=100,
    help='number of total epochs to run (default: 100)')
parser.add_argument(
    '--batch_size',
    action='store',
    dest='batch_size',
    type=int,
    default=32,
    help='Size of the training batch')
parser.add_argument(
    '--model', required=True, dest='model', type=str, help='CNN model')
parser.add_argument(
    '--dataset',
    required=True,
    dest='dataset',
    type=str,
    help='dataset for the model')
parser.add_argument(
    '--optimizer',
    default='Adam',
    dest='optimizer',
    type=str,
    help='optimizer for training (default: Adam)')
parser.add_argument(
    '--lr',
    default=0.0001,
    dest='lr',
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--save-sample',
    action='store',
    dest='generate_sample',
    type=int,
    help='save the sample at given index as NumPy sample data')
parser.add_argument(
    '--metrics',
    dest='metrics',
    default='accuracy',
    type=str,
    help='metrics used in compiling model(default: accuracy)')
parser.add_argument(
    '--save-sample-per-class',
    action='store_true',
    dest='save_sample_per_class',
    default=False,
    help='save one sample with confidence >0.75 for each class')
parser.add_argument(
    '--channel-first',
    action='store_true',
    dest='channelfirst',
    default=False,
    help='samples will be saved in channel-first format [default:channel-last]')
parser.add_argument(
    '--swap-hw',
    action='store_true',
    dest='swap',
    default=False,
    help='samples will be saved in WH format[default:HW]')
args = parser.parse_args()

# parser.print_help()
# print('input args: ', args)


# create a class for logging screen to file
# make sure to change verbose=2 in fit and evaluate
class Logger():
    """
    stdout logger
    """

    def __init__(self, filename):
        """
        init
        """
        self.terminal = sys.stdout
        self.filename = filename
        self.log = None

    def write(self, message):
        """
        write to file
        """
        with open(self.filename, "a") as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        """
        this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.
        """
        pass  # pylint: disable=unnecessary-pass


def reformat_sample(image, first=True, swapped=False):
    """
    reformat image from Tensorflow default HWC formating
    """

    if not first:
        # print('only works in channel is moved from last to first')
        formating = 'NHWC'
        return formating, image

    # if image is NHWC, N is 1. Remove N and
    if image.ndim == 4:
        image = image.reshape(image.shape[1], image.shape[2],
                              image.shape[3])
        print('removed N:', image.shape)
        image = image.swapaxes(0, 2)
        print('converted to cwh:', image.shape)
        formating = 'CWH'
        if not swapped:
            image = image.swapaxes(1, 2)
            print('converted to chw:', image.shape)
            formating = 'CHW'
    elif image.ndim == 3:
        image = image.swapaxes(0, 2)
        print('converted to cwh:', image.shape)
        formating = 'CWH'
        if not swapped:
            image = image.swapaxes(0, 1)
            print('converted to chw:', image.shape)
            formating = 'CHW'
    return formating, image


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)

    epochs = args.epochs
    batch_size = args.batch_size
    cnn_model = args.model
    optimizer_type = args.optimizer
    learningrate = args.lr
    model_dataset = args.dataset
    model_optimizer = args.optimizer
    sample_index = args.generate_sample
    metrics = args.metrics
    save_sample_per_class = args.save_sample_per_class
    channelfirst = args.channelfirst
    swap = args.swap

    # Log stdout to file
    foldername = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', foldername + '-' + model_dataset)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    sys.stdout = Logger(os.path.join(logdir,  # type: ignore[assignment] # noqa: F821
                                     foldername + '.log'))

    # Tensorboard
    file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'metrics'))
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    print('script:', sys.argv[0])
    print('log dir:', logdir)
    print('epochs:', epochs)
    print('batch_size:', batch_size)

    # Dynamically load datasets
    for _, _, files in sorted(os.walk('datasets')):  # type: ignore[assignment] # noqa: F821
        for name in sorted(files):
            if model_dataset == name[:-3] and fnmatch.fnmatch(name, '*.py'):
                try:
                    ds = locate(name[:-3])
                    print(ds)
                    print(ds.get_classnames())
                    # load data set
                    (train_images, train_labels), (test_images, test_labels)= \
                        ds.get_datasets('./data')  # type: ignore[attr-defined] # noqa: F821

                    class_names = ds.get_classnames()   # type: ignore[attr-defined] # noqa: F821
                    print('Class Names:', class_names)
                except AttributeError as e:
                    raise RuntimeError("cannot load" + model_dataset) from e
                break

    # Dynamically load model
    for _, _, files in sorted(os.walk('models')):  # type: ignore[assignment] # noqa: F821
        for name in sorted(files):
            if cnn_model == name[:-3] and fnmatch.fnmatch(name, '*.py'):
                try:
                    md = locate('models.' + name[:-3])

                    # printing model
                    print('\n' + '-' * 20 + 'Model:' + cnn_model + '-' * 20)
                    with open(os.path.join('models', name)) as fin:
                        for line in fin:
                            if not line.startswith("model"):
                                continue
                            print(line.rstrip())
                            for ln in fin:
                                print(ln.rstrip())
                    print('-' * 50)

                except AttributeError as e:
                    raise RuntimeError("cannot load" + cnn_model) from e
                break

    print('Model:', cnn_model)
    print('optimizer:', optimizer_type)
    print('Initial lr:', learningrate)
    print('Metrics:', metrics)
    print('Dataset:', ds)

    print("train_images shape:", train_images.shape)
    # print("valid_images shape:", valid_images.shape)
    print("test_images shape:", test_images.shape)

    print("train_labels shape:", train_labels.shape)
    # print("valid_labels shape:", valid_labels.shape)
    print("test_labels shape:", test_labels.shape)

    # Shuffle data
    pattern = np.arange(0, train_labels.size)
    # print(pattern.shape)
    np.random.shuffle(pattern)
    train_images = train_images[pattern]
    train_labels = train_labels[pattern]

    # pattern = np.arange(0, valid_labels.size)
    # print(pattern.shape)
    np.random.shuffle(pattern)
    # valid_images = valid_images[pattern]
    # valid_labels = valid_labels[pattern]

    # Normalize data to [-1, 1] range
    print('Normalize image to [-1.0,1.0] range')
    train_images = train_images/128.0
    # valid_images = valid_images/128.0
    test_images = test_images/128.0

    # print("train_images min:", train_images.min())
    # print("train_image max:", train_images.max())
    # print("train_labels min:", train_labels.min())
    # print("train_labels max:", train_labels.max())

    # Create a custom learning rate
    lr_init = learningrate  # needed for clamping
    print("Init LR:", lr_init)

    # callbacks
    callbacks = [
        md.lr_schedule,  # type: ignore[attr-defined] # noqa: F821
        tensorboard_callback,
        tf.keras.callbacks.EarlyStopping(patience=10, mode='min', monitor= 'loss')
    ]
    # callbacks=[EarlyStopping(patience=10, mode='min', monitor= 'loss') ]
    # Build the tf.keras model using the Keras model subclassing API:
    model = md.model  # type: ignore[attr-defined] # noqa: F821

    # select optimizer
    if model_optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_init, beta_1=0.9,
                                             beta_2=0.999, epsilon=1e-07)
    elif model_optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(lr=lr_init, momentum=0.0)
    else:
        raise RuntimeError('optimizer not implemented!')

    # compile model with optimizer
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics])
    kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    for train_idx, val_idx in kfold.split(train_images,train_labels):
        #
    # Train model, specify number of epochs and batch size
        history = model.fit(
            train_images[train_idx],
            train_labels[train_idx],
            validation_data=(train_images[val_idx], train_labels[val_idx]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=VERBOSE)

        # Evaluate model with test set
        test_loss, test_acc = model.evaluate(
            test_images, test_labels, verbose=VERBOSE)

        print("Test Accuracy:", test_acc)

    # print confusion matrix
    predict = np.argmax(model.predict(test_images), axis=1)
    print("Confusion Matrix:")
    print(tf.math.confusion_matrix(test_labels, predict))

    # create probability model
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # predicted outcome
    predict_soft = probability_model.predict(test_images)
    predict_soft_index = np.argmax(predict_soft, axis=1)

    print("Check prediction for some indexes:")
    selected = 0

    num_samples = 1 if not save_sample_per_class else test_labels.size
    for i in range(num_samples):
        index = randint(0, test_labels.size-1)

        if save_sample_per_class and test_labels[index] != selected:
            continue
        conf = predict_soft[index][predict_soft_index[index]]
        # only for classes with high confidence
        if save_sample_per_class and conf < 0.75:
            continue

        print("\t\nindex: %d: predicted: %d(%.2f) actual: %d" % (index, predict[index], conf,
                                                                 test_labels[index]))
        selected += 1

        # Adjust the shape similar to the model shape
        print("\tSave sample data")
        sample_image = np.expand_dims(
            np.array(test_images[index], dtype=np.float32), 0)
        prediction = model.predict(sample_image)
        # print('\tSample index:', index)
        print('\tPrediction:', prediction)
        print(
            f'\tPredicted: {np.argmax(prediction)}({class_names[np.argmax(prediction)]})'
        )
        print(
            f'\tExpected: {test_labels[index]}({class_names[int(test_labels[index])]})'
        )

        # save sample data and all predictions in [-0.5,0.5] range
        # np.savez(
        #    logdir + '/sampledata_class_' +
        #    f"{test_labels[index]}_all_predictions",
        #    sample_image=sample_image,
        #    prediction=prediction)

        # scale back image to [-128,127] before storing to a file
        sample_image = sample_image * 256

        print(f'\tSaving sample image in [{sample_image.min()},{sample_image.max()}] range')

        # reformat as needed
        form, sample_image = reformat_sample(sample_image, channelfirst, swap)

        # save as pty
        path = os.path.join(logdir, 'sampledata_class-' +
                            f'{test_labels[index]}_pred-{np.argmax(prediction)}_' + form)
        np.save(path, np.array(sample_image, dtype=np.int32))

        # end if one sample per class is saved
        if save_sample_per_class and selected > len(class_names):
            break

    # print model
    model.summary()

    # weights range
    print("Weight range:")
    for i, layer in enumerate(model.layers):
        weight = (layer.get_weights()[0:1])  # weights
        bias = (layer.get_weights()[1:2])  # bias
        print("\t[%d]-%s:\t Wmin= %.4f, Wmax= %.4f, Bmin= %.4f, Bmax= %.4f" %
              (i, layer.get_config()['name'], tf.math.reduce_min(weight),
               tf.math.reduce_max(weight), tf.math.reduce_min(bias),
               tf.math.reduce_min(bias)))

    # save a model
    saved_model_dir = os.path.join(logdir, model_dataset + '_SavedModel')
    tf.saved_model.save(model, saved_model_dir)
    print('done here')
    # save a copy to onnx folder in export dir
    expdir = os.path.join('export', model_dataset)
    if not os.path.isdir(expdir):
        os.makedirs(expdir)
    tf.saved_model.save(model, expdir)
    print('done again')
    # save a copy of sample data in export dir
    print(sample_index)
    if sample_index:
        index = sample_index
        print(f'Saving a sampledata file of index {index} into export dir')
        sample_image = np.expand_dims(
            np.array(test_images[index], dtype=np.float32), 0)
        prediction = model.predict(sample_image)

        # reformat as needed
        form, sample_image = reformat_sample(sample_image, channelfirst, swap)

        np.save(os.path.join(expdir, 'sampledata'),
                np.array(sample_image * 128, dtype=np.int32))
        fn = open(os.path.join(expdir, 'sampledata.log'), 'w+')
        print(f'index: {index}\nactual class:{test_labels[index]}')
        fn.writelines(f'index: {index}\nactual class:{test_labels[index]}\n')
        print(f'predicted:{np.argmax(prediction)}\npredictions:\n{prediction}')
        fn.writelines(f'predicted:{np.argmax(prediction)}\npredictions:\n{prediction}\n')
        fn.close()

    # print graphical model
    tf.keras.utils.plot_model(
        model, to_file=os.path.join(logdir, 'model.png'), show_shapes=True)

    # return log folder
    sys.exit(saved_model_dir)
