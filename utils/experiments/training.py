import logging
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.python.ops.summary_ops_v2 import create_file_writer_v2

from utils.file.file import save_dict_as_json


def enable_reproducibility(random_seed):
    """ Source: https://github.com/NVIDIA/framework-determinism """

    if random_seed is not None:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)


def get_optimizer(name, learning_rate, learning_rate_decay=None):
    if name == 'adam':
        if learning_rate_decay is not None:
            return optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)
        else:
            return optimizers.Adam(lr=learning_rate)
    elif name == 'adadelta':
        return optimizers.Adadelta()
    else:
        raise Exception('Unknown optimizer. Aborting.')


def save_epoch_result_to_logdir(result, logdir, epoch):
    path = '{}/results_e{}.json'.format(logdir, epoch)
    save_dict_as_json(result, path)
    logging.debug('Saved epoch result of epoch {} to logdir.'.format(epoch))


def predict(model, x):
    return np.asarray(model.predict_on_batch(x))


def log_to_tensorboard(epoch, metric_name, metric_value):
    if type(metric_value) not in [list, dict]:
        tf.summary.scalar(metric_name, data=metric_value, step=epoch)


def log_epoch_result_to_tensorboard(epoch, result, class_names_to_log, calculation_method='subsample_level'):
    for y_class in result:
        if y_class in class_names_to_log:
            for metric_name in result[y_class][calculation_method]['metrics']:
                metric_value = result[y_class][calculation_method]['metrics'][metric_name]
                log_to_tensorboard(epoch, '{}_{}_{}'.format(y_class, metric_name, calculation_method), metric_value)


def initialize_tensorboard_filewriter(tensorboard_logdir):
    file_writer = create_file_writer_v2(tensorboard_logdir)
    file_writer.set_as_default()
    logging.debug('Created TensorBoard File Writer at "{}" and set it as default'.format(tensorboard_logdir))
