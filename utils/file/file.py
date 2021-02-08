import configparser
import json
import logging
import os
import pickle
import shutil

import pandas as pd
from PyPDF2 import PdfFileWriter, PdfFileReader

#TODO: Consider deletion of methods here after deleting data util methods

def save_dict_as_json(dct, path):
    with open(path, 'w') as fp:
        json.dump(dct, fp)
        fp.close()


def load_dict_from_json(path):
    string = load_string_from_file(path)
    dct = json.loads(string)

    return dct


def save_string_to_file(string, path):
    with open(path, 'w') as fp:
        fp.write(string)
        fp.close()


def load_string_from_file(path):
    with open(path, 'r') as fp:
        string = fp.read()
        fp.close()

    return string


def combine_pdfs(paths, targetpath, cleanup=False):
    pdf_writer = PdfFileWriter()

    for path in paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))

    with open(targetpath, 'wb') as fh:
        pdf_writer.write(fh)

    if cleanup:
        for path in paths:
            os.remove(path)


def pickle_data(array, path):
    with open(path, 'wb') as fp:
        pickle.dump(array, fp, pickle.HIGHEST_PROTOCOL)
        fp.close()


def unpickle_data(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def cleanup_directory(path):
    shutil.rmtree(path, ignore_errors=True)
    logging.debug('Removed directory "{}"'.format(path))


def make_dirs_if_not_present(path):
    os.makedirs(path, exist_ok=True)


def logdir_exists(experiment_id, category):
    logdir = '../../logs/{}/{}'.format(category, experiment_id)
    exists = os.path.exists(os.path.abspath(logdir))

    return logdir, exists


def parse_experiment_config(experiment_id, expdir='../../experiments/'):
    config = read_experiment_config(experiment_id, expdir=expdir)
    return parse_config_parameters(config)


def read_experiment_config(experiment_id, expdir='../../experiments/'):
    path = expdir + experiment_id + '.ini'
    config = configparser.ConfigParser()
    read_ok = config.read(path)

    if len(read_ok) == 0:
        raise Exception('Could not read experiment config from "{}", no such file or directory.'.format(path))

    config.set('environment', 'experiment_id', experiment_id)

    return config


def save_experiment_config(config, experiment_id, expdir='../../experiments/'):
    with open(expdir + experiment_id + '.ini', 'w') as fp:
        config.write(fp)


def parse_config_parameters(config):
    # Mandatory parameters

    params = {
        'experiment_series': config['general'].get('experiment_series'),
        'question': config['general'].get('question'),
        'hypothesis': config['general'].get('hypothesis'),
        'remarks': config['general'].get('remarks'),

        'experiment_id': config['environment'].get('experiment_id'),
        'model_id': config['environment'].get('model_id'),
        'random_seed': config['environment'].getint('random_seed'),
        'preprocessor_id': config['environment'].get('preprocessor_id'),
        'evaluator_id': config['environment'].get('evaluator_id'),
        'gpu_id': config['environment'].get('gpu_id'),
        'loglevel': config['environment'].get('loglevel'),

        'dataset_id': config['data'].get('dataset_id'),
        'split_id': config['data'].get('split_id'),
        'leads_to_use': config['data'].get('leads_to_use').split(','),
        'clinical_parameters_outputs': config['data'].get('clinical_parameters_outputs').split(','),
        'subsampling_factor': config['data'].getint('subsampling_factor'),
        'subsampling_window_size': config['data'].getint('subsampling_window_size'),
        'clinical_parameters_inputs': config['data'].get('clinical_parameters_inputs'),
        'ecg_variants': config['data'].get('ecg_variants').split(','),
        'snapshot_id': config['data'].get('snapshot_id'),
        'record_ids_excluded': config['data'].get('record_ids_excluded'),
        'crop_id': config['data'].get('crop_id'),
        'source_id': config['data'].get('source_id'),
        'metadata_id': config['data'].get('metadata_id'),

        'number_epochs': config['hyperparameters_general'].getint('number_epochs'),
        'optimizer': config['hyperparameters_general'].get('optimizer'),
        'learning_rate': config['hyperparameters_general'].getfloat('learning_rate'),
        'learning_rate_decay': config['hyperparameters_general'].getfloat('learning_rate_decay'),
        'shuffle': config['hyperparameters_general'].getboolean('shuffle'),
        'loss_function': config['hyperparameters_general'].get('loss_function'),
        'number_training_repetitions': config['hyperparameters_general'].getint('number_training_repetitions'),
        'validation_type': config['hyperparameters_general'].get('validation_type'),

        'metrics': config['evaluation'].get('metrics').split(','),
        'calculation_methods': config['evaluation'].get('calculation_methods').split(','),
        'class_names': config['evaluation'].get('class_names').split(','),
        'target_metric': config['evaluation'].get('target_metric'),
        'recipients_emails': config['evaluation'].get('recipients_emails'),
        'tensorboard_subdir': config['evaluation'].get('tensorboard_subdir'),
        'sensitivity_threshold': config['evaluation'].getfloat('sensitivity_threshold'),
        'specificity_threshold': config['evaluation'].getfloat('specificity_threshold'),
        'save_raw_results': config['evaluation'].getboolean('save_raw_results')
    }

    # Optional param group: ecg model
    try:
        params_ecgmodel = {
            'ecgmodel_initializer_conv': config['hyperparameters_ecgmodel'].get('initializer_conv'),
            'ecgmodel_initializer_dense': config['hyperparameters_ecgmodel'].get('initializer_dense'),
            'ecgmodel_number_layers_conv': config['hyperparameters_ecgmodel'].getint('number_layers_conv'),
            'ecgmodel_number_filters_conv': config['hyperparameters_ecgmodel'].getint('number_filters_conv'),
            'ecgmodel_number_layers_dense': config['hyperparameters_ecgmodel'].getint('number_layers_dense'),
            'ecgmodel_number_neurons_dense': config['hyperparameters_ecgmodel'].getint('number_neurons_dense'),
            'ecgmodel_size_kernel_conv': config['hyperparameters_ecgmodel'].getint('size_kernel_conv'),
            'ecgmodel_size_kernel_pool': config['hyperparameters_ecgmodel'].getint('size_kernel_pool'),
            'ecgmodel_stride_conv': config['hyperparameters_ecgmodel'].getint('stride_conv'),
            'ecgmodel_stride_pool': config['hyperparameters_ecgmodel'].getint('stride_pool'),
            'ecgmodel_padding_conv': config['hyperparameters_ecgmodel'].get('padding_conv'),
            'ecgmodel_pooling_conv': config['hyperparameters_ecgmodel'].getboolean('pooling_conv'),
            'ecgmodel_pooling_dense': config['hyperparameters_ecgmodel'].getboolean('pooling_dense'),
            'ecgmodel_dropout_conv': config['hyperparameters_ecgmodel'].getboolean('dropout_conv'),
            'ecgmodel_dropout_dense': config['hyperparameters_ecgmodel'].getboolean('dropout_dense'),
            'ecgmodel_dropout_rate_conv': config['hyperparameters_ecgmodel'].getfloat('dropout_rate_conv'),
            'ecgmodel_dropout_rate_dense': config['hyperparameters_ecgmodel'].getfloat('dropout_rate_dense'),
            'ecgmodel_activation_function_conv': config['hyperparameters_ecgmodel'].get('activation_function_conv'),
            'ecgmodel_activation_function_dense': config['hyperparameters_ecgmodel'].get('activation_function_dense')
        }
        params.update(params_ecgmodel)
    except:
        pass

    # Optional param group: clinical parameter model
    try:
        params_clinparammodel = {
            'clinicalparametermodel_initializer_dense': config['hyperparameters_clinicalparametermodel'].get('initializer_dense'),
            'clinicalparametermodel_dropout_dense': config['hyperparameters_clinicalparametermodel'].getboolean('dropout_dense'),
            'clinicalparametermodel_dropout_rate_dense': config['hyperparameters_clinicalparametermodel'].getfloat('dropout_rate_dense'),
            'clinicalparametermodel_activation_function_dense': config['hyperparameters_clinicalparametermodel'].get('activation_function_dense'),
            'clinicalparametermodel_number_neurons_dense': config['hyperparameters_clinicalparametermodel'].getint('number_neurons_dense'),
            'clinicalparametermodel_number_layers_dense': config['hyperparameters_clinicalparametermodel'].getint('number_layers_dense')
        }
        params.update(params_clinparammodel)
    except:
        pass

    # Optional param group: combination model
    try:
        params_ecgmodel = {
            'combinationmodel_initializer_dense': config['hyperparameters_combinationmodel'].get('initializer_dense'),
            'combinationmodel_dropout_dense': config['hyperparameters_combinationmodel'].getboolean('dropout_dense'),
            'combinationmodel_dropout_rate_dense': config['hyperparameters_combinationmodel'].getfloat('dropout_rate_dense'),
            'combinationmodel_activation_function_dense': config['hyperparameters_combinationmodel'].get('activation_function_dense'),
            'combinationmodel_number_neurons_dense': config['hyperparameters_combinationmodel'].getint('number_neurons_dense'),
            'combinationmodel_number_layers_dense': config['hyperparameters_combinationmodel'].getint('number_layers_dense')
        }
        params.update(params_ecgmodel)
    except:
        pass

    # Optional parameters

    if params['record_ids_excluded'] is not None:
        params['record_ids_excluded'] = params['record_ids_excluded'].split(',')

    if params['clinical_parameters_inputs'] is not None:
        params['clinical_parameters_inputs'] = params['clinical_parameters_inputs'].split(',')

    if params['validation_type'] == 'cross_validation':
        params['folds_cross_validation'] = config['hyperparameters_general'].getint('folds_cross_validation')
        if params['folds_cross_validation'] is None:
            raise Exception('Missing parameter "folds_cross_validation". Aborting.')

    if params['validation_type'] == 'bootstrapping':
        params['bootstrapping_n'] = config['hyperparameters_general'].getint('bootstrapping_n')
        if params['bootstrapping_n'] is None:
            raise Exception('Missing parameter "bootstrapping_n". Aborting.')

    return params


def extract_different_config_parameters(experiments):
    rows = []
    parameters = []

    for e in experiments:
        rows.append(parse_experiment_config(e))

    df = pd.DataFrame(rows)

    for c in df:
        variants = list(df[c])

        if type(variants[0]) is list:
            tmp = []
            for v in variants:
                tmp.append(str(v))
            variants = tmp

        if len(set(variants)) > 1:
            parameters.append(c)

    return parameters