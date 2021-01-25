import logging

from utils.file.file import read_experiment_config, save_experiment_config, parse_experiment_config


def single_repeated(experiment_id, number_training_repetitions, runner, run=True, random_seed=None):
    experiments = []
    config = read_experiment_config(experiment_id)

    for r in range(number_training_repetitions):
        new_experiment_id = '{}__r{}'.format(experiment_id, r)

        if run:
            logging.info('Generating sub-experiment "{}" based on main experiment "{}"'.format(new_experiment_id, experiment_id))

            config.remove_option('hyperparameters_general', 'number_training_repetitions')
            config.remove_option('hyperparameters_general', 'validation_type')

            if random_seed is not None:
                config.set('environment', 'random_seed', str(random_seed + r))

            save_experiment_config(config, new_experiment_id)

        experiments.append(new_experiment_id)

    if run:
        runner.run_list_of_experiments(experiments)
    else:
        return experiments


def cross_validation(experiment_id, folds_cross_validation, split_id, runner, run=True):
    experiments = []
    config = read_experiment_config(experiment_id)

    for k in range(folds_cross_validation):
        new_experiment_id = '{}__k{}'.format(experiment_id, k)

        if run:
            logging.info('Generating sub-experiment "{}" based on main experiment "{}"'.format(new_experiment_id, experiment_id))

            config.remove_option('hyperparameters_general', 'validation_type')
            config.remove_option('hyperparameters_general', 'folds_cross_validation')
            config.set('data', 'split_id', '{}_k{}'.format(split_id, k))

            save_experiment_config(config, new_experiment_id)

        experiments.append(new_experiment_id)

    if run:
        runner.run_list_of_experiments(experiments)
    else:
        return experiments


def bootstrapping(experiment_id, bootstrapping_n, split_id, runner, run=True):
    experiments = []
    config = read_experiment_config(experiment_id)

    for n in range(bootstrapping_n):
        new_experiment_id = '{}__n{}'.format(experiment_id, n)

        if run:
            logging.info('Generating sub-experiment "{}" based on main experiment "{}"'.format(new_experiment_id, experiment_id))

            config.remove_option('hyperparameters_general', 'validation_type')
            config.remove_option('hyperparameters_general', 'bootstrapping_n')
            config.set('data', 'split_id', '{}_n{}'.format(split_id, n))

            save_experiment_config(config, new_experiment_id)

        experiments.append(new_experiment_id)

    if run:
        runner.run_list_of_experiments(experiments)
    else:
        return experiments


def cross_validation_repeated(experiment_id, folds_cross_validation, number_training_repetitions, split_id, runner, run=True, random_seed=None):
    experiments = []
    config = read_experiment_config(experiment_id)

    for r in range(number_training_repetitions):
        for k in range(folds_cross_validation):
            new_experiment_id = '{}__k{}_r{}'.format(experiment_id, k, r)

            if run:
                logging.info('Generating sub-experiment "{}" based on main experiment "{}"'.format(new_experiment_id, experiment_id))

                config.remove_option('hyperparameters_general', 'number_training_repetitions')
                config.remove_option('hyperparameters_general', 'validation_type')
                config.remove_option('hyperparameters_general', 'folds_cross_validation')
                config.set('data', 'split_id', '{}_k{}'.format(split_id, k))

                if random_seed is not None:
                    config.set('environment', 'random_seed', str(random_seed + r))

                save_experiment_config(config, new_experiment_id)

            experiments.append(new_experiment_id)

    if run:
        runner.run_list_of_experiments(experiments)
    else:
        return experiments


def bootstrapping_repeated(experiment_id, bootstrapping_n, number_training_repetitions, split_id, runner, run=True, random_seed=None):
    experiments = []
    config = read_experiment_config(experiment_id)

    for r in range(number_training_repetitions):
        for n in range(bootstrapping_n):
            new_experiment_id = '{}__n{}_r{}'.format(experiment_id, n, r)

            if run:
                logging.info('Generating sub-experiment "{}" based on main experiment "{}"'.format(new_experiment_id, experiment_id))

                config.remove_option('hyperparameters_general', 'number_training_repetitions')
                config.remove_option('hyperparameters_general', 'validation_type')
                config.remove_option('hyperparameters_general', 'bootstrapping_n')
                config.set('data', 'split_id', '{}_n{}'.format(split_id, n))

                if random_seed is not None:
                    config.set('environment', 'random_seed', str(random_seed + r))

                save_experiment_config(config, new_experiment_id)

            experiments.append(new_experiment_id)

    if run:
        runner.run_list_of_experiments(experiments)
    else:
        return experiments


def derive_validation_method_from_experiment_config(experiment_id):
    params = parse_experiment_config(experiment_id)
    repeated = params['number_training_repetitions'] not in [None, 0]

    if params['validation_type'] in [None, 'single']:
        if repeated:
            return 'repeated_single'
        else:
            return 'single'
    elif params['validation_type'] == 'cross_validation':
        if repeated:
            return 'repeated_cross_validation'
        else:
            return 'cross_validation'
    elif params['validation_type'] == 'bootstrapping':
        if repeated:
            return 'repeated_bootstrapping'
        else:
            return 'bootstrapping'
