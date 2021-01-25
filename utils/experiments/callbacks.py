import logging

from tensorflow.keras.callbacks import Callback

from utils.experiments.metrics import calculate_metrics_for_predictions
from utils.experiments.model import save_model, save_model_based_on_thresholds
from utils.experiments.training import save_epoch_result_to_logdir, predict, log_epoch_result_to_tensorboard, \
    initialize_tensorboard_filewriter


class CustomCallbackV1(Callback):
    def __init__(self, x_val, y_val, y_classes, record_ids_val, experiment, experiment_logdir, tensorboard_logdir, metrics=None, calculation_methods=None, class_names_to_log=None, metric_thresholds=None):
        super().__init__()

        if metrics is None:
            self.metrics = ['sensitivity', 'specificity']
        else:
            self.metrics = metrics

        if calculation_methods is None:
            self.calculation_methods = ['sample_level', 'subsample_level']
        else:
            self.calculation_methods = calculation_methods

        self.x_val = x_val
        self.y_val = y_val
        self.experiment = experiment
        self.experiment_logdir = experiment_logdir
        self.y_classes = y_classes
        self.record_ids_val = record_ids_val
        self.class_names_to_log = class_names_to_log
        self.metric_thresholds = metric_thresholds

        initialize_tensorboard_filewriter(tensorboard_logdir + '/custommetrics')

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        # Predict on current state of model
        y_pred = predict(self.model, self.x_val)

        # Calculate performance metrics
        result = calculate_metrics_for_predictions(self.y_val, y_pred, self.y_classes, self.record_ids_val, self.metrics, self.calculation_methods, self.class_names_to_log)

        # Save epoch result to logdir
        save_epoch_result_to_logdir(result, self.experiment_logdir, epoch)

        # Log epoch result to tensorboard
        log_epoch_result_to_tensorboard(epoch, result, self.class_names_to_log, calculation_method=self.calculation_methods[0])

        # Save model snapshot to logdir
        save_model_based_on_thresholds(self.model, self.experiment_logdir, epoch, result[self.class_names_to_log[0]][self.calculation_methods[0]]['metrics'], self.metric_thresholds)

        # analyzer = innvestigate.create_analyzer('lrp.epsilon', self.model)
        # analysis = analyzer.analyze(self.x_val)
        # print(analysis)

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
