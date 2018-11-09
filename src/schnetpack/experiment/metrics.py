from sacred import Ingredient

from schnetpack.train.hooks import SacredHook, TensorboardHook, CSVHook
from schnetpack.metrics import MeanAbsoluteError, RootMeanSquaredError

metric_ingredient = Ingredient('metrics')


class LoggerError(Exception):
    pass


class MetricError(Exception):
    pass


@metric_ingredient.config
def cfg():
    metrics = {
        'energy': ['MAE', 'RMSE']
    }

    logger = 'sacred'
    log_train_loss = True
    log_validation_loss = True
    log_learning_rate = True
    every_n_epochs = 1

#
# @metric_ingredient.named_config
# def md17():
#     metrics = {
#         'energy': ['MAE', 'RMSE'],
#         ''
#     }


@metric_ingredient.capture
def get_metrics(metrics, logger, log_train_loss, log_validation_loss, log_learning_rate, every_n_epochs,
                experiment=None, log_path=None):
    to_monitor = []

    for property in metrics.keys():
        for metric in metrics[property]:
            if metric == 'MAE':
                to_monitor.append(
                    MeanAbsoluteError(property, property)
                )
            elif metric == 'RMSE':
                to_monitor.append(
                    RootMeanSquaredError(property, property)
                )
            else:
                raise MetricError('Unrecognized metric {:s}'.format(metric))

    if logger == 'tensorboard':
        if log_path is None:
            raise LoggerError('TensorBoard logger requires log_path.')
        else:
            return TensorboardHook(
                log_path, to_monitor,
                log_train_loss=log_train_loss,
                log_validation_loss=log_validation_loss,
                log_learning_rate=log_learning_rate,
                every_n_epochs=every_n_epochs
            )
    elif logger == 'csv':
        if log_path is None:
            raise LoggerError('CSV logger requires log_path.')
        else:
            return CSVHook(
                log_path, to_monitor,
                log_train_loss=log_train_loss,
                log_validation_loss=log_validation_loss,
                log_learning_rate=log_learning_rate,
                every_n_epochs=every_n_epochs
            )
    elif logger == 'sacred':
        if experiment is None:
            raise LoggerError('Sacred logger requires experiment object.')
        else:
            return SacredHook(
                experiment, to_monitor,
                log_train_loss=log_train_loss,
                log_validation_loss=log_validation_loss,
                log_learning_rate=log_learning_rate,
                every_n_epochs=every_n_epochs
            )
