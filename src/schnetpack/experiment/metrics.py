from sacred import Ingredient

from schnetpack.train.hooks import SacredHook, TensorboardHook, CSVHook
import schnetpack.metrics

from schnetpack.datasets import QM9, MD17, ANI1, MaterialsProject

metric_ingredient = Ingredient('metrics')


class Metrics:
    mae = 'mae'
    rmse = 'rmse'
    lmae = 'length_mae'
    lrmse = 'length_rmse'
    amae = 'angle_mae'
    armse = 'angle_rmse'

    measures = {
        mae: schnetpack.metrics.MeanAbsoluteError,
        rmse: schnetpack.metrics.RootMeanSquaredError,
        lmae: schnetpack.metrics.LengthMAE,
        lrmse: schnetpack.metrics.LengthRMSE,
        amae: schnetpack.metrics.AngleMAE,
        armse: schnetpack.metrics.AngleRMSE
    }


class LoggerError(Exception):
    pass


class MetricError(Exception):
    pass


@metric_ingredient.config
def cfg():
    metrics = {}

    logger = 'sacred'
    log_train_loss = True
    log_validation_loss = True
    log_learning_rate = True
    every_n_epochs = 1


@metric_ingredient.named_config
def md17():
    metrics = {
        MD17.energy: [Metrics.mae, Metrics.rmse],
        MD17.forces: [Metrics.mae, Metrics.rmse, Metrics.lmae, Metrics.lrmse, Metrics.amae, Metrics.armse]
    }


@metric_ingredient.capture
def get_metrics(metrics, logger, log_train_loss, log_validation_loss, log_learning_rate, every_n_epochs,
                experiment=None, log_path=None, property_map={}):
    to_monitor = []

    for property in metrics.keys():
        if property not in property_map.keys():
            raise MetricError('Unrecognized propery {:s}'.format(property))

        db_property = property_map[property]

        for metric in metrics[property]:
            if metric in Metrics.measures:
                to_monitor.append(
                    Metrics.measures[metric](db_property, property)
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
