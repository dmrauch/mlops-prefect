import datetime as dt
import pandas as pd
import prefect
import sklearn.pipeline
from typing import Dict, Tuple, Union

from sklearn import set_config
set_config(transform_output = "pandas")

import mlops_prefect.cv
import mlops_prefect.data
import mlops_prefect.transform
import mlops_prefect.model


@prefect.task
def task_hello() -> None:
    '''
    Dummy task that prints out a greeting
    '''
    print('Hello and welcome to the first task!')


@prefect.flow(name='cluster-classification',
              flow_run_name=dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
def run(n_dims: int = 2,
        algorithms: Union[str, frozenset[str]] = 'DecisionTree',
        cv_metric: str = 'matthews_corrcoef'
        ) -> Tuple[pd.DataFrame,
                   sklearn.pipeline.Pipeline,
                   pd.DataFrame,
                   pd.DataFrame]:
    '''
    Complete pipeline from data generation to model validation

    Parameters
    ----------
    algorithms
        If one algorithm is given, a single classifier is trained and returned.
        If more than one algorithm is specified, multiple classifiers are
        trained (one model for each classifier) and the best-performing model
        is returned. The metric used for determining the best-performing is
        specified with the :param:`cv_metric` parameter.
    cv_metric
        When multiple algorithms are specified with the :param:`algorithm`
        parameter, the model performance is evaluated by means of cross-
        validation (CV), stratified by the target classes, taking the mean
        value of the metric specified by `cv_metric` of the metric values
        obtained from the different CV validation folds.
    '''
    print("This is a minimal flow - let's start!")
    task_hello()

    # generate synthetic data
    df = mlops_prefect.data.generate(n_dims=n_dims)

    # transform cartesian to polar coordinates
    if n_dims == 2:
        df = mlops_prefect.transform.cartesian_to_polar(df)
    elif n_dims == 3:
        df = mlops_prefect.transform.cartesian_to_spherical(df)

    # train/test split: add a 'dataset' column to the dataframe
    df = mlops_prefect.data.split(df)

    # instantiate and train the ML classification model
    # [ ] calculate new derived features as part of the model pipeline
    # [ ] add an optional feature selection/elimination step
    #   [ ] if this is switched off, all the features should be used
    # [ ] make the `algorithm` parameter accept a frozenset, in which case
    #     all the algorithms should be tested and the best one should be
    #     chosen based on a cross-validation strategy
    #   [ ] with an additional parameter, the cross-validation should be
    #       prevented and models based on all specified algorithms should
    #       be trained and combined to an ensemble model
    # [ ] add an optional nested dictionary with the algorithms as the keys
    #     and the values specifying the hyperparameter space for a
    #     hyperparameter optimisation
    if isinstance(algorithms, str):
        algorithms = frozenset({algorithms})
    classifiers = {}
    cv_futures = {}
    # perform model training and cross-validation in parallel
    # - submitting the calculation with `.submit()` allows for parallel
    #   task execution where possible, but returns a future that later
    #   has to be resolved with `.result()` when the results are needed
    # - it is not even necessary to explicitly set
    #   `task_runner=prefect.task_runners.ConcurrentTaskRunner()`
    #   in the flow decorator because the ConcurrentTaskRunner is the
    #   default anyway
    for algorithm in algorithms:

        # train a classifier for each algorithm
        classifiers[algorithm] = mlops_prefect.model.train.submit(
            df, algorithm=algorithm)
        
        # evaluate the model performance
        cv_futures[algorithm] = (
            mlops_prefect.model.cross_validate.submit(
                classifier=classifiers[algorithm], df=df))

    # wait for and gather the results of the parallel execution of the
    # model training and cross-validation
    cv_splits = {}
    cv_results = {}
    for algorithm in algorithms:
        cv_splits[algorithm], cv_results[algorithm] = (
            cv_futures[algorithm].result())

    # collect and aggregate the CV results
    df_cv_results = mlops_prefect.cv.aggregate_cv_results(cv_results)

    # pick the best classifier
    best_algorithm = mlops_prefect.cv.get_best_algorithm(
        classifiers, df_cv_results, cv_metric)

    # [ ] calculate predictions for the entire dataset
    # [ ] possibly in parallel:
    #   [ ] evaluate the model
    #   [ ] calculate feature permutation importance
    #   [ ] plot the true, predicted and misclassified point clouds

    print("Finished the flow!")

    return (df,
            classifiers[best_algorithm],
            cv_splits[best_algorithm],
            df_cv_results)
