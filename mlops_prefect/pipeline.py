import datetime as dt
import pandas as pd
import prefect

from sklearn import set_config
set_config(transform_output = "pandas")

import mlops_prefect.data
import mlops_prefect.transform
import mlops_prefect.model


@prefect.task
def task_hello() -> None:
    '''
    Dummy task that prints out a greeting
    '''
    print('Hello and welcome to the first task!')


@prefect.flow(
        name='cluster-classification',
        flow_run_name=dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
def pipeline(n_dims: int = 2,
             algorithm: str = 'DecisionTree') -> pd.DataFrame:
    '''
    Complete pipeline from data generation to model validation
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
    # [ ] make the `algorithm` parameter accept lists, in which case
    #     all the algorithms should be tested and the best one should be
    #     chosen based on a cross-validation strategy
    #   [ ] with an additional parameter, the cross-validation should be
    #       prevented and models based on all specified algorithms should
    #       be trained and combined to an ensemble model
    # [ ] add an optional nested dictionary with the algorithms as the keys
    #     and the values specifying the hyperparameter space for a
    #     hyperparameter optimisation
    classifier = mlops_prefect.model.train(df, algorithm=algorithm)

    # [ ] calculate predictions for the entire dataset
    # [ ] possible in parallel:
    #   [ ] evaluate the model
    #   [ ] calculate feature permutation importance
    #   [ ] plot the true, predicted and misclassified point clouds

    print("Finished the flow!")

    return df, classifier
