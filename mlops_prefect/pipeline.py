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
    classifier = mlops_prefect.model.train(df, algorithm=algorithm)

    # [ ] calculate predictions for the entire dataset
    # [ ] possible in parallel:
    #   [ ] evaluate the model
    #   [ ] calculate feature permutation importance
    #   [ ] plot the true, predicted and misclassified point clouds

    print("Finished the flow!")

    return df, classifier
