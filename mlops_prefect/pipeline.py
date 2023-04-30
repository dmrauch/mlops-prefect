import datetime as dt
import pandas as pd
import prefect

import mlops_prefect.data
import mlops_prefect.transform


@prefect.task
def task_hello() -> None:
    '''
    Dummy task that prints out a greeting
    '''
    print('Hello and welcome to the first task!')


@prefect.flow(
        name='cluster-classification',
        flow_run_name=dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
def pipeline(n_dims: int = 2) -> pd.DataFrame:
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

    # train/test split
    # df_train, df_test = mlops_prefect.data.split(df)
    df = mlops_prefect.data.split(df)

    # TODO: train ML classification
    # TODO: plot the true, predicted and misclassified point clouds

    print("Finished the flow!")

    return df
