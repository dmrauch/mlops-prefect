import datetime as dt
import pandas as pd
import prefect

import mlops_prefect.data


@prefect.task
def task_hello() -> None:
    '''
    Dummy task that prints out a greeting
    '''
    print('Hello and welcome to the first task!')


@prefect.flow(
        name='cluster-classification',
        flow_run_name=dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
def pipeline() -> pd.DataFrame:
    '''
    Complete pipeline from data generation to model validation
    '''
    print("This is a minimal flow - let's start!")
    task_hello()

    # generate synthetic data
    df = mlops_prefect.data.generate()

    # TODO: train ML classification
    # TODO: plot the true, predicted and misclassified point clouds

    print("Finished the flow!")

    return df
