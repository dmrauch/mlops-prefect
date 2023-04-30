import pandas as pd
import prefect
import prefect.tasks
import sklearn.tree
import sklearn.ensemble
from typing import List


algorithms = {
    'DecisionTree': sklearn.tree.DecisionTreeClassifier,
    'RandomForest': sklearn.ensemble.RandomForestClassifier,
    'ExtraTrees': sklearn.ensemble.ExtraTreesClassifier,
    'AdaBoost': sklearn.ensemble.AdaBoostClassifier,
    'GradientBoosting': sklearn.ensemble.GradientBoostingClassifier,
    'HistGradientBoosting': sklearn.ensemble.HistGradientBoostingClassifier
}


@prefect.task(task_run_name='train-{algorithm}',
              cache_key_fn=prefect.tasks.task_input_hash)
def train(df: pd.DataFrame,
          features: List[str] = ['x', 'y', 'z'],
          algorithm: str = 'DecisionTree'):

    if algorithm not in algorithms.keys():
        raise ValueError("'algorithm' has to be one of {}".format(
            list(algorithms.keys())))

    # locate the training samples
    loc_train = (df.dataset == 'train')

    # initialise and fit the model
    # [ ] add the restriction of the dataframe to the configured features
    #     as part of a scikit-learn model pipeline
    model = (
        algorithms[algorithm](random_state=0)
        .fit(X=df.loc[loc_train, features],
            y=df.loc[loc_train, 'cluster'])
    )
    return model
