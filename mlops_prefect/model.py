import pandas as pd
import prefect
import prefect.tasks
import sklearn.compose
import sklearn.ensemble
import sklearn.tree
import sklearn.pipeline
from typing import List


# allowed classification algorithms
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
          target: str = 'cluster',
          algorithm: str = 'DecisionTree'):

    # validate the inputs
    if algorithm not in algorithms.keys():
        raise ValueError("'algorithm' has to be one of {}".format(
            list(algorithms.keys())))

    # transformer: drop all but the feature columns
    column_filter = sklearn.compose.ColumnTransformer(
        [('feature_filter', 'passthrough', features)],
        verbose_feature_names_out=False  # do not rename the columns
    )

    # assemble the full pipeline
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
            ('column_filter', column_filter),
            ('classifier', algorithms[algorithm](random_state=0))
        ]
    )

    # fit the pipeline, i.e. train the model
    loc_train = (df.dataset == 'train')
    return pipeline.fit(X=df.loc[loc_train],
                        y=df.loc[loc_train, target])
