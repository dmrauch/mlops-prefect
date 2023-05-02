import pandas as pd
import prefect
import prefect.runtime
import prefect.tasks
import sklearn.compose
import sklearn.ensemble
import sklearn.model_selection
import sklearn.tree
import sklearn.pipeline
from typing import Dict, List, Union

import mlops_prefect.cv


# allowed classification algorithms
eligible_algorithms = {
    'DecisionTree': sklearn.tree.DecisionTreeClassifier,
    'RandomForest': sklearn.ensemble.RandomForestClassifier,
    'ExtraTrees': sklearn.ensemble.ExtraTreesClassifier,
    'AdaBoost': sklearn.ensemble.AdaBoostClassifier,
    'GradientBoosting': sklearn.ensemble.GradientBoostingClassifier,
    'HistGradientBoosting': sklearn.ensemble.HistGradientBoostingClassifier
}

def get_algorithm(classifier: sklearn.pipeline.Pipeline) -> str:
    '''
    Get the name of the algorithm from the classifier pipeline instance

    Returns
    -------
    The name of the classification algorithm. One of the keys of
    `eligible_algorithms`.
    '''
    matches = [k for k, v in eligible_algorithms.items()
                 if isinstance(classifier[-1], v)]
    if len(matches) == 0:
        raise TypeError('No matching algorithms found for classifier')
    elif len(matches) > 1:
        raise TypeError('Multiple algorithms found for classifier:', matches)
    return matches[0]


def get_train_task_run_name() -> str:
    parameters = prefect.runtime.task_run.parameters

    algorithm = parameters['algorithm']
    if (isinstance(algorithm, str)
            or (len(algorithm) == 1
                and all([isinstance(alg, str) for alg in algorithm]))):
        # only a single algorithm
        return f'train-{algorithm}'
    elif len(algorithm) > 1 and all([isinstance(alg, str) for alg in algorithm]):
        # ensemble of multiple classifiers
        return f'train-VotingClassifier'

@prefect.task(task_run_name=get_train_task_run_name,
              cache_key_fn=prefect.tasks.task_input_hash)
def train(df: pd.DataFrame,
          features: List[str] = ['x', 'y', 'z'],
          dataset_col: str = 'dataset',
          target_col: str = 'cluster',
          algorithm: Union[str, frozenset[str]] = 'DecisionTree'
          ) -> sklearn.pipeline.Pipeline:

    # transformer: drop all but the feature columns
    column_filter = sklearn.compose.ColumnTransformer(
        [('feature_filter', 'passthrough', features)],
        verbose_feature_names_out=False  # do not rename the columns
    )

    # validate the inputs
    algorithm_not_supported = ValueError(
        'Only the following algorithms are supported: {}'.format(
            list(eligible_algorithms.keys())))

    # instantiate the classifier
    if isinstance(algorithm, str):
        # single model
        if algorithm not in eligible_algorithms.keys():
            raise algorithm_not_supported

        classifier = eligible_algorithms[algorithm](random_state=0)

    elif isinstance(algorithm, frozenset):
        # voting ensemble of multiple classifiers
        for alg in algorithm:
            if alg not in eligible_algorithms.keys():
                raise algorithm_not_supported
        
        # [ ] implement VotingClassifier
        raise NotImplementedError('VotingClassifier not yet implemented')

    # assemble the full pipeline
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
            ('column_filter', column_filter),
            ('classifier', classifier)
        ]
    )

    # fit the pipeline, i.e. train the model
    loc_train = (df[dataset_col] == 'train')
    return pipeline.fit(X=df.loc[loc_train],
                        y=df.loc[loc_train, target_col])



def get_cross_validate_task_run_name() -> str:
    parameters = prefect.runtime.task_run.parameters
    classifier = parameters['classifier']
    return f'cross_validate-{get_algorithm(classifier)}'

@prefect.task(task_run_name=get_cross_validate_task_run_name)
def cross_validate(classifier: sklearn.pipeline.Pipeline,
                   df: pd.DataFrame,
                   dataset_col: str = 'dataset',
                   target_col: str = 'cluster',
                   n_splits: int = 5) -> Dict:

    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=0)

    # get the target class composition in each CV dataset
    class_composition = mlops_prefect.cv.get_class_composition(df, cv)

    # run the cross-validation and collect the metrics
    # [ ] add more metrics
    result = sklearn.model_selection.cross_validate(
        classifier,
        X=df.loc[df[dataset_col] == 'train'],
        y=df.loc[df[dataset_col] == 'train', target_col],
        # scoring='matthews_corrcoef',
        # scoring=['matthews_corrcoef'],
        scoring=['matthews_corrcoef', 'accuracy'],
        cv=cv,
        return_train_score=True
    )
    return class_composition, result
