import pandas as pd
import prefect
from sklearn.model_selection import BaseCrossValidator
import sklearn.pipeline
from typing import Any, Dict


def get_class_composition(data: pd.DataFrame,
                          cv: BaseCrossValidator,
                          target_col: str = 'cluster',
                          dataset_col: str = 'dataset',
                          class_labels: Dict[Any, str] = None,
                          show_test: bool = True) -> pd.DataFrame:
    '''
    Create a table with the class compositions in each cross-validation split

    Parameters
    ----------
    data
        The data. Must contain the columns specified by `target_col` and
        `dataset_col`.
    cv
        The cross-validator instance
    target_col
        The name of the column containing the classification targets. The
        entries in that column are taken as the class values.
    dataset_col
        The name of the column specifying to which dataset a sample belongs.
        Samples with the dataset `train` are subjected to the cross-validation
        procedure, while samples with the dataset `test` are held out and
        can be shown in a separate column by setting `show_test = True`.
    class_labels
        Optional mapping of the class values given in the target column
        to strings specifying the class labels. If given, an additional
        index level will be inserted into the results dataframe.
    show_test
        Show the composition of the test dataset as well if it is included
        in the data

    Returns
    -------
    A table showing the absolute and relative fractions of the various
    classes in the different cross-validation folds as well as, optionally,
    in the test dataset.
    '''

    # number of CV splits
    n_splits = cv.get_n_splits()

    # class/target IDs
    # [ ] class labels can be added optionally
    classes = sorted(data.cluster.unique())

    # prepare the results dataframe columns
    columns = pd.MultiIndex.from_product(
        ([f'#{i}' for i in range(n_splits)], ['train', 'val']),
        names=('split', 'dataset'))
    if show_test and 'test' in data[dataset_col].unique():
        columns = columns.append(
            pd.MultiIndex.from_tuples([('test', '')],
                                      names=('split', 'dataset')))

    # prepare the results dataframe index
    index_levels = [['absolute', 'relative'], classes]
    index_names = ['composition', 'class']
    index = pd.MultiIndex.from_product(
        index_levels, names=index_names)

    # initialise the empty results dataframe
    dfx = pd.DataFrame(columns=columns, index=index, data=0)

    # get the absolute class compositions in each CV split
    for i, (train_idx, val_idx) in enumerate(
            cv.split(X=data.loc[data[dataset_col] == 'train'],
                     y=data.loc[data[dataset_col] == 'train', target_col])):

        # Note that the indices returned by cv.split are row numbers and
        # *not* the proper dataframe indices (row labels). Therefore, the
        # dataframe has to be accessed with iloc to retrieve the correct rows.

        # composition of the training sets
        for class_id, count in (
                data[target_col].iloc[train_idx].value_counts().items()):
            dfx.at[('absolute', class_id), (f'#{i}', 'train')] = count

        # composition of the validation sets
        for class_id, count in (
                data[target_col].iloc[val_idx].value_counts().items()):
            dfx.at[('absolute', class_id), (f'#{i}', 'val')] = count

    # optionally add the test dataset
    if show_test and 'test' in data[dataset_col].unique():
        test_idx = data[dataset_col] == 'test'
        for class_id, count in (
                data.loc[test_idx, target_col].value_counts().items()):
            dfx.at[('absolute', class_id), ('test', '')] = count

    # calculate the number of samples in each CV split
    dfx_abs = dfx.xs('absolute', axis='index', level='composition')
    dfx.loc[('absolute', 'SUM'), (slice(None), slice(None))] = (
        dfx_abs.sum(axis='index'))

    # calculate the relative compositions
    dfx_rel = (dfx_abs / dfx.loc[('absolute', 'SUM')]).astype(float)
    dfx.loc[('relative', slice(None)), (slice(None), slice(None))] = (
        # dfx_rel.round(decimals=3).values)
        dfx_rel.values)
    dfx.loc[('relative', 'SUM'), (slice(None), slice(None))] = (
        dfx_rel.sum(axis='index'))

    # optionally add class labels
    if class_labels is not None:

        # add a new column holding the class labels
        dfx[('class label', '')] = [
            (class_labels[idx] if idx in class_labels else '')
            for idx in dfx.index.get_level_values(level='class')
            ]
        
        # add the class label column to the index
        dfx = (
            dfx
            .reset_index()
            .set_index([('composition', ''), ('class', ''), ('class label', '')])
        )
        dfx.index.rename(('composition', 'class', 'class label'), inplace=True)

    # ensure that the absolute counts are integers
    index = ['absolute', slice(None)]
    if class_labels is not None:
        index.append(slice(None))
    dfx.loc[tuple(index)] = (
        dfx
        .xs('absolute', axis='index', level='composition')
        .astype(int).astype(str).values)

    # ensure that the relative counts are rounded properly
    index = ['relative', slice(None)]
    if class_labels is not None:
        index.append(slice(None))
    for col in dfx.columns:
        dfx.loc[tuple(index), col] = (
            dfx[col]
            .xs('relative', axis='index', level='composition')
            .astype(float).round(3).astype(str)
            .str.pad(width=3+2, side='right', fillchar='0').values)

    return (
        dfx
        .sort_index(axis='index', level='composition')
    )


@prefect.task
def aggregate_cv_results(cv_results: Dict[str, Dict],
                         aggregation: str = 'mean') -> pd.DataFrame:
    '''
    Parameters
    ----------
    cv_results
        Dictionary with the algorithm names as the keys and the CV results
        dictionaries returned by :func:`sklearn.model_selection.cross_validate`
        as they values
    aggregation
        The aggregation function to apply to each metric across the different
        CV folds
    '''
    datasets = set([key.split('_')[0]
                    for cv_result in cv_results.values()
                    for key in cv_result.keys()])
    metrics = set([key.replace('test_', '')
                for cv_result in cv_results.values()
                for key in cv_result.keys() if 'test_' in key])

    dataset_map = {'train': 'train', 'test': 'val'}
    cv_result_cols = {
        f'{dataset_old}_{m}': (m, dataset_new, 'values')
        for m in metrics
        for dataset_old, dataset_new in dataset_map.items()
        if dataset_old in datasets
    }

    cv_result_rows = list(cv_results.keys())
    df_cv_results = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(list(cv_result_cols.values()),
                                        names=('metric', 'dataset', 'aggregation')),
        index=cv_result_rows
    )
    df_cv_results.index.name = 'algorithm'
    for alg, cv_result in cv_results.items():
        for col_old, col_new in cv_result_cols.items():
            df_cv_results.at[alg, col_new] = cv_result[col_old]

    # calculate the mean and standard deviation
    for col in cv_result_cols.values():
        exploded = df_cv_results[col].explode()
        df_cv_results[tuple([*col[:2], aggregation])] = (
            exploded.groupby(exploded.index).agg(aggregation))
        df_cv_results.drop(columns=[col], inplace=True)

    df_cv_results.columns = df_cv_results.columns.droplevel('aggregation')
    df_cv_results = df_cv_results.astype(float)
    return df_cv_results


@prefect.task
def get_best_algorithm(
        classifiers: Dict[str, sklearn.pipeline.Pipeline],
        df_cv_results: pd.DataFrame,
        cv_metric: str = 'mcc') -> sklearn.pipeline.Pipeline:

    best_algorithm = df_cv_results[(cv_metric, 'val')].idxmax()
    return best_algorithm
