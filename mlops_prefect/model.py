import pandas as pd
import prefect
import prefect.tasks
import sklearn.tree
from typing import List


@prefect.task(cache_key_fn=prefect.tasks.task_input_hash)
def train(df: pd.DataFrame,
          features: List[str] = ['x', 'y', 'z']):

    loc_train = (df.dataset == 'train')

    # initialise and fit the model
    # [ ] add the restriction of the dataframe to the configured features
    #     as part of a scikit-learn model pipeline
    model = (
        sklearn.tree.DecisionTreeClassifier()
        .fit(X=df.loc[loc_train, features],
             y=df.loc[loc_train, 'cluster'])
    )
    return model
