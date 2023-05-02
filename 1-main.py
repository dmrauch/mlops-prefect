# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: playground-prefect
#     language: python
#     name: python3
# ---

# %%
# %reload_ext autoreload
# %autoreload 2

# %%
import pandas as pd

import mlops_prefect.pipeline
import mlops_prefect.data

# %% [markdown]
# ## Run the Pipeline

# %%
df, classifier, df_cv_splits, df_cv_results = mlops_prefect.pipeline.run(
    n_dims=3,
    # algorithms=['DecisionTree', 'RandomForest']
    algorithms=['DecisionTree', 'RandomForest', 'GradientBoosting']
)

# %%
df_cv_splits

# %%
print('best model:', mlops_prefect.model.get_algorithm(classifier))
classifier

# %%
df_cv_results

# %% [markdown]
# ## Results: Generated Data

# %%
df

# %%
# plot the cartesian coordinates
mlops_prefect.data.plot(df)

# %% [markdown]
# ### Train/Test Split

# %%
# calculate the overall relative dataset sizes
df.dataset.value_counts()

# %%
# check the stratification of the train/test split:
# calculate the class (= cluster) compositions in each dataset split
(
    # number of samples in each dataset and each cluster
    pd.DataFrame(df.groupby('dataset').cluster.value_counts()
                 .rename('samples')
                 .reset_index()
    )
    # number of samples in each dataset
    .merge(right=df.groupby('dataset').size().rename('dataset size'),
           how='left',
           on='dataset'
    )
    # fraction of each cluster within each dataset
    .assign(**{
        'cluster fraction':
            lambda dfx: dfx['samples']/dfx['dataset size']
    })
    .set_index(['dataset', 'cluster'])
    .reindex([
        ('train', 0), ('train', 1), ('train', 2),
        ('test', 0), ('test', 1), ('test', 2)
    ])
)

# %% [markdown]
# ## Results: Model

# %%
classifier

# %%
# check what the classifier pipeline before the actual model does to the inputs
classifier[:-1].transform(df)

# %%
# calculate all predictions
df = df.assign(prediction=classifier.predict(df))
df

# %% [markdown]
# ### Performance

# %%
import sklearn.metrics

# %%
# variant 1: the score method -> most limited
classifier.score(X=df.loc[df.dataset == 'test', ['x', 'y', 'z']],
                 y=df.loc[df.dataset == 'test', 'cluster'])

# %%
# variant 2: metrics functions 
cm_plot = (
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=df.loc[df.dataset == 'test', 'cluster'],
        y_pred=df.loc[df.dataset == 'test', 'prediction'])
    .plot()
)

# %%
# variant 2 (continued): metrics functions
print(
    sklearn.metrics.classification_report(
        y_true=df.loc[df.dataset == 'test', 'cluster'],
        y_pred=df.loc[df.dataset == 'test', 'prediction'],
        digits=3))

# %% [markdown]
# ## Cross-Validation

# %% [markdown]
# If no feature selection, hyperparameter optimisation or anything else that requires an automated decision is done, then no cross validation is needed as part of the model pipeline. Still, cross-validation may be run manually separately to get an impression of the impact of the dataset splits.

# %%
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate)

import mlops_prefect.cv

# %% [markdown]
# ### Simple: `cross_val_score`()

# %% [markdown]
# *scikit-learn* documentation: [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
#
# - limited to a single metric

# %%
# if the classifier is derived from sklearn.base.ClassifierMixin, the
# cross-validation will automatically use the StratifiedKFold procedure
scores = cross_val_score(classifier,
                         X=df.loc[df.dataset == 'train'],
                         y=df.loc[df.dataset == 'train', 'cluster'])
print(scores)
print(f'-> mean +/- std = {np.mean(scores):.4f} +/- {np.std(scores):.4f}')

# %%
# shuffle = False would also be ok because df contains the points in random
# order and not sorted by cluster
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(classifier,
                         X=df.loc[df.dataset == 'train'],
                         y=df.loc[df.dataset == 'train', 'cluster'],
                         cv=cv)
print(scores)
print(f'-> mean +/- std = {np.mean(scores):.4f} +/- {np.std(scores):.4f}')

# %%
# check the compositions of the CV splits in terms of the target classes
mlops_prefect.cv.get_class_composition(
    df[df.dataset == 'train'],
    cv,
    class_labels={0: 'cluster 0',
                  1: 'cluster 1',
                  2: 'cluster 2'}
)

# %% [markdown]
# ### Advanced: `cross_validate()`

# %% [markdown]
# *scikit-learn* documentation: [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
#
# - supports multiple metrics
# - returns the fit and score times as well as (optionally) the training scores
#   and the fitted models

# %%
import sklearn.metrics

# %%
# single metric
cross_validate(classifier,
               X=df.loc[df.dataset == 'train'],
               y=df.loc[df.dataset == 'train', 'cluster'])

# %%
# - multiple metrics
# - training scores
# - retain all the models
#
# questions:
# [ ] what is the difference between 'micro' and 'weighted'?
#   [ ] why does it make a difference for precision, but not for recall?
# [ ] why do the ROC AUC and average precision scores not work?
cross_validate(
    classifier,
    X=df.loc[df.dataset == 'train'],
    y=df.loc[df.dataset == 'train', 'cluster'],
    scoring={
        # 'roc_auc_ovr_weighted': sklearn.metrics.make_scorer(
        #     sklearn.metrics.roc_auc_score,
        #     multi_class='ovr',
        #     average='weighted'),
        # 'average_precision_weighted': sklearn.metrics.make_scorer(
        #     sklearn.metrics.average_precision_score,
        #     average='weighted'),
        'accuracy': sklearn.metrics.make_scorer(
            sklearn.metrics.accuracy_score),
        'precision_micro': sklearn.metrics.make_scorer(
            sklearn.metrics.precision_score,
            average='micro'),
        'precision_weighted': sklearn.metrics.make_scorer(
            sklearn.metrics.precision_score,
            average='weighted'),
        'recall_micro': sklearn.metrics.make_scorer(
            sklearn.metrics.recall_score,
            average='micro'),
        'recall_weighted': sklearn.metrics.make_scorer(
            sklearn.metrics.recall_score,
            average='weighted'),
        'F1_weighted': sklearn.metrics.make_scorer(
            sklearn.metrics.f1_score,
            average='weighted'),
        'mcc': sklearn.metrics.make_scorer(
            sklearn.metrics.matthews_corrcoef)
    },
    return_train_score=True,
    return_estimator=True)

# %%
