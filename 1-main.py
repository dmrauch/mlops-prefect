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
df, classifier = mlops_prefect.pipeline.pipeline(n_dims=3, algorithm='RandomForest')

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

# %%
