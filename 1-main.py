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
df, model = mlops_prefect.pipeline.pipeline(n_dims=3)

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
# calculate the relative size of the datasets for each class (= cluster)
(
    # number of samples in each cluster and each dataset
    pd.DataFrame(df.groupby('cluster').dataset.value_counts()
                 .rename('samples')
                 .reset_index()
    )
    # number of samples in each cluster
    .merge(right=df.groupby('cluster').size().rename('class cardinality'),
           how='left',
           on='cluster'
    )
    # fraction of each dataset within each cluster
    .assign(**{
        'dataset fraction':
            lambda dfx: dfx['samples']/dfx['class cardinality']
    })
    .set_index(['cluster', 'dataset'])
)

# %% [markdown]
# ## Results: Model

# %%
model

# %%
df[df.dataset == 'test']

# %%
model.predict(df.loc[df.dataset == 'test', ['x', 'y', 'z']])

# %% [markdown]
# ### Performance

# %%
import sklearn.metrics

# %%
# variant 1: the score method -> most limited
model.score(X=df.loc[df.dataset == 'test', ['x', 'y', 'z']],
            y=df.loc[df.dataset == 'test', 'cluster'])

# %%
# variant 2: metrics functions 
cm_plot = (
    sklearn.metrics.ConfusionMatrixDisplay.from_estimator(
        estimator=model,
        X=df.loc[df.dataset == 'test', ['x', 'y', 'z']],
        y=df.loc[df.dataset == 'test', 'cluster'])
    .plot()
)

# %%
# variant 2 (continued): metrics functions
print(
    sklearn.metrics.classification_report(
        y_true=df.loc[df.dataset == 'test', 'cluster'],
        y_pred=model.predict(X=df.loc[df.dataset == 'test', ['x', 'y', 'z']]),
        digits=3))

# %%
