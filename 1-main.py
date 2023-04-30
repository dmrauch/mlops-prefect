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

# %%
df = mlops_prefect.pipeline.pipeline(n_dims=3)

# %%
df

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

# %%
# plot the cartesian coordinates
mlops_prefect.data.plot(df)

# %%
