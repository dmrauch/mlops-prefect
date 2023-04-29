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
import mlops_prefect.pipeline
import mlops_prefect.data

# %%
df = mlops_prefect.pipeline.pipeline()

# %%
mlops_prefect.data.plot(df)

# %%
