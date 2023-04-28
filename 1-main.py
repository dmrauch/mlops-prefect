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
import prefect_trial.module1

# %%
prefect_trial.module1.hello_flow()

# %%
