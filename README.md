# Data Pipelines and Workflow Orchestration with Prefect


> This is an example **multi-class classification machine learning** project to showcase tools and best practices in the areas of
> - data science: *scikit-learn*
> - machine learning operations (MLOps): *MLflow*, *Prefect*
> - software development: *Sphinx*


In the end, this repository will contain and showcase the following aspects of an end-to-end machine learning project:

- there will be a pipeline to
  - [X] generate data: cluster IDs and cartesian coordinates
    - [X] 2D
    - [X] 3D
  - [X] transform the data
    - [X] 2D cartesian -> polar coordinates
    - [X] 3D cartesian -> spherical coordinates
  - [X] train an ML model: classify the coordinates to the cluster IDs
  - [ ] evaluate the model
    - [ ] performance metrics
    - [ ] global feature importance
      - [ ] permutation importance
      - [ ] mean Shapley values
    - [ ] local feature importance
      - [ ] Shapley values
  - advanced features
    - [ ] hyperparameter optimisation by means of cross validation
    - [ ] probability calibration: *scikit-learn* [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) and [calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html)
    - [ ] multiple (calibrated) classifiers combined with a *scikit-learn* [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
    - [X] configurable alternative: best-model selection from list of specified algorithms
    - [ ] probabilistic / conformal predictions
    - [ ] add derived features and run automatic feature selection
      - [ ] *scikit-learn*
      - [ ] *tsfresh*
- [ ] the pipeline will be implemented with [*Prefect*](https://www.prefect.io/)
  - [X] use caching of intermediate pipeline results
  - [ ] add Prefect/Juypter integration
  - [ ] try out [parallel subflows](https://docs.prefect.io/latest/concepts/flows/#composing-flows)
    - need to use `.submit` as per [doc: guide](https://docs.prefect.io/latest/guides/dask-ray-task-runners/) and [doc: tutorial](https://docs.prefect.io/latest/tutorials/execution/)
- [ ] experiments will be tracked with *MLflow*
  - [ ] using not the *local filesystem*, but rather the *SQLite* backend store option, in order to support model serving
- best practices
  - [X] use test-driven development
  - [ ] add *Black* and other linting and code formatting tools
  - [ ] automatically check test coverage
  - [ ] an automatic source code documentation built using *Sphinx*
  - [X] development environments and installation requirements should be handled in a clean and consistent way
  - [ ] the code will be built into a package using *Poetry*
  - [ ] everything should run locally, but also in a *Docker* container


## Resources

- Prefect
  - [Homepage](https://www.prefect.io/)
  - [GitHub](https://github.com/prefecthq/prefect)
  - [Documentation](https://docs.prefect.io/latest/)
    - [Concepts](https://docs.prefect.io/latest/concepts/)
    - [Tutorials](https://docs.prefect.io/latest/tutorials/)
  - [Integrations](https://docs.prefect.io/latest/integrations/)
    - [Juypter](https://prefecthq.github.io/prefect-jupyter/) (based on papermill)
    - [GCP](https://prefecthq.github.io/prefect-gcp/)


## Questions

- [ ] How to run *Prefect*-integrated code without *Prefect* (e.g. in an
      environment where this is not supported)?
- [ ] Does *Prefect* have a concept of configuration files to pass parameters to the pipeline or to override default parameters of individual tasks?
- [ ] How best to generate visualisations and dataframe printouts during
  intermediate steps of the pipeline and transport them outside?
  - I'm not sure all of this diagnostic information should be logged to *MLflow*
  - Perhaps that's what [artifact](https://docs.prefect.io/latest/concepts/artifacts/) mechanism is for


## Thoughts and Notes

- Adding 3D coordinates gives an opportunity to use tSNE for creating a 2D visualisation
- New features can be defined in preprocessing steps in the pipeline using
  *Pandas* or as part of a feature engineering step in the model itself using
  *scikit-learn*. My personal thoughts on this are the following:
  - If *all* of the feature engineering can be done in *scikit-learn*, then
    this is preferable because simply exporting the model (as a *scikit-learn*
    pipeline) will include the additional features
  - If there are elements that have to be implemented outside of the
    *scikit-learn* model pipeline, then the proper outer pipeline (i.e. the
    part that is implemented in *Prefect* in this demo) has to be deployed
    anyways and it is preferable to make the pipeline as clean and consistent
    as possible - which may mean limiting the amount of feature engineering
    done with *scikit-learn*.


## Getting Started

### Prepare the Development Environment

- In this package, the runtime/deployment dependencies are listed in the `requirements.txt` file, whereas additional development dependencies are collected in the `playground-prefect.yml` file.
- The `requirements.txt` file, however, is included in the `playground-prefect.yml`
- Therefore, to create the development environment, it is sufficient to run
  ```(bash)
  $ conda env create -f playground-prefect.yml
  ```
  or, alternatively, using the faster *mamba* package manager
  ```(bash)
  $ mamba env create -f playground-prefect.yml
  ```
  which will install the packages listed in the `requirements.txt` into the same environment as well.
- The development environment can then be activated with
  ```(bash)
  $ conda activate playground-prefect
  ```
- The advantage of this structure is that a `requirements.txt` file is provided, which can be used for packaging, while at the same time avoiding having to maintain two partially overlapping dependency lists.


### Use Jupytext to Turn Notebooks Into Equivalent Python Scripts

- Use `jupytext` to convert the Python scripts in the root folder (such as `1-main.py`) to Jupyter notebooks:

    ```
    $ jupytext --set-formats ipynb,py:percent 1-main.py
    ```

- After modifying a notebook, sync the `.ipynb` and the `.py` files with

    ```
    $ jupytext --sync 1-main.ipynb
    ```


### Start Up Prefect

- Spin up a Prefect server:

    ```
    $ prefect server start
    ```

    This will by default start the web UI at http://127.0.0.1:4200


## Prefect Cheat Sheet

### Settings and Profiles

- Specify or change a setting:
  ```(bash)
  $ prefect config set PREFECT_TASKS_REFRESH_CACHE='True'
  ```
- Reset to the default value:
  ```(bash)
  $ prefect config unset PREFECT_TASKS_REFRESH_CACHE
  ```
- View the currently active settings:
  ```(bash)
  $ prefect config view
  ```

#### Profiles

- List all available profiles:
  ```(bash)
  $ prefect profile ls
  ```
- View the settings associated with the currently active profile:
  ```(bash)
  $ prefect profile inspect
  ```
