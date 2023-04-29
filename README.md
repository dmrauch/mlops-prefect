# Data Pipelines and Workflow Orchestration with Prefect


In the end, this repository will contain and showcase the following aspects of an end-to-end machine learning project:

- there will be a pipeline to
  - [ ] generate data: cluster IDs and 3D cartesian coordinates
  - [ ] transform the data: 3D cartesian -> spherical coordinates
  - [ ] train an ML model: classify the coordinates to the cluster IDs
  - [ ] evaluate the model
    - [ ] performance metrics
    - [ ] global feature importance
      - [ ] permutation importance
      - [ ] mean Shapley values
    - [ ] local feature importance
      - [ ] Shapley values
- [ ] the pipeline will be implemented with [*Prefect*](https://www.prefect.io/)
  - [ ] add Prefect/Juypter integration
  - [ ] try out [parallel subflows](https://docs.prefect.io/latest/concepts/flows/#composing-flows)
- [ ] experiments will be tracked with *MLflow*
  - [ ] using not the *local filesystem*, but rather the *SQLite* backend store option, in order to support model serving
- best practices
  - [ ] use test-driven development
  - [ ] add *Black* and other linting and code formatting tools
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
