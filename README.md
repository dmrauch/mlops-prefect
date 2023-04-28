# Data Pipelines and Workflow Orchestration with Prefect


## Resources

- homepage: https://www.prefect.io/
- GitHub: https://github.com/prefecthq/prefect
- Documentation: https://docs.prefect.io/latest/
  - Concepts: https://docs.prefect.io/latest/concepts/
  - Tutorials: https://docs.prefect.io/latest/tutorials/
- Integrations: https://docs.prefect.io/latest/integrations/
  - Juypter (based on papermill): https://prefecthq.github.io/prefect-jupyter/
  - GCP: https://prefecthq.github.io/prefect-gcp/


## Getting Started

- Use `jupytext` to convert the Python scripts in the root folder (such as `1-main.py`) to Jupyter notebooks:

    ```
    $ jupytext --set-formats ipynb,py:percent 1-main.py
    ```

- After modifying a notebook, sync the `.ipynb` and the `.py` files with

    ```
    $ jupytext --sync 1-main.ipynb
    ```

- Spin up a Prefect server:

    ```
    $ prefect server start
    ```

    This will by default start the web UI at http://127.0.0.1:4200


## To Do

- TODO: Add `black` and other linting and code formatting tools
- TODO: Turn this into an installable package
- TODO: Handle requirements and development environment
- TODO: Add documentation skeleton
- TODO: Add Prefect/Juypter integration
