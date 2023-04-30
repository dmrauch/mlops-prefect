import colorcet as cc
import holoviews as hv
import hvplot.pandas
import pandas as pd
import prefect
import sklearn.datasets


@prefect.task
def generate(seed: int = 0,
             n_samples: int = 10_000,
             n_modes: int = 3,
             n_dims: int = 2) -> pd.DataFrame:
    '''
    Generate a synthetic dataset of multiple 2D Gaussian distributions

    Parameters
    ----------
    seed
        The seed for the random number generator
    n_samples
        The number of samples to generate
    n_modes
        The number of different Gaussian distributions to create.
        Each of the Gaussian distributions will be sampled with equal
        probability.
    n_dims
        The number of dimensions for which to generate cartesian coordinates.
        Must be either 2 or 3.

    Returns
    -------
    A dataframe containing the following columns:

    - 'cluster': The ID of the Gaussian distribution
    - 'x': The x-coordinate drawn from that specific Gaussian distribution
    - 'y': The y-coordinate drawn from that specific Gaussian distribution
    - 'z': The z-coordinate drawn from that specific Gaussian distribution.
      Only if `n_dims = 3`.
    '''
    if n_dims not in [2, 3]:
        raise ValueError("'n_dims' must be either 2 or 3")

    print('generating pseudo-data')

    # generate the clusters
    X, y = sklearn.datasets.make_blobs(
        n_samples=n_samples,
        n_features=n_dims,
        centers=n_modes,
        random_state=seed
    )

    # turn into a dataframe
    columns = ['x', 'y', 'z']
    df = pd.DataFrame(data=X, columns=columns[:n_dims])
    df['cluster'] = y
    return df


def plot(df: pd.DataFrame,
         height: int = 500) -> hv.Scatter:
    '''
    Plot the point cloud coloured by the respective clusters
    '''
    n_clusters = df.cluster.nunique()
    colormap = cc.glasbey_cool[:n_clusters]

    # Avoiding 'by' on purpose because this plots the points in the order
    # of the groups and therefore the various clusters on top of each other,
    # instead of the points in their original order
    p = df.hvplot.scatter(
        x='x', y='y', color='cluster',
        height=height, responsive=True, colormap=colormap)

    return p
