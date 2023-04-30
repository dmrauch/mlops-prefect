import numpy as np
import pandas as pd

from mlops_prefect.transform import (
    cartesian_to_polar, cartesian_to_spherical)


def test_cartesian_to_polar():

    df_in = pd.DataFrame(data={
        'x': [1.0, 0.0, -1.0, 0.0, 2.0, -2.0],
        'y': [0.0, 1.0, 0.0, -1.0, 0.0, -2.0]
    })
    df_in_shape = df_in.shape

    # apply the transformation
    df_out = cartesian_to_polar.fn(df_in)

    # the original dataframe should not be altered
    assert df_in.shape == df_in_shape

    # existence of the radius and angle columns
    assert 'r' in df_out.columns
    assert 'phi' in df_out.columns

    # correct radii and angles
    pd.testing.assert_series_equal(
        df_out.r,
        pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, np.sqrt(8.0)], name='r')
    )
    pd.testing.assert_series_equal(
        df_out.phi,
        pd.Series([0.0, np.pi/2, np.pi, -np.pi/2, 0.0, -3/4*np.pi], name='phi')
    )


def test_cartesian_to_spherical():

    df_in = pd.DataFrame(data={
        'x': [1.0, 0.0, -1.0,  0.0, 0.0,  0.0, -1.0],
        'y': [0.0, 1.0,  0.0, -1.0, 0.0,  0.0,  0.0],
        'z': [0.0, 0.0,  0.0,  0.0, 1.0, -1.0, -1.0]
    })
    df_in_shape = df_in.shape

    # apply the transformation
    df_out = cartesian_to_spherical.fn(df_in)

    # the original dataframe should not be altered
    # assert df_in.shape == df_in_shape

    # existence of the radius and angle columns
    assert 'r' in df_out.columns
    assert 'theta' in df_out.columns
    assert 'phi' in df_out.columns

    df_exp = pd.DataFrame(data={
        'r': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.sqrt(2.0)],
        'theta': [np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0.0, np.pi, 3/4*np.pi],
        'phi': [0.0, np.pi/2, np.pi, -np.pi/2, 0.0, 0.0, np.pi]
    })
    # correct radii and angles
    pd.testing.assert_frame_equal(
        df_exp,
        df_out.drop(columns=['x', 'y', 'z'])
    )
