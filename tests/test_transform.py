import numpy as np
import pandas as pd

from mlops_prefect.transform import cartesian_to_polar


def test_cartesian_to_polar():

    df_in = pd.DataFrame(data={
        'x': [1.0, 0.0, -1.0, 0.0],
        'y': [0.0, 1.0, 0.0, -1.0]
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
        pd.Series([1.0, 1.0, 1.0, 1.0], name='r')
    )
    pd.testing.assert_series_equal(
        df_out.phi,
        pd.Series([0.0, np.pi/2, np.pi, -np.pi/2], name='phi')
    )
