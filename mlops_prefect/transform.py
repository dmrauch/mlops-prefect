import numpy as np
import pandas as pd
import prefect


@prefect.task
def cartesian_to_polar(df: pd.DataFrame,
                       col_x: str = 'x',
                       col_y: str = 'y',
                       col_r: str = 'r',
                       col_phi: str = 'phi') -> pd.DataFrame:
    '''
    Convert 2D cartesian to polar coordinates

    Parameters
    ----------
    df
        A dataframe containing at least the columns with the 2D cartesian
        coordinates of the points. The expected column names are specified
        by `col_x` and `col_y`.
    col_x
        The name of the input column containing the x-values
    col_y
        The name of the input column containing the y-values
    col_r
        The name of the output column containing the radii
    col_phi
        The name of the output column containing the polar angles

    Returns
    -------
    The dataframe with two additional columns containing the radii
    and polar angles
    '''
    # this would change the original df that was passed in
    # - you can try this out and should see the unit tests fail
    # df[col_r] = np.sqrt(df[col_x]**2 + df[col_y]**2)
    # df[col_phi] = np.arctan2(df[col_y], df[col_x])

    # this creates a new local variable df with two additional columns
    # and leaves the original df that was passed in unchanged
    # - using a dictionary that is unpacked and passed as kwargs to the
    #   assign function enables variable column names
    #   (calling `assign(col_r=...)` would create a column `col_r`
    #   instead of using the value of `col_r`)
    transformation = {
        col_r: np.sqrt(df[col_x]**2 + df[col_y]**2),
        col_phi: np.arctan2(df[col_y], df[col_x])
    }
    df = (
        df
        .assign(**transformation)
    )

    return df
