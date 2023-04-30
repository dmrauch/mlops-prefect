import numpy as np
import pandas as pd
import prefect
from typing import Union


def get_r(x: Union[float, np.ndarray],
          y: Union[float, np.ndarray],
          z: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray]:
    '''
    Vectorised calculation of the radius
    '''
    return np.sqrt(x**2 + y**2 + z**2)

def get_theta(z: Union[float, np.ndarray],
              r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Vectorised calculation of the polar angle

    Returns
    -------
    Polar angle theta, counted downwards from the positive z-axis.
    '''
    return np.arccos(z / r)

def get_phi(x: Union[float, np.ndarray],
            y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Vectorised calculation of the azimuthal angle

    Returns
    -------
    Azimuthal angle phi, counted counterclockwise within the xy-plane,
    starting from the positive x-axis.
    '''
    return np.arctan2(y, x)


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
    # - using lambda functions makes it possible to use newly created
    #   columns for defining other columns,
    #   recommended reading on chaining assigns:
    #   - https://ponder.io/professional-pandas-the-pandas-assign-method-and-chaining/
    #   - https://practicaldatascience.co.uk/data-science/how-to-use-method-chaining-in-pandas
    transformation = {
        col_r: lambda dfx: get_r(x=dfx[col_x], y=dfx[col_y]),
        col_phi: lambda dfx: get_phi(x=dfx[col_x], y=dfx[col_y])
    }
    df = (
        df
        .assign(**transformation)
    )

    return df


@prefect.task
def cartesian_to_spherical(df: pd.DataFrame,
                           col_x: str = 'x',
                           col_y: str = 'y',
                           col_z: str = 'z',
                           col_r: str = 'r',
                           col_theta: str = 'theta',
                           col_phi: str = 'phi') -> pd.DataFrame:
    '''
    Convert 3D cartesian to spherical coordinates

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
    col_z
        The name of the input column containing the z-values
    col_r
        The name of the output column containing the radii
    col_theta
        The name of the output column containing the polar angles
        (counted downwards from the positive z-axis)
    col_phi
        The name of the output column containing the azimuthal angles
        (counted counterclockwise within the xy-plane, starting from the
        positive x-axis)

    Returns
    -------
    The dataframe with three additional columns containing the radii
    as well as polar and azimuthal angles
    '''
    # this would change the original df that was passed in
    # - you can try this out and should see the unit tests fail
    # df[col_r] = np.sqrt(df[col_x]**2 + df[col_y]**2 + df[col_z]**2)
    # df[col_theta] = np.arccos(df[col_z] / df[col_r])
    # df[col_phi] = np.arctan2(df[col_y], df[col_x])

    # this creates a new local variable df with three additional columns
    # and leaves the original df that was passed in unchanged
    # - using a dictionary that is unpacked and passed as kwargs to the
    #   assign function enables variable column names
    #   (calling `assign(col_r=...)` would create a column `col_r`
    #   instead of using the value of `col_r`)
    # - using lambda functions makes it possible to use newly created
    #   columns for defining other columns,
    #   recommended reading on chaining assigns:
    #   - https://ponder.io/professional-pandas-the-pandas-assign-method-and-chaining/
    #   - https://practicaldatascience.co.uk/data-science/how-to-use-method-chaining-in-pandas
    transformation = {
        col_r: lambda dfx: get_r(x=dfx[col_x], y=dfx[col_y], z=dfx[col_z]),
        col_theta: lambda dfx: get_theta(z=dfx[col_z], r=dfx[col_r]),
        col_phi: lambda dfx: get_phi(x=dfx[col_x], y=dfx[col_y])
    }
    df = (
        df
        .assign(**transformation)
    )

    return df
