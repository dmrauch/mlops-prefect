import pytest

import mlops_prefect.data


@pytest.mark.parametrize(
        'n_samples, n_modes, n_dims',
        [
            (100, 3, 2),
            (1_000, 3, 2),
            (100, 5, 2),
            (100, 2, 3)
        ]
)
def test_generate(n_samples: int,
                  n_modes: int,
                  n_dims: int):

    # call the data generation function in an isolated manner
    # (not as part of a flow)
    df = mlops_prefect.data.generate.fn(
        seed=0, n_samples=n_samples, n_modes=n_modes, n_dims=n_dims)

    # test the number of samples
    assert len(df) == n_samples

    # test the cluster IDs
    assert 'cluster' in df.columns
    assert min(df.cluster.unique()) == 0
    assert max(df.cluster.unique()) == n_modes - 1

    # test the existence of the coordinate columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    if n_dims == 2:
        assert 'z' not in df.columns
    elif n_dims == 3:
        assert 'z' in df.columns
    else:
        raise ValueError("'n_dims' must be either 2 or 3")


@pytest.mark.parametrize(
        'n_dims',
        [
            (1,),
            (4,)
        ]
)
def test_generate_n_dims(n_dims: int):
    '''
    Test that specifying a wrong dimension results in a ValueError
    '''
    with pytest.raises(ValueError, match="'n_dims' must be either 2 or 3"):
        mlops_prefect.data.generate.fn(
            seed=0, n_samples=1, n_modes=1, n_dims=n_dims)
