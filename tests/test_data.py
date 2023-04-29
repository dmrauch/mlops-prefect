import pytest

import mlops_prefect.data


@pytest.mark.parametrize(
        'n_samples, n_modes',
        [
            (100, 3),
            (1_000, 3),
            (100, 5)
        ]
)
def test_generate(n_samples: int,
                  n_modes: int):

    # call the data generation function in an isolated manner
    # (not as part of a flow)
    df = mlops_prefect.data.generate.fn(
        seed=0, n_samples=n_samples, n_modes=n_modes)

    # test the number of samples
    assert len(df) == n_samples

    # test the cluster IDs
    assert 'cluster' in df.columns
    assert min(df.cluster.unique()) == 0
    assert max(df.cluster.unique()) == n_modes - 1

    # test the existence of the coordinate columns
    assert 'x' in df.columns
    assert 'y' in df.columns
