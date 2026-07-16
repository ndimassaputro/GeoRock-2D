import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def create_synthetic_boreholes(
    x,
    true_rockhead,
    borehole_x_positions,
    noise_std_m=0.25,
    random_seed=20260716,
):
    """
    Create sparse synthetic borehole observations of rockhead depth.

    Parameters
    ----------
    x : numpy.ndarray
        Horizontal coordinates of the model.

    true_rockhead : numpy.ndarray
        True synthetic rockhead profile.

    borehole_x_positions : array-like
        Horizontal borehole locations.

    noise_std_m : float
        Standard deviation of synthetic depth error.

    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    borehole_table : pandas.DataFrame
        Synthetic borehole observations.
    """

    borehole_x_positions = np.asarray(
        borehole_x_positions,
        dtype=float,
    )

    if borehole_x_positions.ndim != 1:
        raise ValueError(
            "borehole_x_positions must be one-dimensional."
        )

    if noise_std_m < 0:
        raise ValueError(
            "noise_std_m cannot be negative."
        )

    true_depths = np.interp(
        borehole_x_positions,
        x,
        true_rockhead,
    )

    rng = np.random.default_rng(
        random_seed
    )

    observed_depths = (
        true_depths
        + rng.normal(
            loc=0.0,
            scale=noise_std_m,
            size=len(
                borehole_x_positions
            ),
        )
    )

    borehole_names = [
        f"BH-{index + 1}"
        for index in range(
            len(
                borehole_x_positions
            )
        )
    ]

    borehole_table = pd.DataFrame(
        {
            "borehole_id": borehole_names,
            "x_m": borehole_x_positions,
            "true_rockhead_depth_m": true_depths,
            "observed_rockhead_depth_m": observed_depths,
            "assumed_depth_noise_std_m": (
                noise_std_m
            ),
        }
    )

    return borehole_table


def build_geophysics_baseline_profile(
    x,
    geophysical_rockhead,
    borehole_x,
    borehole_depth,
):
    """
    Construct a continuous baseline profile from partial geophysical
    estimates and sparse borehole observations.

    The function combines valid geophysical points and borehole
    constraints, then applies shape-preserving interpolation.

    Notes
    -----
    This is an interface interpretation workflow, not joint inversion.
    """

    valid_geophysical_mask = np.isfinite(
        geophysical_rockhead
    )

    combined_x = np.concatenate(
        [
            x[
                valid_geophysical_mask
            ],
            np.asarray(
                borehole_x,
                dtype=float,
            ),
        ]
    )

    combined_depth = np.concatenate(
        [
            geophysical_rockhead[
                valid_geophysical_mask
            ],
            np.asarray(
                borehole_depth,
                dtype=float,
            ),
        ]
    )

    sort_indices = np.argsort(
        combined_x
    )

    combined_x = combined_x[
        sort_indices
    ]

    combined_depth = combined_depth[
        sort_indices
    ]

    unique_x, inverse_indices = np.unique(
        combined_x,
        return_inverse=True,
    )

    averaged_depth = np.zeros_like(
        unique_x,
        dtype=float,
    )

    for unique_index in range(
        len(unique_x)
    ):
        averaged_depth[
            unique_index
        ] = np.mean(
            combined_depth[
                inverse_indices
                == unique_index
            ]
        )

    if len(unique_x) < 2:
        raise RuntimeError(
            "At least two unique constraint locations are required."
        )

    interpolator = PchipInterpolator(
        unique_x,
        averaged_depth,
        extrapolate=True,
    )

    integrated_profile = interpolator(
        x
    )

    return integrated_profile


def apply_borehole_correction(
    x,
    baseline_profile,
    borehole_x,
    borehole_depth,
    influence_length_m=22.0,
):
    """
    Correct a continuous interface profile toward borehole observations.

    Each borehole contributes a Gaussian distance-weighted correction.

    Parameters
    ----------
    x : numpy.ndarray
        Full horizontal model coordinates.

    baseline_profile : numpy.ndarray
        Continuous geophysics-plus-interpolation profile.

    borehole_x : array-like
        Borehole horizontal coordinates.

    borehole_depth : array-like
        Observed borehole rockhead depths.

    influence_length_m : float
        Horizontal correction influence length.

    Returns
    -------
    corrected_profile : numpy.ndarray
        Borehole-constrained full profile.
    """

    if influence_length_m <= 0:
        raise ValueError(
            "influence_length_m must be positive."
        )

    borehole_x = np.asarray(
        borehole_x,
        dtype=float,
    )

    borehole_depth = np.asarray(
        borehole_depth,
        dtype=float,
    )

    corrected_profile = np.asarray(
        baseline_profile,
        dtype=float,
    ).copy()

    correction_numerator = np.zeros_like(
        x,
        dtype=float,
    )

    correction_denominator = np.zeros_like(
        x,
        dtype=float,
    )

    for position, observed_depth in zip(
        borehole_x,
        borehole_depth,
    ):

        baseline_at_borehole = np.interp(
            position,
            x,
            baseline_profile,
        )

        local_residual = (
            observed_depth
            - baseline_at_borehole
        )

        weight = np.exp(
            -0.5
            * (
                (
                    x
                    - position
                )
                / influence_length_m
            )
            ** 2
        )

        correction_numerator += (
            weight
            * local_residual
        )

        correction_denominator += weight

    valid_weight = (
        correction_denominator
        > 1.0e-12
    )

    corrected_profile[
        valid_weight
    ] += (
        correction_numerator[
            valid_weight
        ]
        / correction_denominator[
            valid_weight
        ]
    )

    return corrected_profile