import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear


def invert_active_slowness(
    sensitivity_matrix,
    observed_travel_times_ms,
    smoothness_matrix,
    active_flat_indices,
    regularization_lambda=10.0,
    reference_velocity_m_s=1500.0,
    damping_ratio=0.1,
    minimum_velocity_m_s=400.0,
    maximum_velocity_m_s=4000.0,
):
    """
    Perform bounded regularized least-squares inversion.

    Parameters
    ----------
    sensitivity_matrix : numpy.ndarray
        Full sensitivity matrix with shape
        observations × full model cells.

    observed_travel_times_ms : numpy.ndarray
        Observed travel-time vector in milliseconds.

    smoothness_matrix : scipy.sparse.csr_matrix
        First-order spatial smoothness matrix.

    active_flat_indices : numpy.ndarray
        Flat indices of cells crossed by rays.

    regularization_lambda : float
        Strength of spatial smoothness regularization.

    reference_velocity_m_s : float
        Homogeneous reference-model velocity.

    damping_ratio : float
        Relative strength of damping toward the reference model.

    minimum_velocity_m_s : float
        Minimum allowed velocity.

    maximum_velocity_m_s : float
        Maximum allowed velocity.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Result returned by scipy.optimize.lsq_linear.

    estimated_slowness_ms_m : numpy.ndarray
        Estimated active-cell slowness in milliseconds per metre.

    predicted_travel_times_ms : numpy.ndarray
        Predicted travel times from the estimated model.
    """

    if regularization_lambda <= 0:
        raise ValueError(
            "regularization_lambda must be positive."
        )

    if reference_velocity_m_s <= 0:
        raise ValueError(
            "reference_velocity_m_s must be positive."
        )

    observed = np.asarray(
        observed_travel_times_ms,
        dtype=float,
    )

    if observed.ndim != 1:
        raise ValueError(
            "Observed travel times must be one-dimensional."
        )

    active_sensitivity = sensitivity_matrix[
        :,
        active_flat_indices,
    ]

    active_sensitivity = sparse.csr_matrix(
        active_sensitivity
    )

    number_of_active_cells = len(
        active_flat_indices
    )

    # Slowness is represented in ms/m:
    # slowness_ms_m = 1000 / velocity_m_s
    reference_slowness = (
        1000.0
        / reference_velocity_m_s
    )

    reference_model = np.full(
        number_of_active_cells,
        reference_slowness,
        dtype=float,
    )

    damping_lambda = (
        regularization_lambda
        * damping_ratio
    )

    identity_matrix = sparse.identity(
        number_of_active_cells,
        format="csr",
        dtype=float,
    )

    augmented_matrix = sparse.vstack(
        [
            active_sensitivity,
            regularization_lambda
            * smoothness_matrix,
            damping_lambda
            * identity_matrix,
        ],
        format="csr",
    )

    augmented_data = np.concatenate(
        [
            observed,
            np.zeros(
                smoothness_matrix.shape[0],
                dtype=float,
            ),
            damping_lambda
            * reference_model,
        ]
    )

    minimum_slowness = (
        1000.0
        / maximum_velocity_m_s
    )

    maximum_slowness = (
        1000.0
        / minimum_velocity_m_s
    )

    result = lsq_linear(
        augmented_matrix,
        augmented_data,
        bounds=(
            minimum_slowness,
            maximum_slowness,
        ),
        method="trf",
        lsmr_tol="auto",
        max_iter=300,
        verbose=0,
    )

    estimated_slowness_ms_m = result.x

    predicted_travel_times_ms = (
        active_sensitivity
        @ estimated_slowness_ms_m
    )

    return (
        result,
        estimated_slowness_ms_m,
        np.asarray(
            predicted_travel_times_ms
        ).ravel(),
    )


def active_model_to_full_velocity(
    estimated_slowness_ms_m,
    active_flat_indices,
    full_model_shape,
    inactive_velocity_m_s=np.nan,
):
    """
    Insert estimated active-cell values into a full 2D velocity model.

    Inactive cells are assigned NaN by default because they are not
    constrained by travel-time observations.
    """

    number_of_full_cells = int(
        np.prod(full_model_shape)
    )

    full_velocity_vector = np.full(
        number_of_full_cells,
        inactive_velocity_m_s,
        dtype=float,
    )

    active_velocity = (
        1000.0
        / estimated_slowness_ms_m
    )

    full_velocity_vector[
        active_flat_indices
    ] = active_velocity

    return full_velocity_vector.reshape(
        full_model_shape
    )