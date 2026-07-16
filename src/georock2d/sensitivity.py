import numpy as np

from georock2d.forward import sample_v_shaped_ray


def calculate_cell_indices(
    x_coordinates,
    z_coordinates,
    x,
    z,
):
    """
    Convert physical coordinates into model-cell indices.

    Parameters
    ----------
    x_coordinates : numpy.ndarray
        Horizontal coordinates in metres.

    z_coordinates : numpy.ndarray
        Depth coordinates in metres.

    x : numpy.ndarray
        Horizontal cell-centre coordinates.

    z : numpy.ndarray
        Vertical cell-centre coordinates.

    Returns
    -------
    x_indices : numpy.ndarray
        Horizontal cell indices.

    z_indices : numpy.ndarray
        Vertical cell indices.
    """

    dx = x[1] - x[0]
    dz = z[1] - z[0]

    x_indices = np.floor(
        x_coordinates / dx
    ).astype(int)

    z_indices = np.floor(
        z_coordinates / dz
    ).astype(int)

    x_indices = np.clip(
        x_indices,
        0,
        len(x) - 1,
    )

    z_indices = np.clip(
        z_indices,
        0,
        len(z) - 1,
    )

    return x_indices, z_indices


def calculate_ray_cell_lengths(
    source,
    receiver,
    x,
    z,
    maximum_depth=25.0,
    number_of_points_per_segment=150,
):
    """
    Calculate the path length travelled through every model cell.

    Each segment of the conceptual V-shaped ray is assigned to the
    cell containing its midpoint.

    Parameters
    ----------
    source : numpy.ndarray
        Source coordinate [x, z].

    receiver : numpy.ndarray
        Receiver coordinate [x, z].

    x : numpy.ndarray
        Horizontal cell-centre coordinates.

    z : numpy.ndarray
        Vertical cell-centre coordinates.

    maximum_depth : float
        Maximum conceptual turning depth in metres.

    number_of_points_per_segment : int
        Number of samples along each part of the V-shaped ray.

    Returns
    -------
    cell_lengths : numpy.ndarray
        Two-dimensional array containing ray length in each cell.
    """

    ray_x, ray_z = sample_v_shaped_ray(
        source=source,
        receiver=receiver,
        maximum_depth=maximum_depth,
        number_of_points_per_segment=number_of_points_per_segment,
    )

    # Length of every short ray segment
    segment_lengths = np.sqrt(
        np.diff(ray_x) ** 2
        + np.diff(ray_z) ** 2
    )

    # Segment midpoints are used for assigning segments to cells
    midpoint_x = (
        ray_x[:-1]
        + ray_x[1:]
    ) / 2.0

    midpoint_z = (
        ray_z[:-1]
        + ray_z[1:]
    ) / 2.0

    x_indices, z_indices = calculate_cell_indices(
        x_coordinates=midpoint_x,
        z_coordinates=midpoint_z,
        x=x,
        z=z,
    )

    cell_lengths = np.zeros(
        shape=(len(z), len(x)),
        dtype=float,
    )

    # Add each segment length to the appropriate model cell
    np.add.at(
        cell_lengths,
        (z_indices, x_indices),
        segment_lengths,
    )

    return cell_lengths


def build_sensitivity_matrix(
    pairs,
    x,
    z,
    maximum_depth=25.0,
    number_of_points_per_segment=150,
):
    """
    Build the travel-time sensitivity matrix G.

    Each row represents one source-receiver observation.
    Each column represents one model cell.

    Matrix element G[i, j] is the ray-path length travelled by
    observation i through cell j.

    Parameters
    ----------
    pairs : list
        Source-receiver coordinate pairs.

    x : numpy.ndarray
        Horizontal cell-centre coordinates.

    z : numpy.ndarray
        Vertical cell-centre coordinates.

    maximum_depth : float
        Maximum conceptual turning depth.

    number_of_points_per_segment : int
        Number of points used to discretize each ray segment.

    Returns
    -------
    sensitivity_matrix : numpy.ndarray
        Matrix with shape:
        number of observations × number of model cells.
    """

    number_of_observations = len(pairs)
    number_of_cells = len(x) * len(z)

    sensitivity_matrix = np.zeros(
        shape=(
            number_of_observations,
            number_of_cells,
        ),
        dtype=float,
    )

    for observation_id, (source, receiver) in enumerate(pairs):

        cell_lengths = calculate_ray_cell_lengths(
            source=source,
            receiver=receiver,
            x=x,
            z=z,
            maximum_depth=maximum_depth,
            number_of_points_per_segment=number_of_points_per_segment,
        )

        sensitivity_matrix[
            observation_id,
            :
        ] = cell_lengths.ravel()

    return sensitivity_matrix


def calculate_linear_travel_times(
    sensitivity_matrix,
    velocity,
):
    """
    Calculate travel times using d = Gm.

    Parameters
    ----------
    sensitivity_matrix : numpy.ndarray
        Path-length sensitivity matrix G.

    velocity : numpy.ndarray
        Two-dimensional velocity model in m/s.

    Returns
    -------
    travel_times_s : numpy.ndarray
        Travel-time vector in seconds.
    """

    if np.any(velocity <= 0):
        raise ValueError(
            "Velocity values must be greater than zero."
        )

    slowness = 1.0 / velocity

    model_vector = slowness.ravel()

    travel_times_s = (
        sensitivity_matrix
        @ model_vector
    )

    return travel_times_s