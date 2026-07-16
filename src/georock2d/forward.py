import numpy as np


def sample_straight_ray(
    source,
    receiver,
    number_of_points=300,
):
    """
    Sample points along a straight source-receiver ray.

    Parameters
    ----------
    source : numpy.ndarray
        Source coordinate [x, z].

    receiver : numpy.ndarray
        Receiver coordinate [x, z].

    number_of_points : int
        Number of sample points along the ray.

    Returns
    -------
    ray_x : numpy.ndarray
        Horizontal coordinates along the ray.

    ray_z : numpy.ndarray
        Depth coordinates along the ray.
    """

    ray_x = np.linspace(
        source[0],
        receiver[0],
        number_of_points,
    )

    ray_z = np.linspace(
        source[1],
        receiver[1],
        number_of_points,
    )

    return ray_x, ray_z


def coordinates_to_indices(
    ray_x,
    ray_z,
    x,
    z,
):
    """
    Convert ray coordinates to nearest model-cell indices.
    """

    dx = x[1] - x[0]
    dz = z[1] - z[0]

    x_indices = np.floor(
        ray_x / dx
    ).astype(int)

    z_indices = np.floor(
        ray_z / dz
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


def calculate_straight_ray_travel_time(
    source,
    receiver,
    x,
    z,
    velocity,
    number_of_points=300,
):
    """
    Calculate approximate travel time along a straight ray.

    Notes
    -----
    This is a simplified baseline model. It does not simulate
    refracted first arrivals or curved ray paths.
    """

    ray_x, ray_z = sample_straight_ray(
        source,
        receiver,
        number_of_points,
    )

    x_indices, z_indices = coordinates_to_indices(
        ray_x,
        ray_z,
        x,
        z,
    )

    sampled_velocity = velocity[
        z_indices,
        x_indices,
    ]

    segment_lengths = np.sqrt(
        np.diff(ray_x) ** 2
        + np.diff(ray_z) ** 2
    )

    segment_velocity = (
        sampled_velocity[:-1]
        + sampled_velocity[1:]
    ) / 2.0

    travel_time = np.sum(
        segment_lengths
        / segment_velocity
    )

    return travel_time
def sample_v_shaped_ray(
    source,
    receiver,
    maximum_depth=25.0,
    number_of_points_per_segment=150,
):
    """
    Create a simple V-shaped subsurface ray path.

    The turning depth increases with source-receiver offset.
    This is a conceptual baseline, not a physical ray tracer.
    """

    midpoint_x = (
        source[0] + receiver[0]
    ) / 2.0

    offset = abs(
        receiver[0] - source[0]
    )

    turning_depth = min(
        0.35 * offset,
        maximum_depth,
    )

    midpoint = np.array(
        [
            midpoint_x,
            turning_depth,
        ]
    )

    first_x = np.linspace(
        source[0],
        midpoint[0],
        number_of_points_per_segment,
    )

    first_z = np.linspace(
        source[1],
        midpoint[1],
        number_of_points_per_segment,
    )

    second_x = np.linspace(
        midpoint[0],
        receiver[0],
        number_of_points_per_segment,
    )

    second_z = np.linspace(
        midpoint[1],
        receiver[1],
        number_of_points_per_segment,
    )

    ray_x = np.concatenate(
        (
            first_x,
            second_x[1:],
        )
    )

    ray_z = np.concatenate(
        (
            first_z,
            second_z[1:],
        )
    )

    return ray_x, ray_z


def calculate_v_ray_travel_time(
    source,
    receiver,
    x,
    z,
    velocity,
):
    """
    Calculate approximate travel time along a V-shaped path.
    """

    ray_x, ray_z = sample_v_shaped_ray(
        source,
        receiver,
    )

    x_indices, z_indices = coordinates_to_indices(
        ray_x,
        ray_z,
        x,
        z,
    )

    sampled_velocity = velocity[
        z_indices,
        x_indices,
    ]

    segment_lengths = np.sqrt(
        np.diff(ray_x) ** 2
        + np.diff(ray_z) ** 2
    )

    segment_velocity = (
        sampled_velocity[:-1]
        + sampled_velocity[1:]
    ) / 2.0

    travel_time = np.sum(
        segment_lengths
        / segment_velocity
    )

    return travel_time