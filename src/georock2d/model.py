import numpy as np


def create_velocity_model():
    """
    Create a synthetic 2D seismic velocity model.

    Returns
    -------
    x : numpy.ndarray
        Horizontal coordinates in metres.
    z : numpy.ndarray
        Depth coordinates in metres.
    velocity : numpy.ndarray
        Two-dimensional P-wave velocity model in m/s.
    rockhead_depth : numpy.ndarray
        Rockhead depth along the horizontal profile.
    """

    # Model dimensions
    length = 120.0
    depth = 45.0

    # Grid spacing
    dx = 2.0
    dz = 1.0

    # Cell-centre coordinates
    x = np.arange(dx / 2, length, dx)
    z = np.arange(dz / 2, depth, dz)

    # Create the 2D coordinate grid
    x_grid, z_grid = np.meshgrid(x, z)

    # Synthetic undulating rockhead
    rockhead_depth = (
        18.0
        + 4.0 * np.sin(2.0 * np.pi * x / 70.0)
        + 5.0 * np.exp(-0.5 * ((x - 82.0) / 13.0) ** 2)
    )

    # Initial velocity model filled with soil velocity
    velocity = np.full_like(x_grid, 650.0, dtype=float)

    # Add weathered-rock transition zone
    transition_top = rockhead_depth - 1.0
    transition_bottom = rockhead_depth + 1.0

    velocity[
        (z_grid >= transition_top[np.newaxis, :])
        & (z_grid < transition_bottom[np.newaxis, :])
    ] = 1200.0

    # Add competent rock
    velocity[
        z_grid >= transition_bottom[np.newaxis, :]
    ] = 2800.0

    return x, z, velocity, rockhead_depth