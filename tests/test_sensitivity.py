"""Tests for the GeoRock-2D path-length sensitivity matrix."""

import numpy as np

from georock2d.acquisition import (
    create_acquisition_geometry,
    create_source_receiver_pairs,
)
from georock2d.model import create_velocity_model
from georock2d.sensitivity import (
    build_sensitivity_matrix,
    calculate_linear_travel_times,
)


def create_test_inputs():
    """Create the standard synthetic model and acquisition inputs."""

    x, z, velocity, _ = create_velocity_model()

    sources, receivers = create_acquisition_geometry()

    pairs = create_source_receiver_pairs(
        sources=sources,
        receivers=receivers,
        minimum_offset=4.0,
    )

    return x, z, velocity, pairs


def test_sensitivity_matrix_shape() -> None:
    """G should contain one row per observation and one column per cell."""

    x, z, _, pairs = create_test_inputs()

    sensitivity_matrix = build_sensitivity_matrix(
        pairs=pairs,
        x=x,
        z=z,
        maximum_depth=25.0,
        number_of_points_per_segment=150,
    )

    assert len(pairs) == 196
    assert sensitivity_matrix.shape == (196, 2700)


def test_sensitivity_matrix_is_nonnegative() -> None:
    """Path lengths in G cannot be negative."""

    x, z, _, pairs = create_test_inputs()

    sensitivity_matrix = build_sensitivity_matrix(
        pairs=pairs,
        x=x,
        z=z,
        maximum_depth=25.0,
        number_of_points_per_segment=150,
    )

    assert np.all(np.isfinite(sensitivity_matrix))
    assert np.all(sensitivity_matrix >= 0.0)
    assert np.count_nonzero(sensitivity_matrix) > 0


def test_linear_forward_response_length() -> None:
    """The d = Gm result must contain one value per observation."""

    x, z, velocity, pairs = create_test_inputs()

    sensitivity_matrix = build_sensitivity_matrix(
        pairs=pairs,
        x=x,
        z=z,
        maximum_depth=25.0,
        number_of_points_per_segment=150,
    )

    travel_times = calculate_linear_travel_times(
        sensitivity_matrix=sensitivity_matrix,
        velocity=velocity,
    )

    assert travel_times.shape == (196,)
    assert np.all(np.isfinite(travel_times))
    assert np.all(travel_times > 0.0)