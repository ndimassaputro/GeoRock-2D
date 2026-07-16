"""Tests for GeoRock-2D forward travel-time calculations."""

import numpy as np

from georock2d.forward import (
    calculate_v_ray_travel_time,
    sample_v_shaped_ray,
)
from georock2d.model import create_velocity_model


def test_v_shaped_ray_starts_and_ends_correctly() -> None:
    """The sampled ray must connect the specified source and receiver."""

    source = np.array([5.0, 0.0])
    receiver = np.array([85.0, 0.0])

    ray_x, ray_z = sample_v_shaped_ray(
        source=source,
        receiver=receiver,
    )

    assert np.isclose(ray_x[0], source[0])
    assert np.isclose(ray_z[0], source[1])

    assert np.isclose(ray_x[-1], receiver[0])
    assert np.isclose(ray_z[-1], receiver[1])

    assert np.max(ray_z) > 0.0


def test_calculated_travel_time_is_positive() -> None:
    """A valid source–receiver pair must produce positive travel time."""

    x, z, velocity, _ = create_velocity_model()

    source = np.array([5.0, 0.0])
    receiver = np.array([85.0, 0.0])

    travel_time = calculate_v_ray_travel_time(
        source=source,
        receiver=receiver,
        x=x,
        z=z,
        velocity=velocity,
    )

    assert np.isfinite(travel_time)
    assert travel_time > 0.0


def test_longer_offset_produces_longer_path() -> None:
    """The conceptual ray length should increase with offset."""

    source = np.array([5.0, 0.0])

    near_receiver = np.array([25.0, 0.0])
    far_receiver = np.array([105.0, 0.0])

    near_x, near_z = sample_v_shaped_ray(
        source=source,
        receiver=near_receiver,
    )

    far_x, far_z = sample_v_shaped_ray(
        source=source,
        receiver=far_receiver,
    )

    near_length = np.sum(
        np.sqrt(
            np.diff(near_x) ** 2
            + np.diff(near_z) ** 2
        )
    )

    far_length = np.sum(
        np.sqrt(
            np.diff(far_x) ** 2
            + np.diff(far_z) ** 2
        )
    )

    assert far_length > near_length