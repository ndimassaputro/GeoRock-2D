"""Tests for the synthetic GeoRock-2D velocity model."""

import numpy as np

from georock2d.model import create_velocity_model


def test_velocity_model_shape() -> None:
    """The synthetic model should have the expected dimensions."""

    x, z, velocity, rockhead_depth = create_velocity_model()

    assert velocity.shape == (45, 60)
    assert len(x) == 60
    assert len(z) == 45
    assert rockhead_depth.shape == (60,)


def test_velocity_values_are_positive() -> None:
    """All P-wave velocity values must be physically positive."""

    _, _, velocity, _ = create_velocity_model()

    assert np.all(np.isfinite(velocity))
    assert np.all(velocity > 0.0)


def test_rockhead_remains_inside_model_domain() -> None:
    """The synthetic rockhead must lie inside the model depth."""

    _, z, _, rockhead_depth = create_velocity_model()

    model_bottom = z[-1] + 0.5

    assert np.all(np.isfinite(rockhead_depth))
    assert np.all(rockhead_depth > 0.0)
    assert np.all(rockhead_depth < model_bottom)