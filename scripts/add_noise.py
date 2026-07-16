from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """
    Add reproducible Gaussian measurement noise to synthetic travel times.

    The script reads the noiseless synthetic dataset and creates a second
    dataset containing noisy travel-time observations.

    Notes
    -----
    The selected noise level is a synthetic modelling assumption.
    It is not calibrated against a specific field survey.
    """

    # ============================================================
    # PROJECT PATHS
    # ============================================================

    project_root = Path(__file__).resolve().parents[1]

    input_file = (
        project_root
        / "data"
        / "synthetic"
        / "travel_times.csv"
    )

    output_file = (
        project_root
        / "data"
        / "synthetic"
        / "travel_times_noisy.csv"
    )

    # ============================================================
    # SETTINGS
    # ============================================================

    random_seed = 20260716

    # Standard deviation of travel-time noise in milliseconds
    noise_std_ms = 1.0

    # ============================================================
    # LOAD DATASET
    # ============================================================

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input dataset was not found:\n{input_file}"
        )

    dataset = pd.read_csv(input_file)

    required_columns = {
        "pair_id",
        "source_x_m",
        "receiver_x_m",
        "offset_m",
        "travel_time_s",
        "travel_time_ms",
    }

    missing_columns = required_columns.difference(dataset.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # ============================================================
    # GENERATE NOISE
    # ============================================================

    rng = np.random.default_rng(random_seed)

    noise_ms = rng.normal(
        loc=0.0,
        scale=noise_std_ms,
        size=len(dataset),
    )

    dataset["noise_ms"] = noise_ms

    dataset["observed_travel_time_ms"] = (
        dataset["travel_time_ms"]
        + dataset["noise_ms"]
    )

    dataset["observed_travel_time_s"] = (
        dataset["observed_travel_time_ms"]
        / 1000.0
    )

    # Prevent physically invalid negative travel times
    if (dataset["observed_travel_time_s"] <= 0).any():
        raise RuntimeError(
            "Noise generation produced non-positive travel times. "
            "Reduce the selected noise standard deviation."
        )

    # ============================================================
    # ERROR METRICS
    # ============================================================

    residual_ms = (
        dataset["observed_travel_time_ms"]
        - dataset["travel_time_ms"]
    )

    rmse_ms = np.sqrt(
        np.mean(residual_ms**2)
    )

    mean_error_ms = residual_ms.mean()

    maximum_absolute_error_ms = np.abs(residual_ms).max()

    # ============================================================
    # SAVE DATASET
    # ============================================================

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        output_file,
        index=False,
    )

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print("=" * 70)
    print("GeoRock-2D Measurement Noise Generation")
    print("=" * 70)

    print(f"\nInput observations: {len(dataset)}")
    print(f"Random seed: {random_seed}")
    print(f"Selected noise standard deviation: {noise_std_ms:.3f} ms")

    print("\nNoise statistics:")
    print(dataset["noise_ms"].describe())

    print(f"\nMean noise: {mean_error_ms:.4f} ms")
    print(f"Noise RMSE: {rmse_ms:.4f} ms")
    print(
        "Maximum absolute noise: "
        f"{maximum_absolute_error_ms:.4f} ms"
    )

    print(
        "\nObserved travel-time range: "
        f"{dataset['observed_travel_time_ms'].min():.3f}–"
        f"{dataset['observed_travel_time_ms'].max():.3f} ms"
    )

    print(f"\nNoisy dataset saved successfully to:\n{output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()