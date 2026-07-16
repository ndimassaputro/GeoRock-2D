from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """
    Add reproducible Gaussian noise to the travel-time dataset
    generated directly from d = Gm.
    """

    project_root = Path(__file__).resolve().parents[1]

    input_file = (
        project_root
        / "data"
        / "synthetic"
        / "linear_travel_times.csv"
    )

    output_file = (
        project_root
        / "data"
        / "synthetic"
        / "linear_travel_times_noisy.csv"
    )

    random_seed = 20260716
    noise_std_ms = 1.0

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{input_file}"
        )

    dataset = pd.read_csv(input_file)

    required_columns = {
        "pair_id",
        "linear_travel_time_s",
        "linear_travel_time_ms",
    }

    missing_columns = required_columns.difference(dataset.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    rng = np.random.default_rng(random_seed)

    noise_ms = rng.normal(
        loc=0.0,
        scale=noise_std_ms,
        size=len(dataset),
    )

    dataset["noise_ms"] = noise_ms

    dataset["observed_travel_time_ms"] = (
        dataset["linear_travel_time_ms"]
        + dataset["noise_ms"]
    )

    dataset["observed_travel_time_s"] = (
        dataset["observed_travel_time_ms"]
        / 1000.0
    )

    if (dataset["observed_travel_time_s"] <= 0).any():
        raise RuntimeError(
            "Non-positive observed travel time was generated."
        )

    residual_ms = (
        dataset["observed_travel_time_ms"]
        - dataset["linear_travel_time_ms"]
    )

    rmse_ms = np.sqrt(
        np.mean(residual_ms**2)
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        output_file,
        index=False,
    )

    print("=" * 70)
    print("GeoRock-2D Linear Travel-Time Noise Model")
    print("=" * 70)

    print(f"\nNumber of observations: {len(dataset)}")
    print(f"Random seed: {random_seed}")
    print(f"Noise standard deviation: {noise_std_ms:.3f} ms")
    print(f"Realized noise RMSE: {rmse_ms:.3f} ms")

    print(
        "Observed travel-time range: "
        f"{dataset['observed_travel_time_ms'].min():.3f}–"
        f"{dataset['observed_travel_time_ms'].max():.3f} ms"
    )

    print(f"\nDataset saved to:\n{output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()