from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """
    Plot noiseless and noisy synthetic travel-time observations.
    """

    # ============================================================
    # PROJECT PATHS
    # ============================================================

    project_root = Path(__file__).resolve().parents[1]

    input_file = (
        project_root
        / "data"
        / "synthetic"
        / "travel_times_noisy.csv"
    )

    output_folder = (
        project_root
        / "results"
        / "figures"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_folder
        / "03_synthetic_travel_times.png"
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    if not input_file.exists():
        raise FileNotFoundError(
            f"Noisy travel-time dataset was not found:\n{input_file}"
        )

    dataset = pd.read_csv(input_file)

    # ============================================================
    # CREATE FIGURE
    # ============================================================

    figure, axis = plt.subplots(
        figsize=(9, 5.5),
    )

    axis.scatter(
        dataset["offset_m"],
        dataset["travel_time_ms"],
        s=28,
        alpha=0.7,
        label="Noiseless synthetic data",
    )

    axis.scatter(
        dataset["offset_m"],
        dataset["observed_travel_time_ms"],
        s=18,
        alpha=0.6,
        label="Noisy observations",
    )

    axis.set_xlabel(
        "Source–receiver offset (m)"
    )

    axis.set_ylabel(
        "Travel time (ms)"
    )

    axis.set_title(
        "GeoRock-2D Synthetic Travel-Time Observations"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print("=" * 70)
    print("GeoRock-2D Travel-Time Visualization")
    print("=" * 70)

    print(f"\nNumber of observations: {len(dataset)}")
    print(f"Figure saved successfully to:\n{output_file}")

    print(
        "\nNoiseless travel-time range: "
        f"{dataset['travel_time_ms'].min():.3f}–"
        f"{dataset['travel_time_ms'].max():.3f} ms"
    )

    print(
        "Noisy travel-time range: "
        f"{dataset['observed_travel_time_ms'].min():.3f}–"
        f"{dataset['observed_travel_time_ms'].max():.3f} ms"
    )

    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()