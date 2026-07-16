from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

# scripts/generate_dataset.py
# parents[0] = scripts
# parents[1] = GeoRock-2D project root
project_root = Path(__file__).resolve().parents[1]

# Add the src folder so Python can import the georock2d package
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from georock2d.model import create_velocity_model
from georock2d.acquisition import (
    create_acquisition_geometry,
    create_source_receiver_pairs,
)
from georock2d.forward import calculate_v_ray_travel_time


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """
    Generate a synthetic seismic travel-time dataset.

    The workflow:
    1. Creates the synthetic velocity model.
    2. Creates sources and receivers.
    3. Forms valid source-receiver pairs.
    4. Calculates conceptual V-shaped ray travel times.
    5. Stores the results in a CSV file.

    Notes
    -----
    The V-shaped ray geometry is a conceptual baseline.
    It does not reproduce full seismic refraction physics.
    """

    # --------------------------------------------------------
    # 1. Create the true synthetic velocity model
    # --------------------------------------------------------

    x, z, velocity, rockhead_depth = create_velocity_model()

    # --------------------------------------------------------
    # 2. Create acquisition geometry
    # --------------------------------------------------------

    sources, receivers = create_acquisition_geometry()

    # --------------------------------------------------------
    # 3. Create valid source-receiver pairs
    # --------------------------------------------------------

    pairs = create_source_receiver_pairs(
        sources=sources,
        receivers=receivers,
        minimum_offset=4.0,
    )

    # --------------------------------------------------------
    # 4. Calculate travel time for every pair
    # --------------------------------------------------------

    records = []

    for pair_id, (source, receiver) in enumerate(pairs, start=1):

        travel_time = calculate_v_ray_travel_time(
            source=source,
            receiver=receiver,
            x=x,
            z=z,
            velocity=velocity,
        )

        source_x = float(source[0])
        source_z = float(source[1])

        receiver_x = float(receiver[0])
        receiver_z = float(receiver[1])

        offset = abs(receiver_x - source_x)

        records.append(
            {
                "pair_id": pair_id,
                "source_x_m": source_x,
                "source_z_m": source_z,
                "receiver_x_m": receiver_x,
                "receiver_z_m": receiver_z,
                "offset_m": offset,
                "travel_time_s": float(travel_time),
                "travel_time_ms": float(travel_time * 1000.0),
            }
        )

    # --------------------------------------------------------
    # 5. Convert records into a pandas DataFrame
    # --------------------------------------------------------

    dataset = pd.DataFrame(records)

    if dataset.empty:
        raise RuntimeError(
            "The generated dataset is empty. "
            "Check the acquisition geometry and minimum offset."
        )

    # --------------------------------------------------------
    # 6. Check the generated values
    # --------------------------------------------------------

    if dataset["travel_time_s"].isna().any():
        raise RuntimeError(
            "NaN values were found in the calculated travel times."
        )

    if (dataset["travel_time_s"] <= 0).any():
        raise RuntimeError(
            "All calculated travel times must be positive."
        )

    # --------------------------------------------------------
    # 7. Print a summary
    # --------------------------------------------------------

    print("=" * 70)
    print("GeoRock-2D Synthetic Travel-Time Dataset")
    print("=" * 70)

    print(f"\nNumber of sources: {len(sources)}")
    print(f"Number of receivers: {len(receivers)}")
    print(f"Number of valid source-receiver pairs: {len(pairs)}")

    print("\nFirst five observations:")
    print(dataset.head().to_string(index=False))

    print("\nDataset statistics:")
    print(
        dataset[
            [
                "offset_m",
                "travel_time_s",
                "travel_time_ms",
            ]
        ].describe()
    )

    print(
        "\nTravel-time range: "
        f"{dataset['travel_time_s'].min():.6f}–"
        f"{dataset['travel_time_s'].max():.6f} s"
    )

    print(
        "Travel-time range: "
        f"{dataset['travel_time_ms'].min():.3f}–"
        f"{dataset['travel_time_ms'].max():.3f} ms"
    )

    # --------------------------------------------------------
    # 8. Save the dataset
    # --------------------------------------------------------

    output_folder = project_root / "data" / "synthetic"

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_folder / "travel_times.csv"

    dataset.to_csv(
        output_file,
        index=False,
    )

    print(f"\nDataset saved successfully to:\n{output_file}")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()