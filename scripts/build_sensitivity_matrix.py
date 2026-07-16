from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

project_root = Path(__file__).resolve().parents[1]

src_path = project_root / "src"

sys.path.insert(
    0,
    str(src_path),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from georock2d.model import create_velocity_model

from georock2d.acquisition import (
    create_acquisition_geometry,
    create_source_receiver_pairs,
)

from georock2d.sensitivity import (
    build_sensitivity_matrix,
    calculate_linear_travel_times,
)


def main():
    """
    Build and evaluate the GeoRock-2D sensitivity matrix.
    """

    # ========================================================
    # 1. TRUE MODEL
    # ========================================================

    x, z, velocity, rockhead_depth = create_velocity_model()

    number_of_horizontal_cells = len(x)
    number_of_vertical_cells = len(z)
    number_of_model_cells = velocity.size

    # ========================================================
    # 2. ACQUISITION GEOMETRY
    # ========================================================

    sources, receivers = create_acquisition_geometry()

    pairs = create_source_receiver_pairs(
        sources=sources,
        receivers=receivers,
        minimum_offset=4.0,
    )

    # ========================================================
    # 3. BUILD SENSITIVITY MATRIX
    # ========================================================

    print("=" * 70)
    print("GeoRock-2D Sensitivity Matrix Construction")
    print("=" * 70)

    print("\nBuilding sensitivity matrix...")

    sensitivity_matrix = build_sensitivity_matrix(
        pairs=pairs,
        x=x,
        z=z,
        maximum_depth=25.0,
        number_of_points_per_segment=150,
    )

    # ========================================================
    # 4. CALCULATE LINEAR FORWARD RESPONSE
    # ========================================================

    travel_times_s = calculate_linear_travel_times(
        sensitivity_matrix=sensitivity_matrix,
        velocity=velocity,
    )

    travel_times_ms = travel_times_s * 1000.0

    # ========================================================
    # 5. RAY-DENSITY MODEL
    # ========================================================

    ray_density_vector = np.sum(
        sensitivity_matrix,
        axis=0,
    )

    ray_density = ray_density_vector.reshape(
        velocity.shape
    )

    cells_with_coverage = np.count_nonzero(
        ray_density > 0
    )

    coverage_percentage = (
        cells_with_coverage
        / number_of_model_cells
        * 100.0
    )

    # ========================================================
    # 6. SAVE NUMERICAL OUTPUTS
    # ========================================================

    model_output_folder = (
        project_root
        / "results"
        / "models"
    )

    model_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_output_file = (
        model_output_folder
        / "sensitivity_matrix.npz"
    )

    np.savez_compressed(
        matrix_output_file,
        sensitivity_matrix=sensitivity_matrix,
        ray_density=ray_density,
        x=x,
        z=z,
    )

    # ========================================================
    # 7. SAVE INVERSION-CONSISTENT DATASET
    # ========================================================

    records = []

    for pair_id, (
        (source, receiver),
        travel_time_s,
        travel_time_ms,
    ) in enumerate(
        zip(
            pairs,
            travel_times_s,
            travel_times_ms,
        ),
        start=1,
    ):

        records.append(
            {
                "pair_id": pair_id,
                "source_x_m": float(source[0]),
                "source_z_m": float(source[1]),
                "receiver_x_m": float(receiver[0]),
                "receiver_z_m": float(receiver[1]),
                "offset_m": float(
                    abs(receiver[0] - source[0])
                ),
                "linear_travel_time_s": float(
                    travel_time_s
                ),
                "linear_travel_time_ms": float(
                    travel_time_ms
                ),
            }
        )

    dataset = pd.DataFrame(records)

    dataset_output_folder = (
        project_root
        / "data"
        / "synthetic"
    )

    dataset_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_output_file = (
        dataset_output_folder
        / "linear_travel_times.csv"
    )

    dataset.to_csv(
        dataset_output_file,
        index=False,
    )

    # ========================================================
    # 8. PLOT RAY DENSITY
    # ========================================================

    dx = x[1] - x[0]
    dz = z[1] - z[0]

    x_edges = np.arange(
        0.0,
        x[-1] + dx,
        dx,
    )

    z_edges = np.arange(
        0.0,
        z[-1] + dz,
        dz,
    )

    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    image = axis.pcolormesh(
        x_edges,
        z_edges,
        ray_density,
        shading="auto",
        cmap="viridis",
    )

    axis.plot(
        x,
        rockhead_depth,
        linewidth=2.0,
        label="True rockhead",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Depth below ground surface (m)"
    )

    axis.set_title(
        "GeoRock-2D Ray-Path Density"
    )

    axis.set_xlim(
        0.0,
        120.0,
    )

    axis.set_ylim(
        45.0,
        0.0,
    )

    axis.legend()

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Cumulative ray-path length (m)"
    )

    figure.tight_layout()

    figure_output_folder = (
        project_root
        / "results"
        / "figures"
    )

    figure_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_output_file = (
        figure_output_folder
        / "04_ray_density.png"
    )

    figure.savefig(
        figure_output_file,
        dpi=300,
        bbox_inches="tight",
    )

    # ========================================================
    # 9. PRINT RESULTS
    # ========================================================

    print("\nMatrix dimensions:")
    print(
        f"Observations: {sensitivity_matrix.shape[0]}"
    )
    print(
        f"Model cells: {sensitivity_matrix.shape[1]}"
    )
    print(
        "Sensitivity matrix shape: "
        f"{sensitivity_matrix.shape}"
    )

    print("\nModel grid:")
    print(
        f"Horizontal cells: {number_of_horizontal_cells}"
    )
    print(
        f"Vertical cells: {number_of_vertical_cells}"
    )
    print(
        f"Total model cells: {number_of_model_cells}"
    )

    print("\nRay coverage:")
    print(
        f"Cells crossed by at least one ray: "
        f"{cells_with_coverage}"
    )
    print(
        f"Model-cell coverage: "
        f"{coverage_percentage:.2f}%"
    )

    print(
        "\nLinear travel-time range: "
        f"{travel_times_ms.min():.3f}–"
        f"{travel_times_ms.max():.3f} ms"
    )

    print(
        f"\nSensitivity matrix saved to:\n"
        f"{matrix_output_file}"
    )

    print(
        f"\nLinear dataset saved to:\n"
        f"{dataset_output_file}"
    )

    print(
        f"\nRay-density figure saved to:\n"
        f"{figure_output_file}"
    )

    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()