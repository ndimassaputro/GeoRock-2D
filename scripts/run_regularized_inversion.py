from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(project_root / "src"),
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from georock2d.model import create_velocity_model

from georock2d.regularization import (
    build_first_order_smoothness_matrix,
)

from georock2d.inversion import (
    invert_active_slowness,
    active_model_to_full_velocity,
)


def calculate_rmse(
    observed,
    predicted,
):
    """
    Calculate root-mean-square error.
    """

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (observed - predicted) ** 2
            )
        )
    )


def main():
    """
    Run the first GeoRock-2D regularized travel-time inversion.
    """

    # ========================================================
    # SETTINGS
    # ========================================================

    regularization_lambda = 10.0
    reference_velocity_m_s = 1500.0
    damping_ratio = 0.1

    # ========================================================
    # INPUT PATHS
    # ========================================================

    matrix_file = (
        project_root
        / "results"
        / "models"
        / "sensitivity_matrix.npz"
    )

    data_file = (
        project_root
        / "data"
        / "synthetic"
        / "linear_travel_times_noisy.csv"
    )

    if not matrix_file.exists():
        raise FileNotFoundError(
            f"Sensitivity matrix not found:\n"
            f"{matrix_file}"
        )

    if not data_file.exists():
        raise FileNotFoundError(
            f"Noisy dataset not found:\n"
            f"{data_file}"
        )

    # ========================================================
    # LOAD TRUE MODEL
    # ========================================================

    x, z, true_velocity, true_rockhead = (
        create_velocity_model()
    )

    # ========================================================
    # LOAD SENSITIVITY MATRIX
    # ========================================================

    matrix_data = np.load(
        matrix_file
    )

    sensitivity_matrix = matrix_data[
        "sensitivity_matrix"
    ]

    ray_density = matrix_data[
        "ray_density"
    ]

    # ========================================================
    # LOAD OBSERVED DATA
    # ========================================================

    dataset = pd.read_csv(
        data_file
    )

    observed_travel_times_ms = dataset[
        "observed_travel_time_ms"
    ].to_numpy(
        dtype=float
    )

    if (
        len(observed_travel_times_ms)
        != sensitivity_matrix.shape[0]
    ):
        raise ValueError(
            "Number of observations does not match "
            "the sensitivity-matrix rows."
        )

    # ========================================================
    # ACTIVE CELLS
    # ========================================================

    active_mask = ray_density > 0.0

    (
        smoothness_matrix,
        active_flat_indices,
    ) = build_first_order_smoothness_matrix(
        active_mask=active_mask,
    )

    # ========================================================
    # RUN INVERSION
    # ========================================================

    print("=" * 70)
    print("GeoRock-2D Regularized Inversion")
    print("=" * 70)

    print(
        f"\nRegularization lambda: "
        f"{regularization_lambda}"
    )

    print(
        f"Reference velocity: "
        f"{reference_velocity_m_s:.1f} m/s"
    )

    print(
        f"Active cells: "
        f"{len(active_flat_indices)}"
    )

    print(
        f"Smoothness equations: "
        f"{smoothness_matrix.shape[0]}"
    )

    (
        inversion_result,
        estimated_slowness_ms_m,
        predicted_travel_times_ms,
    ) = invert_active_slowness(
        sensitivity_matrix=sensitivity_matrix,
        observed_travel_times_ms=(
            observed_travel_times_ms
        ),
        smoothness_matrix=smoothness_matrix,
        active_flat_indices=(
            active_flat_indices
        ),
        regularization_lambda=(
            regularization_lambda
        ),
        reference_velocity_m_s=(
            reference_velocity_m_s
        ),
        damping_ratio=damping_ratio,
    )

    estimated_velocity = (
        active_model_to_full_velocity(
            estimated_slowness_ms_m=(
                estimated_slowness_ms_m
            ),
            active_flat_indices=(
                active_flat_indices
            ),
            full_model_shape=(
                true_velocity.shape
            ),
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    travel_time_rmse_ms = calculate_rmse(
        observed_travel_times_ms,
        predicted_travel_times_ms,
    )

    active_true_velocity = (
        true_velocity.ravel()[
            active_flat_indices
        ]
    )

    active_estimated_velocity = (
        estimated_velocity.ravel()[
            active_flat_indices
        ]
    )

    velocity_rmse_m_s = calculate_rmse(
        active_true_velocity,
        active_estimated_velocity,
    )

    mean_absolute_velocity_error = float(
        np.mean(
            np.abs(
                active_true_velocity
                - active_estimated_velocity
            )
        )
    )

    # ========================================================
    # SAVE MODEL
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

    model_output_file = (
        model_output_folder
        / "first_regularized_inversion.npz"
    )

    np.savez_compressed(
        model_output_file,
        estimated_velocity_m_s=(
            estimated_velocity
        ),
        predicted_travel_times_ms=(
            predicted_travel_times_ms
        ),
        observed_travel_times_ms=(
            observed_travel_times_ms
        ),
        active_mask=active_mask,
        regularization_lambda=(
            regularization_lambda
        ),
        x=x,
        z=z,
    )

    # ========================================================
    # PLOT INVERTED MODEL
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
        estimated_velocity,
        shading="auto",
        cmap="viridis",
        vmin=500.0,
        vmax=3000.0,
    )

    axis.plot(
        x,
        true_rockhead,
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
        "GeoRock-2D First Regularized Velocity Inversion"
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
        "Estimated P-wave velocity (m/s)"
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
        / "05_first_regularized_inversion.png"
    )

    figure.savefig(
        figure_output_file,
        dpi=300,
        bbox_inches="tight",
    )

    # ========================================================
    # SAVE DATA FIT
    # ========================================================

    fit_table = dataset.copy()

    fit_table[
        "predicted_travel_time_ms"
    ] = predicted_travel_times_ms

    fit_table[
        "travel_time_residual_ms"
    ] = (
        observed_travel_times_ms
        - predicted_travel_times_ms
    )

    table_output_folder = (
        project_root
        / "results"
        / "tables"
    )

    table_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit_output_file = (
        table_output_folder
        / "first_inversion_data_fit.csv"
    )

    fit_table.to_csv(
        fit_output_file,
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        f"\nInversion success: "
        f"{inversion_result.success}"
    )

    print(
        f"Solver message: "
        f"{inversion_result.message}"
    )

    print(
        "\nTravel-time RMSE: "
        f"{travel_time_rmse_ms:.4f} ms"
    )

    print(
        "Active-cell velocity RMSE: "
        f"{velocity_rmse_m_s:.2f} m/s"
    )

    print(
        "Active-cell mean absolute velocity error: "
        f"{mean_absolute_velocity_error:.2f} m/s"
    )

    print(
        "\nEstimated active-cell velocity range: "
        f"{np.nanmin(estimated_velocity):.1f}–"
        f"{np.nanmax(estimated_velocity):.1f} m/s"
    )

    print(
        f"\nModel saved to:\n"
        f"{model_output_file}"
    )

    print(
        f"\nData-fit table saved to:\n"
        f"{fit_output_file}"
    )

    print(
        f"\nFigure saved to:\n"
        f"{figure_output_file}"
    )

    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()