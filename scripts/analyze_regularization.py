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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def calculate_roughness_norm(
    smoothness_matrix,
    slowness_model,
):
    """
    Calculate the L2 norm of spatial slowness differences.
    """

    roughness_vector = (
        smoothness_matrix
        @ slowness_model
    )

    return float(
        np.linalg.norm(
            roughness_vector
        )
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """
    Test multiple regularization strengths and create
    regularization diagnostic outputs.
    """

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    lambda_values = np.array(
        [
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
            30.0,
            100.0,
        ],
        dtype=float,
    )

    reference_velocity_m_s = 1500.0
    damping_ratio = 0.1

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

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
            f"Sensitivity matrix not found:\n{matrix_file}"
        )

    if not data_file.exists():
        raise FileNotFoundError(
            f"Noisy travel-time dataset not found:\n{data_file}"
        )

    # --------------------------------------------------------
    # LOAD TRUE MODEL
    # --------------------------------------------------------

    x, z, true_velocity, true_rockhead = (
        create_velocity_model()
    )

    # --------------------------------------------------------
    # LOAD SENSITIVITY INFORMATION
    # --------------------------------------------------------

    matrix_data = np.load(
        matrix_file
    )

    sensitivity_matrix = matrix_data[
        "sensitivity_matrix"
    ]

    ray_density = matrix_data[
        "ray_density"
    ]

    # --------------------------------------------------------
    # LOAD OBSERVATIONS
    # --------------------------------------------------------

    dataset = pd.read_csv(
        data_file
    )

    observed_travel_times_ms = dataset[
        "observed_travel_time_ms"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # BUILD ACTIVE MODEL AND SMOOTHNESS MATRIX
    # --------------------------------------------------------

    active_mask = ray_density > 0.0

    (
        smoothness_matrix,
        active_flat_indices,
    ) = build_first_order_smoothness_matrix(
        active_mask=active_mask,
    )

    active_true_velocity = (
        true_velocity.ravel()[
            active_flat_indices
        ]
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    model_output_folder = (
        project_root
        / "results"
        / "models"
        / "regularization_tests"
    )

    table_output_folder = (
        project_root
        / "results"
        / "tables"
    )

    figure_output_folder = (
        project_root
        / "results"
        / "figures"
    )

    model_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # RUN ALL REGULARIZATION TESTS
    # --------------------------------------------------------

    records = []

    print("=" * 72)
    print("GeoRock-2D Regularization Analysis")
    print("=" * 72)

    for lambda_value in lambda_values:

        print(
            f"\nRunning inversion with "
            f"lambda = {lambda_value:g}"
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
            regularization_lambda=float(
                lambda_value
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

        active_estimated_velocity = (
            estimated_velocity.ravel()[
                active_flat_indices
            ]
        )

        travel_time_rmse_ms = calculate_rmse(
            observed_travel_times_ms,
            predicted_travel_times_ms,
        )

        velocity_rmse_m_s = calculate_rmse(
            active_true_velocity,
            active_estimated_velocity,
        )

        mean_absolute_velocity_error_m_s = float(
            np.mean(
                np.abs(
                    active_true_velocity
                    - active_estimated_velocity
                )
            )
        )

        roughness_norm = calculate_roughness_norm(
            smoothness_matrix=(
                smoothness_matrix
            ),
            slowness_model=(
                estimated_slowness_ms_m
            ),
        )

        data_residual_norm = float(
            np.linalg.norm(
                observed_travel_times_ms
                - predicted_travel_times_ms
            )
        )

        records.append(
            {
                "lambda": float(
                    lambda_value
                ),
                "inversion_success": bool(
                    inversion_result.success
                ),
                "travel_time_rmse_ms": (
                    travel_time_rmse_ms
                ),
                "data_residual_norm_ms": (
                    data_residual_norm
                ),
                "roughness_norm_ms_per_m": (
                    roughness_norm
                ),
                "velocity_rmse_m_s": (
                    velocity_rmse_m_s
                ),
                "mean_absolute_velocity_error_m_s": (
                    mean_absolute_velocity_error_m_s
                ),
                "minimum_estimated_velocity_m_s": float(
                    np.nanmin(
                        estimated_velocity
                    )
                ),
                "maximum_estimated_velocity_m_s": float(
                    np.nanmax(
                        estimated_velocity
                    )
                ),
            }
        )

        model_output_file = (
            model_output_folder
            / f"inversion_lambda_{lambda_value:g}.npz"
        )

        np.savez_compressed(
            model_output_file,
            estimated_velocity_m_s=(
                estimated_velocity
            ),
            estimated_slowness_ms_m=(
                estimated_slowness_ms_m
            ),
            predicted_travel_times_ms=(
                predicted_travel_times_ms
            ),
            observed_travel_times_ms=(
                observed_travel_times_ms
            ),
            active_mask=active_mask,
            lambda_value=float(
                lambda_value
            ),
            x=x,
            z=z,
        )

        print(
            f"  Travel-time RMSE: "
            f"{travel_time_rmse_ms:.4f} ms"
        )

        print(
            f"  Roughness norm: "
            f"{roughness_norm:.4f}"
        )

        print(
            f"  Velocity RMSE: "
            f"{velocity_rmse_m_s:.2f} m/s"
        )

    # --------------------------------------------------------
    # SAVE SUMMARY TABLE
    # --------------------------------------------------------

    results = pd.DataFrame(
        records
    )

    summary_file = (
        table_output_folder
        / "regularization_analysis.csv"
    )

    results.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # IDENTIFY SYNTHETIC BENCHMARK BEST LAMBDA
    # --------------------------------------------------------

    best_velocity_row = results.loc[
        results[
            "velocity_rmse_m_s"
        ].idxmin()
    ]

    best_velocity_lambda = float(
        best_velocity_row[
            "lambda"
        ]
    )

    # Important:
    # This criterion uses the known true synthetic model.
    # It is valid for benchmark evaluation but unavailable
    # for real field data.

    # --------------------------------------------------------
    # FIGURE 1: L-CURVE
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(7.5, 6.0),
    )

    axis.plot(
        results[
            "data_residual_norm_ms"
        ],
        results[
            "roughness_norm_ms_per_m"
        ],
        marker="o",
    )

    for _, row in results.iterrows():

        axis.annotate(
            f"λ={row['lambda']:g}",
            (
                row[
                    "data_residual_norm_ms"
                ],
                row[
                    "roughness_norm_ms_per_m"
                ],
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    axis.set_xscale(
        "log"
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlabel(
        "Data residual norm (ms)"
    )

    axis.set_ylabel(
        "Model roughness norm (ms/m)"
    )

    axis.set_title(
        "GeoRock-2D Regularization Diagnostic"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    figure.tight_layout()

    l_curve_file = (
        figure_output_folder
        / "06_regularization_l_curve.png"
    )

    figure.savefig(
        l_curve_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 2: ERROR VERSUS LAMBDA
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(8.0, 5.5),
    )

    axis.plot(
        results["lambda"],
        results[
            "velocity_rmse_m_s"
        ],
        marker="o",
        label="Active-cell velocity RMSE",
    )

    axis.set_xscale(
        "log"
    )

    axis.set_xlabel(
        "Regularization parameter λ"
    )

    axis.set_ylabel(
        "Velocity RMSE (m/s)"
    )

    axis.set_title(
        "Synthetic Recovery Error versus Regularization"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    error_curve_file = (
        figure_output_folder
        / "07_velocity_error_vs_lambda.png"
    )

    figure.savefig(
        error_curve_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # PRINT FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 72)

    print(
        "Regularization analysis completed."
    )

    print(
        "\nLowest synthetic active-cell velocity RMSE:"
    )

    print(
        f"Lambda = {best_velocity_lambda:g}"
    )

    print(
        "Velocity RMSE = "
        f"{best_velocity_row['velocity_rmse_m_s']:.2f} m/s"
    )

    print(
        "Travel-time RMSE = "
        f"{best_velocity_row['travel_time_rmse_ms']:.4f} ms"
    )

    print(
        "\nImportant interpretation:"
    )

    print(
        "The minimum velocity RMSE uses the known synthetic true model."
    )

    print(
        "For real field data, lambda selection must rely on "
        "diagnostics such as the L-curve, cross-validation, "
        "data uncertainty, or independent borehole information."
    )

    print(
        f"\nSummary table saved to:\n{summary_file}"
    )

    print(
        f"\nL-curve saved to:\n{l_curve_file}"
    )

    print(
        f"\nVelocity-error curve saved to:\n{error_curve_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()