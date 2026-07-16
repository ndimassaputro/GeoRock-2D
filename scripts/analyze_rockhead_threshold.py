from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


# ============================================================
# PROJECT PATH
# ============================================================

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(project_root / "src"),
)


# ============================================================
# PROJECT IMPORT
# ============================================================

from georock2d.model import create_velocity_model


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def smooth_nan_velocity(
    velocity_model,
    sigma_vertical=1.0,
    sigma_horizontal=1.2,
):
    """
    Smooth a velocity model while preserving unsupported NaN regions.

    Parameters
    ----------
    velocity_model : numpy.ndarray
        Two-dimensional velocity model.

    sigma_vertical : float
        Gaussian smoothing scale in vertical grid cells.

    sigma_horizontal : float
        Gaussian smoothing scale in horizontal grid cells.

    Returns
    -------
    smoothed_velocity : numpy.ndarray
        Smoothed velocity model.
    """

    finite_mask = np.isfinite(
        velocity_model
    )

    weighted_velocity = np.where(
        finite_mask,
        velocity_model,
        0.0,
    )

    smoothed_values = gaussian_filter(
        weighted_velocity,
        sigma=(
            sigma_vertical,
            sigma_horizontal,
        ),
    )

    smoothed_weights = gaussian_filter(
        finite_mask.astype(float),
        sigma=(
            sigma_vertical,
            sigma_horizontal,
        ),
    )

    smoothed_velocity = np.full_like(
        velocity_model,
        np.nan,
        dtype=float,
    )

    supported_mask = (
        smoothed_weights > 1.0e-8
    )

    smoothed_velocity[
        supported_mask
    ] = (
        smoothed_values[
            supported_mask
        ]
        / smoothed_weights[
            supported_mask
        ]
    )

    return smoothed_velocity


def extract_threshold_rockhead(
    velocity_model,
    ray_density,
    z,
    velocity_threshold_m_s,
    minimum_depth_m=10.0,
    maximum_depth_m=32.0,
    minimum_ray_density=2.0,
):
    """
    Estimate rockhead using a velocity-threshold crossing.

    The first valid transition from velocity below the threshold
    to velocity at or above the threshold is selected.

    Parameters
    ----------
    velocity_model : numpy.ndarray
        Smoothed estimated velocity model.

    ray_density : numpy.ndarray
        Cumulative ray-path length in each cell.

    z : numpy.ndarray
        Cell-centre depth coordinates.

    velocity_threshold_m_s : float
        Velocity threshold used for interface extraction.

    minimum_depth_m : float
        Minimum accepted interface depth.

    maximum_depth_m : float
        Maximum accepted interface depth.

    minimum_ray_density : float
        Minimum ray density required in adjacent cells.

    Returns
    -------
    estimated_rockhead_m : numpy.ndarray
        Estimated rockhead depths.
    """

    number_of_columns = (
        velocity_model.shape[1]
    )

    estimated_rockhead_m = np.full(
        number_of_columns,
        np.nan,
        dtype=float,
    )

    for column_id in range(
        number_of_columns
    ):

        velocity_column = velocity_model[
            :,
            column_id
        ]

        density_column = ray_density[
            :,
            column_id
        ]

        for upper_index in range(
            len(z) - 1
        ):

            lower_index = upper_index + 1

            upper_depth = z[
                upper_index
            ]

            lower_depth = z[
                lower_index
            ]

            if upper_depth < minimum_depth_m:
                continue

            if lower_depth > maximum_depth_m:
                break

            upper_velocity = velocity_column[
                upper_index
            ]

            lower_velocity = velocity_column[
                lower_index
            ]

            if not (
                np.isfinite(
                    upper_velocity
                )
                and np.isfinite(
                    lower_velocity
                )
            ):
                continue

            if (
                density_column[
                    upper_index
                ]
                < minimum_ray_density
                or density_column[
                    lower_index
                ]
                < minimum_ray_density
            ):
                continue

            threshold_crossed = (
                upper_velocity
                < velocity_threshold_m_s
                <= lower_velocity
            )

            if not threshold_crossed:
                continue

            velocity_difference = (
                lower_velocity
                - upper_velocity
            )

            if velocity_difference <= 0:
                continue

            fraction = (
                velocity_threshold_m_s
                - upper_velocity
            ) / velocity_difference

            estimated_depth = (
                upper_depth
                + fraction
                * (
                    lower_depth
                    - upper_depth
                )
            )

            estimated_rockhead_m[
                column_id
            ] = estimated_depth

            break

    return estimated_rockhead_m


def calculate_metrics(
    true_rockhead,
    estimated_rockhead,
):
    """
    Calculate interface-recovery metrics.
    """

    valid_mask = np.isfinite(
        estimated_rockhead
    )

    number_of_valid = np.count_nonzero(
        valid_mask
    )

    if number_of_valid == 0:
        return {
            "number_of_valid_positions": 0,
            "coverage_percentage": 0.0,
            "mean_error_m": np.nan,
            "mean_absolute_error_m": np.nan,
            "rmse_m": np.nan,
            "maximum_absolute_error_m": np.nan,
        }

    errors = (
        estimated_rockhead[
            valid_mask
        ]
        - true_rockhead[
            valid_mask
        ]
    )

    return {
        "number_of_valid_positions": (
            number_of_valid
        ),
        "coverage_percentage": (
            number_of_valid
            / len(true_rockhead)
            * 100.0
        ),
        "mean_error_m": float(
            np.mean(errors)
        ),
        "mean_absolute_error_m": float(
            np.mean(
                np.abs(errors)
            )
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(
                    errors**2
                )
            )
        ),
        "maximum_absolute_error_m": float(
            np.max(
                np.abs(errors)
            )
        ),
    }


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """
    Evaluate rockhead extraction sensitivity to velocity threshold.
    """

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    threshold_values_m_s = np.array(
        [
            1400.0,
            1500.0,
            1600.0,
            1700.0,
            1800.0,
            1900.0,
            2000.0,
        ],
        dtype=float,
    )

    minimum_depth_m = 10.0
    maximum_depth_m = 32.0
    minimum_ray_density = 2.0

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    inversion_file = (
        project_root
        / "results"
        / "models"
        / "regularization_tests"
        / "inversion_lambda_3.npz"
    )

    sensitivity_file = (
        project_root
        / "results"
        / "models"
        / "sensitivity_matrix.npz"
    )

    if not inversion_file.exists():
        raise FileNotFoundError(
            f"Inversion model not found:\n"
            f"{inversion_file}"
        )

    if not sensitivity_file.exists():
        raise FileNotFoundError(
            f"Sensitivity model not found:\n"
            f"{sensitivity_file}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    inversion_data = np.load(
        inversion_file
    )

    estimated_velocity = inversion_data[
        "estimated_velocity_m_s"
    ]

    sensitivity_data = np.load(
        sensitivity_file
    )

    ray_density = sensitivity_data[
        "ray_density"
    ]

    x, z, true_velocity, true_rockhead = (
        create_velocity_model()
    )

    # --------------------------------------------------------
    # SMOOTH VELOCITY MODEL
    # --------------------------------------------------------

    smoothed_velocity = smooth_nan_velocity(
        velocity_model=estimated_velocity,
        sigma_vertical=1.0,
        sigma_horizontal=1.2,
    )

    # --------------------------------------------------------
    # TEST ALL THRESHOLDS
    # --------------------------------------------------------

    records = []

    rockhead_results = {}

    print("=" * 72)
    print("GeoRock-2D Rockhead Threshold Sensitivity")
    print("=" * 72)

    for threshold in threshold_values_m_s:

        estimated_rockhead = (
            extract_threshold_rockhead(
                velocity_model=smoothed_velocity,
                ray_density=ray_density,
                z=z,
                velocity_threshold_m_s=float(
                    threshold
                ),
                minimum_depth_m=minimum_depth_m,
                maximum_depth_m=maximum_depth_m,
                minimum_ray_density=(
                    minimum_ray_density
                ),
            )
        )

        metrics = calculate_metrics(
            true_rockhead=true_rockhead,
            estimated_rockhead=(
                estimated_rockhead
            ),
        )

        records.append(
            {
                "velocity_threshold_m_s": float(
                    threshold
                ),
                **metrics,
            }
        )

        rockhead_results[
            float(threshold)
        ] = estimated_rockhead

        print(
            f"\nThreshold = {threshold:.0f} m/s"
        )

        print(
            "  Valid positions: "
            f"{metrics['number_of_valid_positions']} "
            f"/ {len(x)}"
        )

        print(
            "  Coverage: "
            f"{metrics['coverage_percentage']:.2f}%"
        )

        print(
            "  Mean error: "
            f"{metrics['mean_error_m']:.3f} m"
        )

        print(
            "  MAE: "
            f"{metrics['mean_absolute_error_m']:.3f} m"
        )

        print(
            "  RMSE: "
            f"{metrics['rmse_m']:.3f} m"
        )

    # --------------------------------------------------------
    # CREATE SUMMARY TABLE
    # --------------------------------------------------------

    results = pd.DataFrame(
        records
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

    summary_file = (
        table_output_folder
        / "rockhead_threshold_sensitivity.csv"
    )

    results.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # SELECT SYNTHETIC BENCHMARK THRESHOLD
    # --------------------------------------------------------

    valid_results = results.dropna(
        subset=[
            "rmse_m",
        ]
    )

    if valid_results.empty:
        raise RuntimeError(
            "No threshold produced a valid interface estimate."
        )

    # Require at least 25% profile coverage
    coverage_filtered = valid_results[
        valid_results[
            "coverage_percentage"
        ] >= 25.0
    ]

    if coverage_filtered.empty:
        coverage_filtered = valid_results

    best_row = coverage_filtered.loc[
        coverage_filtered[
            "rmse_m"
        ].idxmin()
    ]

    selected_threshold = float(
        best_row[
            "velocity_threshold_m_s"
        ]
    )

    selected_rockhead = (
        rockhead_results[
            selected_threshold
        ]
    )

    selected_valid_mask = np.isfinite(
        selected_rockhead
    )

    # --------------------------------------------------------
    # FIGURE 1: ERROR AND COVERAGE
    # --------------------------------------------------------

    figure, primary_axis = plt.subplots(
        figsize=(8.5, 5.5),
    )

    primary_axis.plot(
        results[
            "velocity_threshold_m_s"
        ],
        results[
            "rmse_m"
        ],
        marker="o",
        label="Rockhead RMSE",
    )

    primary_axis.plot(
        results[
            "velocity_threshold_m_s"
        ],
        results[
            "mean_absolute_error_m"
        ],
        marker="s",
        label="Rockhead MAE",
    )

    primary_axis.set_xlabel(
        "Velocity threshold (m/s)"
    )

    primary_axis.set_ylabel(
        "Rockhead error (m)"
    )

    primary_axis.grid(
        visible=True,
        alpha=0.3,
    )

    secondary_axis = (
        primary_axis.twinx()
    )

    secondary_axis.plot(
        results[
            "velocity_threshold_m_s"
        ],
        results[
            "coverage_percentage"
        ],
        marker="^",
        linestyle="--",
        label="Valid-profile coverage",
    )

    secondary_axis.set_ylabel(
        "Valid-profile coverage (%)"
    )

    lines_1, labels_1 = (
        primary_axis.get_legend_handles_labels()
    )

    lines_2, labels_2 = (
        secondary_axis.get_legend_handles_labels()
    )

    primary_axis.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    primary_axis.set_title(
        "Rockhead Threshold Sensitivity"
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

    sensitivity_figure_file = (
        figure_output_folder
        / "12_threshold_sensitivity.png"
    )

    figure.savefig(
        sensitivity_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 2: SELECTED THRESHOLD RESULT
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9.5, 4.8),
    )

    axis.plot(
        x,
        true_rockhead,
        linewidth=2.0,
        label="True rockhead",
    )

    axis.plot(
        x[
            selected_valid_mask
        ],
        selected_rockhead[
            selected_valid_mask
        ],
        linestyle="--",
        marker="o",
        linewidth=2.0,
        markersize=4.0,
        label=(
            f"Selected threshold "
            f"({selected_threshold:.0f} m/s)"
        ),
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Rockhead depth (m)"
    )

    axis.set_title(
        "Selected Coverage-Aware Rockhead Estimate"
    )

    axis.set_xlim(
        0.0,
        120.0,
    )

    axis.invert_yaxis()

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    selected_figure_file = (
        figure_output_folder
        / "13_selected_threshold_rockhead.png"
    )

    figure.savefig(
        selected_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # SAVE SELECTED RESULT
    # --------------------------------------------------------

    selected_table = pd.DataFrame(
        {
            "x_m": x,
            "true_rockhead_depth_m": (
                true_rockhead
            ),
            "estimated_rockhead_depth_m": (
                selected_rockhead
            ),
            "rockhead_error_m": (
                selected_rockhead
                - true_rockhead
            ),
            "valid_estimate": (
                selected_valid_mask
            ),
        }
    )

    selected_table_file = (
        table_output_folder
        / "selected_threshold_rockhead.csv"
    )

    selected_table.to_csv(
        selected_table_file,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 72)

    print(
        "Threshold sensitivity analysis completed."
    )

    print(
        "\nSelected synthetic benchmark threshold:"
    )

    print(
        f"Threshold = "
        f"{selected_threshold:.0f} m/s"
    )

    print(
        f"Coverage = "
        f"{best_row['coverage_percentage']:.2f}%"
    )

    print(
        f"MAE = "
        f"{best_row['mean_absolute_error_m']:.3f} m"
    )

    print(
        f"RMSE = "
        f"{best_row['rmse_m']:.3f} m"
    )

    print(
        "\nImportant limitation:"
    )

    print(
        "Selection using the known true rockhead is valid only "
        "for this synthetic benchmark."
    )

    print(
        "For field data, threshold selection would require "
        "borehole calibration, geological knowledge, or "
        "independent geophysical constraints."
    )

    print(
        f"\nSummary table saved to:\n"
        f"{summary_file}"
    )

    print(
        f"\nSelected result saved to:\n"
        f"{selected_table_file}"
    )

    print(
        f"\nSensitivity figure saved to:\n"
        f"{sensitivity_figure_file}"
    )

    print(
        f"\nSelected rockhead figure saved to:\n"
        f"{selected_figure_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()