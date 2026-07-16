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

from georock2d.boreholes import (
    create_synthetic_boreholes,
    build_geophysics_baseline_profile,
    apply_borehole_correction,
)


# ============================================================
# METRICS
# ============================================================

def calculate_profile_metrics(
    true_profile,
    estimated_profile,
):
    """
    Calculate full-profile rockhead error metrics.
    """

    errors = (
        estimated_profile
        - true_profile
    )

    return {
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
    Integrate partial geophysical rockhead estimates with sparse
    synthetic borehole observations.
    """

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    borehole_x_positions = np.array(
        [
            18.0,
            58.0,
            103.0,
        ],
        dtype=float,
    )

    borehole_noise_std_m = 0.25

    borehole_random_seed = 20260716

    influence_length_m = 22.0

    # --------------------------------------------------------
    # INPUT FILE
    # --------------------------------------------------------

    geophysical_file = (
        project_root
        / "results"
        / "tables"
        / "selected_threshold_rockhead.csv"
    )

    if not geophysical_file.exists():
        raise FileNotFoundError(
            f"Selected geophysical rockhead file not found:\n"
            f"{geophysical_file}"
        )

    # --------------------------------------------------------
    # TRUE MODEL
    # --------------------------------------------------------

    x, z, true_velocity, true_rockhead = (
        create_velocity_model()
    )

    # --------------------------------------------------------
    # LOAD GEOPHYSICAL ROCKHEAD
    # --------------------------------------------------------

    geophysical_table = pd.read_csv(
        geophysical_file
    )

    geophysical_rockhead = geophysical_table[
        "estimated_rockhead_depth_m"
    ].to_numpy(
        dtype=float
    )

    if len(
        geophysical_rockhead
    ) != len(x):
        raise ValueError(
            "Geophysical profile length does not match model x coordinates."
        )

    geophysical_valid_mask = np.isfinite(
        geophysical_rockhead
    )

    # --------------------------------------------------------
    # CREATE SYNTHETIC BOREHOLES
    # --------------------------------------------------------

    borehole_table = (
        create_synthetic_boreholes(
            x=x,
            true_rockhead=true_rockhead,
            borehole_x_positions=(
                borehole_x_positions
            ),
            noise_std_m=(
                borehole_noise_std_m
            ),
            random_seed=(
                borehole_random_seed
            ),
        )
    )

    # --------------------------------------------------------
    # BUILD CONTINUOUS BASELINE
    # --------------------------------------------------------

    baseline_profile = (
        build_geophysics_baseline_profile(
            x=x,
            geophysical_rockhead=(
                geophysical_rockhead
            ),
            borehole_x=(
                borehole_table[
                    "x_m"
                ].to_numpy()
            ),
            borehole_depth=(
                borehole_table[
                    "observed_rockhead_depth_m"
                ].to_numpy()
            ),
        )
    )

    # --------------------------------------------------------
    # APPLY BOREHOLE CORRECTION
    # --------------------------------------------------------

    integrated_profile = (
        apply_borehole_correction(
            x=x,
            baseline_profile=(
                baseline_profile
            ),
            borehole_x=(
                borehole_table[
                    "x_m"
                ].to_numpy()
            ),
            borehole_depth=(
                borehole_table[
                    "observed_rockhead_depth_m"
                ].to_numpy()
            ),
            influence_length_m=(
                influence_length_m
            ),
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    baseline_metrics = (
        calculate_profile_metrics(
            true_profile=true_rockhead,
            estimated_profile=(
                baseline_profile
            ),
        )
    )

    integrated_metrics = (
        calculate_profile_metrics(
            true_profile=true_rockhead,
            estimated_profile=(
                integrated_profile
            ),
        )
    )

    # Local geophysics-only metrics remain calculated
    # only where geophysical estimates are valid.
    local_geophysical_errors = (
        geophysical_rockhead[
            geophysical_valid_mask
        ]
        - true_rockhead[
            geophysical_valid_mask
        ]
    )

    geophysical_local_rmse = float(
        np.sqrt(
            np.mean(
                local_geophysical_errors**2
            )
        )
    )

    geophysical_local_mae = float(
        np.mean(
            np.abs(
                local_geophysical_errors
            )
        )
    )

    # --------------------------------------------------------
    # SAVE BOREHOLE DATA
    # --------------------------------------------------------

    data_output_folder = (
        project_root
        / "data"
        / "synthetic"
    )

    data_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    borehole_output_file = (
        data_output_folder
        / "synthetic_boreholes.csv"
    )

    borehole_table.to_csv(
        borehole_output_file,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE INTEGRATED PROFILE
    # --------------------------------------------------------

    result_table = pd.DataFrame(
        {
            "x_m": x,
            "true_rockhead_depth_m": (
                true_rockhead
            ),
            "geophysical_rockhead_depth_m": (
                geophysical_rockhead
            ),
            "baseline_interpolated_depth_m": (
                baseline_profile
            ),
            "integrated_rockhead_depth_m": (
                integrated_profile
            ),
            "integrated_error_m": (
                integrated_profile
                - true_rockhead
            ),
        }
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

    result_output_file = (
        table_output_folder
        / "borehole_integrated_rockhead.csv"
    )

    result_table.to_csv(
        result_output_file,
        index=False,
    )

    # --------------------------------------------------------
    # FIGURE 1: INTEGRATION RESULT
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(10.5, 5.5),
    )

    axis.plot(
        x,
        true_rockhead,
        linewidth=2.5,
        label="True rockhead",
    )

    axis.plot(
        x[
            geophysical_valid_mask
        ],
        geophysical_rockhead[
            geophysical_valid_mask
        ],
        linestyle="None",
        marker="o",
        markersize=5.0,
        label="Geophysics-only valid estimates",
    )

    axis.plot(
        x,
        baseline_profile,
        linestyle="--",
        linewidth=1.8,
        label="Continuous baseline interpolation",
    )

    axis.plot(
        x,
        integrated_profile,
        linewidth=2.5,
        label="Borehole-constrained interpretation",
    )

    axis.scatter(
        borehole_table[
            "x_m"
        ],
        borehole_table[
            "observed_rockhead_depth_m"
        ],
        marker="v",
        s=90,
        label="Synthetic boreholes",
        zorder=5,
    )

    for _, row in borehole_table.iterrows():

        axis.annotate(
            row[
                "borehole_id"
            ],
            (
                row[
                    "x_m"
                ],
                row[
                    "observed_rockhead_depth_m"
                ],
            ),
            xytext=(
                0,
                -18,
            ),
            textcoords="offset points",
            horizontalalignment="center",
        )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Rockhead depth (m)"
    )

    axis.set_title(
        "Geophysics and Sparse-Borehole Rockhead Integration"
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

    figure_output_folder = (
        project_root
        / "results"
        / "figures"
    )

    figure_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    integration_figure_file = (
        figure_output_folder
        / "14_borehole_integrated_rockhead.png"
    )

    figure.savefig(
        integration_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 2: ERROR ALONG PROFILE
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9.5, 4.8),
    )

    baseline_error = (
        baseline_profile
        - true_rockhead
    )

    integrated_error = (
        integrated_profile
        - true_rockhead
    )

    axis.plot(
        x,
        baseline_error,
        linewidth=2.0,
        label="Baseline interpolation error",
    )

    axis.plot(
        x,
        integrated_error,
        linewidth=2.0,
        label="Borehole-constrained error",
    )

    axis.axhline(
        y=0.0,
        linewidth=1.0,
    )

    axis.scatter(
        borehole_table[
            "x_m"
        ],
        np.zeros(
            len(
                borehole_table
            )
        ),
        marker="v",
        s=70,
        label="Borehole positions",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Rockhead depth error (m)"
    )

    axis.set_title(
        "Rockhead Interpretation Error Along Profile"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    error_figure_file = (
        figure_output_folder
        / "15_borehole_integration_error.png"
    )

    figure.savefig(
        error_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # SAVE METRIC SUMMARY
    # --------------------------------------------------------

    metrics_table = pd.DataFrame(
        [
            {
                "scenario": (
                    "Geophysics-only valid region"
                ),
                "spatial_coverage_percent": (
                    np.count_nonzero(
                        geophysical_valid_mask
                    )
                    / len(x)
                    * 100.0
                ),
                "mae_m": (
                    geophysical_local_mae
                ),
                "rmse_m": (
                    geophysical_local_rmse
                ),
                "maximum_absolute_error_m": float(
                    np.max(
                        np.abs(
                            local_geophysical_errors
                        )
                    )
                ),
            },
            {
                "scenario": (
                    "Full-profile baseline interpolation"
                ),
                "spatial_coverage_percent": (
                    100.0
                ),
                "mae_m": (
                    baseline_metrics[
                        "mean_absolute_error_m"
                    ]
                ),
                "rmse_m": (
                    baseline_metrics[
                        "rmse_m"
                    ]
                ),
                "maximum_absolute_error_m": (
                    baseline_metrics[
                        "maximum_absolute_error_m"
                    ]
                ),
            },
            {
                "scenario": (
                    "Borehole-constrained full profile"
                ),
                "spatial_coverage_percent": (
                    100.0
                ),
                "mae_m": (
                    integrated_metrics[
                        "mean_absolute_error_m"
                    ]
                ),
                "rmse_m": (
                    integrated_metrics[
                        "rmse_m"
                    ]
                ),
                "maximum_absolute_error_m": (
                    integrated_metrics[
                        "maximum_absolute_error_m"
                    ]
                ),
            },
        ]
    )

    metrics_output_file = (
        table_output_folder
        / "borehole_integration_metrics.csv"
    )

    metrics_table.to_csv(
        metrics_output_file,
        index=False,
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("=" * 74)
    print("GeoRock-2D Borehole-Constrained Interpretation")
    print("=" * 74)

    print(
        f"\nNumber of boreholes: "
        f"{len(borehole_table)}"
    )

    print(
        f"Borehole noise standard deviation: "
        f"{borehole_noise_std_m:.3f} m"
    )

    print(
        f"Correction influence length: "
        f"{influence_length_m:.1f} m"
    )

    print(
        "\nSynthetic borehole observations:"
    )

    print(
        borehole_table.to_string(
            index=False
        )
    )

    print(
        "\nGeophysics-only valid region:"
    )

    print(
        f"Coverage = "
        f"{np.count_nonzero(geophysical_valid_mask) / len(x) * 100.0:.2f}%"
    )

    print(
        f"MAE = "
        f"{geophysical_local_mae:.3f} m"
    )

    print(
        f"RMSE = "
        f"{geophysical_local_rmse:.3f} m"
    )

    print(
        "\nFull-profile baseline interpolation:"
    )

    print(
        f"MAE = "
        f"{baseline_metrics['mean_absolute_error_m']:.3f} m"
    )

    print(
        f"RMSE = "
        f"{baseline_metrics['rmse_m']:.3f} m"
    )

    print(
        "\nBorehole-constrained full profile:"
    )

    print(
        f"MAE = "
        f"{integrated_metrics['mean_absolute_error_m']:.3f} m"
    )

    print(
        f"RMSE = "
        f"{integrated_metrics['rmse_m']:.3f} m"
    )

    print(
        f"Maximum absolute error = "
        f"{integrated_metrics['maximum_absolute_error_m']:.3f} m"
    )

    print(
        f"\nBoreholes saved to:\n"
        f"{borehole_output_file}"
    )

    print(
        f"\nIntegrated profile saved to:\n"
        f"{result_output_file}"
    )

    print(
        f"\nMetrics saved to:\n"
        f"{metrics_output_file}"
    )

    print(
        f"\nIntegration figure saved to:\n"
        f"{integration_figure_file}"
    )

    print(
        f"\nError figure saved to:\n"
        f"{error_figure_file}"
    )

    print("=" * 74)


if __name__ == "__main__":
    main()