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
# HELPER FUNCTION
# ============================================================

def calculate_profile_metrics(
    true_profile,
    estimated_profile,
):
    """
    Calculate full-profile rockhead error metrics.

    Parameters
    ----------
    true_profile : numpy.ndarray
        True synthetic rockhead depth.

    estimated_profile : numpy.ndarray
        Estimated full-profile rockhead depth.

    Returns
    -------
    dict
        Mean error, MAE, RMSE, and maximum absolute error.
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
    Test multiple borehole correction influence lengths.
    """

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    influence_lengths_m = np.array(
        [
            5.0,
            8.0,
            10.0,
            12.0,
            15.0,
            18.0,
            22.0,
            25.0,
            30.0,
            40.0,
        ],
        dtype=float,
    )

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
    # LOAD GEOPHYSICAL ESTIMATE
    # --------------------------------------------------------

    geophysical_table = pd.read_csv(
        geophysical_file
    )

    geophysical_rockhead = geophysical_table[
        "estimated_rockhead_depth_m"
    ].to_numpy(
        dtype=float
    )

    geophysical_valid_mask = np.isfinite(
        geophysical_rockhead
    )

    # --------------------------------------------------------
    # CREATE SYNTHETIC BOREHOLES
    # --------------------------------------------------------

    borehole_table = create_synthetic_boreholes(
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

    # --------------------------------------------------------
    # BUILD BASELINE PROFILE
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

    baseline_metrics = calculate_profile_metrics(
        true_profile=true_rockhead,
        estimated_profile=baseline_profile,
    )

    # --------------------------------------------------------
    # TEST ALL INFLUENCE LENGTHS
    # --------------------------------------------------------

    records = []

    profiles = {}

    print("=" * 74)
    print("GeoRock-2D Borehole Influence-Length Analysis")
    print("=" * 74)

    print(
        "\nBaseline interpolation:"
    )

    print(
        f"MAE = "
        f"{baseline_metrics['mean_absolute_error_m']:.3f} m"
    )

    print(
        f"RMSE = "
        f"{baseline_metrics['rmse_m']:.3f} m"
    )

    for influence_length in influence_lengths_m:

        corrected_profile = apply_borehole_correction(
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
            influence_length_m=float(
                influence_length
            ),
        )

        metrics = calculate_profile_metrics(
            true_profile=true_rockhead,
            estimated_profile=corrected_profile,
        )

        records.append(
            {
                "influence_length_m": float(
                    influence_length
                ),
                **metrics,
            }
        )

        profiles[
            float(
                influence_length
            )
        ] = corrected_profile

        print(
            f"\nInfluence length = "
            f"{influence_length:.1f} m"
        )

        print(
            f"  Mean error = "
            f"{metrics['mean_error_m']:.3f} m"
        )

        print(
            f"  MAE = "
            f"{metrics['mean_absolute_error_m']:.3f} m"
        )

        print(
            f"  RMSE = "
            f"{metrics['rmse_m']:.3f} m"
        )

        print(
            f"  Maximum absolute error = "
            f"{metrics['maximum_absolute_error_m']:.3f} m"
        )

    # --------------------------------------------------------
    # SELECT BEST INFLUENCE LENGTH
    # --------------------------------------------------------

    results = pd.DataFrame(
        records
    )

    best_row = results.loc[
        results[
            "rmse_m"
        ].idxmin()
    ]

    selected_influence_length = float(
        best_row[
            "influence_length_m"
        ]
    )

    selected_profile = profiles[
        selected_influence_length
    ]

    # --------------------------------------------------------
    # SAVE TABLES
    # --------------------------------------------------------

    table_output_folder = (
        project_root
        / "results"
        / "tables"
    )

    table_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    sensitivity_table_file = (
        table_output_folder
        / "borehole_influence_sensitivity.csv"
    )

    results.to_csv(
        sensitivity_table_file,
        index=False,
    )

    selected_profile_table = pd.DataFrame(
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
            "optimized_integrated_depth_m": (
                selected_profile
            ),
            "optimized_integrated_error_m": (
                selected_profile
                - true_rockhead
            ),
        }
    )

    selected_profile_file = (
        table_output_folder
        / "optimized_borehole_integrated_rockhead.csv"
    )

    selected_profile_table.to_csv(
        selected_profile_file,
        index=False,
    )

    # --------------------------------------------------------
    # FIGURE 1: ERROR VERSUS INFLUENCE LENGTH
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(8.5, 5.5),
    )

    axis.plot(
        results[
            "influence_length_m"
        ],
        results[
            "rmse_m"
        ],
        marker="o",
        label="Full-profile RMSE",
    )

    axis.plot(
        results[
            "influence_length_m"
        ],
        results[
            "mean_absolute_error_m"
        ],
        marker="s",
        label="Full-profile MAE",
    )

    axis.axhline(
        y=baseline_metrics[
            "rmse_m"
        ],
        linestyle="--",
        label="Baseline interpolation RMSE",
    )

    axis.axvline(
        x=selected_influence_length,
        linestyle=":",
        label=(
            "Selected influence length "
            f"({selected_influence_length:.1f} m)"
        ),
    )

    axis.set_xlabel(
        "Borehole correction influence length (m)"
    )

    axis.set_ylabel(
        "Rockhead error (m)"
    )

    axis.set_title(
        "Borehole Influence-Length Sensitivity"
    )

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

    sensitivity_figure_file = (
        figure_output_folder
        / "16_borehole_influence_sensitivity.png"
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
    # FIGURE 2: OPTIMIZED INTEGRATED PROFILE
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
        label="Baseline interpolation",
    )

    axis.plot(
        x,
        selected_profile,
        linewidth=2.5,
        label=(
            "Optimized borehole-constrained profile"
        ),
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
            xytext=(0, -18),
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
        "Optimized Geophysics–Borehole Rockhead Interpretation"
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

    optimized_figure_file = (
        figure_output_folder
        / "17_optimized_borehole_integration.png"
    )

    figure.savefig(
        optimized_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 3: ERROR ALONG PROFILE
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9.5, 4.8),
    )

    baseline_error = (
        baseline_profile
        - true_rockhead
    )

    optimized_error = (
        selected_profile
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
        optimized_error,
        linewidth=2.0,
        label="Optimized integrated error",
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
        "Optimized Rockhead Error Along Profile"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    error_figure_file = (
        figure_output_folder
        / "18_optimized_borehole_error.png"
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
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 74)

    print(
        "Borehole influence-length analysis completed."
    )

    print(
        "\nSelected influence length:"
    )

    print(
        f"{selected_influence_length:.1f} m"
    )

    print(
        "\nOptimized borehole-constrained profile:"
    )

    print(
        f"Mean error = "
        f"{best_row['mean_error_m']:.3f} m"
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
        f"Maximum absolute error = "
        f"{best_row['maximum_absolute_error_m']:.3f} m"
    )

    rmse_improvement_percent = (
        (
            baseline_metrics[
                "rmse_m"
            ]
            - best_row[
                "rmse_m"
            ]
        )
        / baseline_metrics[
            "rmse_m"
        ]
        * 100.0
    )

    mae_improvement_percent = (
        (
            baseline_metrics[
                "mean_absolute_error_m"
            ]
            - best_row[
                "mean_absolute_error_m"
            ]
        )
        / baseline_metrics[
            "mean_absolute_error_m"
        ]
        * 100.0
    )

    print(
        "\nImprovement relative to baseline interpolation:"
    )

    print(
        f"RMSE improvement = "
        f"{rmse_improvement_percent:.2f}%"
    )

    print(
        f"MAE improvement = "
        f"{mae_improvement_percent:.2f}%"
    )

    print(
        f"\nSensitivity table saved to:\n"
        f"{sensitivity_table_file}"
    )

    print(
        f"\nOptimized profile saved to:\n"
        f"{selected_profile_file}"
    )

    print(
        f"\nSensitivity figure saved to:\n"
        f"{sensitivity_figure_file}"
    )

    print(
        f"\nOptimized integration figure saved to:\n"
        f"{optimized_figure_file}"
    )

    print(
        f"\nOptimized error figure saved to:\n"
        f"{error_figure_file}"
    )

    print("=" * 74)


if __name__ == "__main__":
    main()