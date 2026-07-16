from pathlib import Path
import sys
import time

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

from georock2d.boreholes import (
    create_synthetic_boreholes,
    build_geophysics_baseline_profile,
    apply_borehole_correction,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def smooth_nan_velocity(
    velocity_model,
    sigma_vertical=1.0,
    sigma_horizontal=1.2,
):
    """
    Smooth a velocity model while retaining unsupported regions as NaN.
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
    velocity_threshold_m_s=1400.0,
    minimum_depth_m=10.0,
    maximum_depth_m=32.0,
    minimum_ray_density=2.0,
):
    """
    Extract partial rockhead using a coverage-aware velocity threshold.
    """

    number_of_columns = velocity_model.shape[1]

    estimated_rockhead = np.full(
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

            estimated_rockhead[
                column_id
            ] = (
                upper_depth
                + fraction
                * (
                    lower_depth
                    - upper_depth
                )
            )

            break

    return estimated_rockhead


def calculate_profile_metrics(
    true_profile,
    estimated_profile,
):
    """
    Calculate full-profile rockhead metrics.
    """

    error = (
        estimated_profile
        - true_profile
    )

    return {
        "mean_error_m": float(
            np.mean(error)
        ),
        "mae_m": float(
            np.mean(
                np.abs(error)
            )
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(
                    error**2
                )
            )
        ),
        "maximum_absolute_error_m": float(
            np.max(
                np.abs(error)
            )
        ),
    }


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """
    Run empirical Monte Carlo uncertainty analysis for GeoRock-2D.
    """

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    number_of_realizations = 50

    master_random_seed = 20260716

    travel_time_noise_std_ms = 1.0

    regularization_lambda = 3.0

    reference_velocity_m_s = 1500.0

    damping_ratio = 0.1

    velocity_threshold_m_s = 1400.0

    minimum_depth_m = 10.0
    maximum_depth_m = 32.0

    minimum_ray_density = 2.0

    borehole_x_positions = np.array(
        [
            18.0,
            58.0,
            103.0,
        ],
        dtype=float,
    )

    borehole_noise_std_m = 0.25

    borehole_influence_length_m = 5.0

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    sensitivity_file = (
        project_root
        / "results"
        / "models"
        / "sensitivity_matrix.npz"
    )

    linear_data_file = (
        project_root
        / "data"
        / "synthetic"
        / "linear_travel_times.csv"
    )

    if not sensitivity_file.exists():
        raise FileNotFoundError(
            f"Sensitivity matrix was not found:\n"
            f"{sensitivity_file}"
        )

    if not linear_data_file.exists():
        raise FileNotFoundError(
            f"Linear travel-time data were not found:\n"
            f"{linear_data_file}"
        )

    # --------------------------------------------------------
    # LOAD TRUE MODEL
    # --------------------------------------------------------

    x, z, true_velocity, true_rockhead = (
        create_velocity_model()
    )

    # --------------------------------------------------------
    # LOAD SENSITIVITY MATRIX
    # --------------------------------------------------------

    sensitivity_data = np.load(
        sensitivity_file
    )

    sensitivity_matrix = sensitivity_data[
        "sensitivity_matrix"
    ]

    ray_density = sensitivity_data[
        "ray_density"
    ]

    active_mask = ray_density > 0.0

    (
        smoothness_matrix,
        active_flat_indices,
    ) = build_first_order_smoothness_matrix(
        active_mask=active_mask,
    )

    # --------------------------------------------------------
    # LOAD NOISELESS TRAVEL TIMES
    # --------------------------------------------------------

    linear_dataset = pd.read_csv(
        linear_data_file
    )

    noiseless_travel_times_ms = (
        linear_dataset[
            "linear_travel_time_ms"
        ].to_numpy(
            dtype=float
        )
    )

    if (
        len(noiseless_travel_times_ms)
        != sensitivity_matrix.shape[0]
    ):
        raise ValueError(
            "Travel-time vector length does not match "
            "the sensitivity matrix."
        )

    # --------------------------------------------------------
    # RANDOM NUMBER GENERATOR
    # --------------------------------------------------------

    master_rng = np.random.default_rng(
        master_random_seed
    )

    realization_seeds = master_rng.integers(
        low=1,
        high=2_147_483_647,
        size=number_of_realizations,
    )

    # --------------------------------------------------------
    # OUTPUT CONTAINERS
    # --------------------------------------------------------

    integrated_profiles = np.full(
        (
            number_of_realizations,
            len(x),
        ),
        np.nan,
        dtype=float,
    )

    geophysical_profiles = np.full(
        (
            number_of_realizations,
            len(x),
        ),
        np.nan,
        dtype=float,
    )

    realization_records = []

    print("=" * 76)
    print("GeoRock-2D Monte Carlo Uncertainty Analysis")
    print("=" * 76)

    print(
        f"\nNumber of realizations: "
        f"{number_of_realizations}"
    )

    print(
        f"Travel-time noise standard deviation: "
        f"{travel_time_noise_std_ms:.3f} ms"
    )

    print(
        f"Regularization lambda: "
        f"{regularization_lambda:g}"
    )

    print(
        f"Rockhead velocity threshold: "
        f"{velocity_threshold_m_s:.0f} m/s"
    )

    print(
        f"Borehole influence length: "
        f"{borehole_influence_length_m:.1f} m"
    )

    # --------------------------------------------------------
    # MONTE CARLO LOOP
    # --------------------------------------------------------

    for realization_id, realization_seed in enumerate(
        realization_seeds,
        start=1,
    ):

        rng = np.random.default_rng(
            int(
                realization_seed
            )
        )

        travel_time_noise_ms = rng.normal(
            loc=0.0,
            scale=travel_time_noise_std_ms,
            size=len(
                noiseless_travel_times_ms
            ),
        )

        observed_travel_times_ms = (
            noiseless_travel_times_ms
            + travel_time_noise_ms
        )

        (
            inversion_result,
            estimated_slowness_ms_m,
            predicted_travel_times_ms,
        ) = invert_active_slowness(
            sensitivity_matrix=(
                sensitivity_matrix
            ),
            observed_travel_times_ms=(
                observed_travel_times_ms
            ),
            smoothness_matrix=(
                smoothness_matrix
            ),
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

        smoothed_velocity = smooth_nan_velocity(
            velocity_model=estimated_velocity,
            sigma_vertical=1.0,
            sigma_horizontal=1.2,
        )

        geophysical_rockhead = (
            extract_threshold_rockhead(
                velocity_model=(
                    smoothed_velocity
                ),
                ray_density=ray_density,
                z=z,
                velocity_threshold_m_s=(
                    velocity_threshold_m_s
                ),
                minimum_depth_m=(
                    minimum_depth_m
                ),
                maximum_depth_m=(
                    maximum_depth_m
                ),
                minimum_ray_density=(
                    minimum_ray_density
                ),
            )
        )

        geophysical_profiles[
            realization_id - 1,
            :
        ] = geophysical_rockhead

        # Each realization also receives a new borehole-noise sample.
        borehole_table = (
            create_synthetic_boreholes(
                x=x,
                true_rockhead=(
                    true_rockhead
                ),
                borehole_x_positions=(
                    borehole_x_positions
                ),
                noise_std_m=(
                    borehole_noise_std_m
                ),
                random_seed=int(
                    realization_seed
                ),
            )
        )

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
                    borehole_influence_length_m
                ),
            )
        )

        integrated_profiles[
            realization_id - 1,
            :
        ] = integrated_profile

        travel_time_rmse_ms = float(
            np.sqrt(
                np.mean(
                    (
                        observed_travel_times_ms
                        - predicted_travel_times_ms
                    )
                    ** 2
                )
            )
        )

        profile_metrics = (
            calculate_profile_metrics(
                true_profile=true_rockhead,
                estimated_profile=(
                    integrated_profile
                ),
            )
        )

        valid_geophysical_percentage = (
            np.count_nonzero(
                np.isfinite(
                    geophysical_rockhead
                )
            )
            / len(x)
            * 100.0
        )

        realization_records.append(
            {
                "realization_id": (
                    realization_id
                ),
                "random_seed": int(
                    realization_seed
                ),
                "inversion_success": bool(
                    inversion_result.success
                ),
                "travel_time_rmse_ms": (
                    travel_time_rmse_ms
                ),
                "geophysical_coverage_percent": (
                    valid_geophysical_percentage
                ),
                **profile_metrics,
            }
        )

        if (
            realization_id == 1
            or realization_id % 5 == 0
            or realization_id
            == number_of_realizations
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Completed realization "
                f"{realization_id:02d}/"
                f"{number_of_realizations} "
                f"| RMSE = "
                f"{profile_metrics['rmse_m']:.3f} m "
                f"| elapsed = "
                f"{elapsed:.1f} s"
            )

    # --------------------------------------------------------
    # MONTE CARLO STATISTICS
    # --------------------------------------------------------

    mean_profile = np.mean(
        integrated_profiles,
        axis=0,
    )

    median_profile = np.median(
        integrated_profiles,
        axis=0,
    )

    standard_deviation_profile = np.std(
        integrated_profiles,
        axis=0,
        ddof=1,
    )

    lower_95_profile = np.percentile(
        integrated_profiles,
        2.5,
        axis=0,
    )

    upper_95_profile = np.percentile(
        integrated_profiles,
        97.5,
        axis=0,
    )

    uncertainty_width = (
        upper_95_profile
        - lower_95_profile
    )

    true_inside_interval = (
        (
            true_rockhead
            >= lower_95_profile
        )
        & (
            true_rockhead
            <= upper_95_profile
        )
    )

    coverage_probability_percent = (
        np.mean(
            true_inside_interval
        )
        * 100.0
    )

    mean_profile_metrics = (
        calculate_profile_metrics(
            true_profile=true_rockhead,
            estimated_profile=(
                mean_profile
            ),
        )
    )

    realization_table = pd.DataFrame(
        realization_records
    )

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

    profile_table = pd.DataFrame(
        {
            "x_m": x,
            "true_rockhead_depth_m": (
                true_rockhead
            ),
            "mean_integrated_depth_m": (
                mean_profile
            ),
            "median_integrated_depth_m": (
                median_profile
            ),
            "standard_deviation_m": (
                standard_deviation_profile
            ),
            "lower_95_depth_m": (
                lower_95_profile
            ),
            "upper_95_depth_m": (
                upper_95_profile
            ),
            "uncertainty_width_m": (
                uncertainty_width
            ),
            "true_inside_95_interval": (
                true_inside_interval
            ),
        }
    )

    profile_table_file = (
        table_output_folder
        / "monte_carlo_profile_statistics.csv"
    )

    profile_table.to_csv(
        profile_table_file,
        index=False,
    )

    realization_table_file = (
        table_output_folder
        / "monte_carlo_realization_metrics.csv"
    )

    realization_table.to_csv(
        realization_table_file,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE NUMERICAL ARRAYS
    # --------------------------------------------------------

    model_output_folder = (
        project_root
        / "results"
        / "models"
    )

    model_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    monte_carlo_model_file = (
        model_output_folder
        / "monte_carlo_rockhead_profiles.npz"
    )

    np.savez_compressed(
        monte_carlo_model_file,
        integrated_profiles=(
            integrated_profiles
        ),
        geophysical_profiles=(
            geophysical_profiles
        ),
        mean_profile=mean_profile,
        median_profile=median_profile,
        standard_deviation_profile=(
            standard_deviation_profile
        ),
        lower_95_profile=(
            lower_95_profile
        ),
        upper_95_profile=(
            upper_95_profile
        ),
        x=x,
        true_rockhead=(
            true_rockhead
        ),
    )

    # --------------------------------------------------------
    # FIGURE 1: UNCERTAINTY BAND
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(10.5, 5.5),
    )

    axis.fill_between(
        x,
        lower_95_profile,
        upper_95_profile,
        alpha=0.3,
        label="95% empirical interval",
    )

    axis.plot(
        x,
        true_rockhead,
        linewidth=2.5,
        label="True rockhead",
    )

    axis.plot(
        x,
        mean_profile,
        linewidth=2.2,
        label="Monte Carlo mean",
    )

    axis.plot(
        x,
        median_profile,
        linestyle="--",
        linewidth=1.8,
        label="Monte Carlo median",
    )

    axis.scatter(
        borehole_x_positions,
        np.interp(
            borehole_x_positions,
            x,
            true_rockhead,
        ),
        marker="v",
        s=80,
        label="Borehole locations",
        zorder=5,
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Rockhead depth (m)"
    )

    axis.set_title(
        "GeoRock-2D Monte Carlo Rockhead Uncertainty"
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

    uncertainty_figure_file = (
        figure_output_folder
        / "19_monte_carlo_uncertainty_band.png"
    )

    figure.savefig(
        uncertainty_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 2: UNCERTAINTY WIDTH
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(9.5, 4.8),
    )

    axis.plot(
        x,
        uncertainty_width,
        linewidth=2.2,
    )

    axis.scatter(
        borehole_x_positions,
        np.interp(
            borehole_x_positions,
            x,
            uncertainty_width,
        ),
        marker="v",
        s=75,
        label="Borehole locations",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "95% interval width (m)"
    )

    axis.set_title(
        "Rockhead Uncertainty Width Along Profile"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    width_figure_file = (
        figure_output_folder
        / "20_uncertainty_width.png"
    )

    figure.savefig(
        width_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 3: RMSE DISTRIBUTION
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(8.5, 5.2),
    )

    axis.hist(
        realization_table[
            "rmse_m"
        ],
        bins=10,
        edgecolor="black",
        alpha=0.75,
    )

    axis.axvline(
        realization_table[
            "rmse_m"
        ].mean(),
        linestyle="--",
        linewidth=2.0,
        label=(
            "Mean realization RMSE"
        ),
    )

    axis.set_xlabel(
        "Full-profile rockhead RMSE (m)"
    )

    axis.set_ylabel(
        "Number of realizations"
    )

    axis.set_title(
        "Monte Carlo Distribution of Rockhead RMSE"
    )

    axis.grid(
        visible=True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    rmse_figure_file = (
        figure_output_folder
        / "21_monte_carlo_rmse_distribution.png"
    )

    figure.savefig(
        rmse_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # PRINT FINAL SUMMARY
    # --------------------------------------------------------

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    print("\n" + "=" * 76)

    print(
        "Monte Carlo analysis completed."
    )

    print(
        f"\nTotal runtime: "
        f"{elapsed_time:.1f} s"
    )

    print(
        "\nMean integrated profile:"
    )

    print(
        f"Mean error = "
        f"{mean_profile_metrics['mean_error_m']:.3f} m"
    )

    print(
        f"MAE = "
        f"{mean_profile_metrics['mae_m']:.3f} m"
    )

    print(
        f"RMSE = "
        f"{mean_profile_metrics['rmse_m']:.3f} m"
    )

    print(
        f"Maximum absolute error = "
        f"{mean_profile_metrics['maximum_absolute_error_m']:.3f} m"
    )

    print(
        "\nAcross individual realizations:"
    )

    print(
        f"Mean RMSE = "
        f"{realization_table['rmse_m'].mean():.3f} m"
    )

    print(
        f"RMSE standard deviation = "
        f"{realization_table['rmse_m'].std(ddof=1):.3f} m"
    )

    print(
        f"Minimum RMSE = "
        f"{realization_table['rmse_m'].min():.3f} m"
    )

    print(
        f"Maximum RMSE = "
        f"{realization_table['rmse_m'].max():.3f} m"
    )

    print(
        "\nUncertainty statistics:"
    )

    print(
        f"Mean 95% interval width = "
        f"{uncertainty_width.mean():.3f} m"
    )

    print(
        f"Maximum 95% interval width = "
        f"{uncertainty_width.max():.3f} m"
    )

    print(
        "Empirical coverage probability = "
        f"{coverage_probability_percent:.2f}%"
    )

    print(
        f"\nProfile statistics saved to:\n"
        f"{profile_table_file}"
    )

    print(
        f"\nRealization metrics saved to:\n"
        f"{realization_table_file}"
    )

    print(
        f"\nMonte Carlo arrays saved to:\n"
        f"{monte_carlo_model_file}"
    )

    print(
        f"\nUncertainty figure saved to:\n"
        f"{uncertainty_figure_file}"
    )

    print(
        f"\nUncertainty-width figure saved to:\n"
        f"{width_figure_file}"
    )

    print(
        f"\nRMSE-distribution figure saved to:\n"
        f"{rmse_figure_file}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()