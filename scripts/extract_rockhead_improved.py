from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
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
    Apply normalized Gaussian smoothing while preserving NaN regions.

    Parameters
    ----------
    velocity_model : numpy.ndarray
        Two-dimensional estimated velocity model containing NaN values
        outside the ray-covered region.

    sigma_vertical : float
        Vertical Gaussian smoothing scale in grid cells.

    sigma_horizontal : float
        Horizontal Gaussian smoothing scale in grid cells.

    Returns
    -------
    smoothed_velocity : numpy.ndarray
        Smoothed velocity model with unsupported areas retained as NaN.
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

    supported = smoothed_weights > 1.0e-8

    smoothed_velocity[supported] = (
        smoothed_values[supported]
        / smoothed_weights[supported]
    )

    return smoothed_velocity


def extract_threshold_rockhead(
    velocity_model,
    ray_density,
    z,
    velocity_threshold_m_s=1700.0,
    minimum_depth_m=10.0,
    maximum_depth_m=32.0,
    minimum_ray_density=2.0,
):
    """
    Extract rockhead from the first supported upward velocity-threshold
    crossing in each model column.

    A valid crossing must:
    1. occur inside the selected depth interval;
    2. connect two cells with adequate ray coverage;
    3. cross from velocity below the threshold to velocity at or above it.

    Parameters
    ----------
    velocity_model : numpy.ndarray
        Smoothed estimated velocity model.

    ray_density : numpy.ndarray
        Cumulative ray-path length in every model cell.

    z : numpy.ndarray
        Cell-centre depth coordinates.

    velocity_threshold_m_s : float
        Synthetic interface velocity threshold.

    minimum_depth_m : float
        Shallowest accepted rockhead depth.

    maximum_depth_m : float
        Deepest accepted rockhead depth.

    minimum_ray_density : float
        Minimum cumulative ray-path length required in both cells.

    Returns
    -------
    estimated_rockhead_m : numpy.ndarray
        Estimated rockhead depth for each horizontal column.
    """

    number_of_columns = velocity_model.shape[1]

    estimated_rockhead_m = np.full(
        number_of_columns,
        np.nan,
        dtype=float,
    )

    for column_id in range(number_of_columns):

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

            upper_depth = z[upper_index]
            lower_depth = z[lower_index]

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
                np.isfinite(upper_velocity)
                and np.isfinite(lower_velocity)
            ):
                continue

            if (
                density_column[upper_index]
                < minimum_ray_density
                or density_column[lower_index]
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

            interpolation_fraction = (
                velocity_threshold_m_s
                - upper_velocity
            ) / velocity_difference

            estimated_depth = (
                upper_depth
                + interpolation_fraction
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
    Calculate interface-recovery metrics at valid positions.
    """

    valid_mask = np.isfinite(
        estimated_rockhead
    )

    number_of_valid_positions = np.count_nonzero(
        valid_mask
    )

    if number_of_valid_positions == 0:
        raise RuntimeError(
            "No valid rockhead estimates were generated."
        )

    errors = (
        estimated_rockhead[valid_mask]
        - true_rockhead[valid_mask]
    )

    return {
        "valid_mask": valid_mask,
        "number_of_valid_positions": (
            number_of_valid_positions
        ),
        "valid_percentage": (
            number_of_valid_positions
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
    Run coverage-aware velocity-threshold rockhead extraction.
    """

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    selected_lambda = 3.0

    velocity_threshold_m_s = 1700.0

    minimum_depth_m = 10.0
    maximum_depth_m = 32.0

    minimum_ray_density = 2.0

    reference_velocity_m_s = 1500.0

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
            f"Sensitivity file not found:\n"
            f"{sensitivity_file}"
        )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    inversion_data = np.load(
        inversion_file
    )

    estimated_velocity = inversion_data[
        "estimated_velocity_m_s"
    ]

    active_mask = inversion_data[
        "active_mask"
    ].astype(bool)

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
    # SMOOTH ESTIMATED VELOCITY
    # --------------------------------------------------------

    smoothed_velocity = smooth_nan_velocity(
        velocity_model=estimated_velocity,
        sigma_vertical=1.0,
        sigma_horizontal=1.2,
    )

    # --------------------------------------------------------
    # EXTRACT ROCKHEAD
    # --------------------------------------------------------

    estimated_rockhead = extract_threshold_rockhead(
        velocity_model=smoothed_velocity,
        ray_density=ray_density,
        z=z,
        velocity_threshold_m_s=(
            velocity_threshold_m_s
        ),
        minimum_depth_m=minimum_depth_m,
        maximum_depth_m=maximum_depth_m,
        minimum_ray_density=(
            minimum_ray_density
        ),
    )

    metrics = calculate_metrics(
        true_rockhead=true_rockhead,
        estimated_rockhead=estimated_rockhead,
    )

    valid_mask = metrics[
        "valid_mask"
    ]

    # --------------------------------------------------------
    # SAVE TABLE
    # --------------------------------------------------------

    result_table = pd.DataFrame(
        {
            "x_m": x,
            "true_rockhead_depth_m": (
                true_rockhead
            ),
            "estimated_rockhead_depth_m": (
                estimated_rockhead
            ),
            "rockhead_error_m": (
                estimated_rockhead
                - true_rockhead
            ),
            "valid_estimate": (
                valid_mask
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

    table_output_file = (
        table_output_folder
        / "rockhead_threshold_lambda_3.csv"
    )

    result_table.to_csv(
        table_output_file,
        index=False,
    )

    # --------------------------------------------------------
    # PREPARE FULL-DOMAIN DISPLAY
    # --------------------------------------------------------

    display_velocity = np.where(
        active_mask,
        smoothed_velocity,
        reference_velocity_m_s,
    )

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

    # --------------------------------------------------------
    # FIGURE 1: FULL-DOMAIN MAP
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(10.5, 5.5),
    )

    image = axis.pcolormesh(
        x_edges,
        z_edges,
        display_velocity,
        shading="auto",
        cmap="viridis",
        vmin=500.0,
        vmax=3000.0,
    )

    inactive_overlay = np.where(
        active_mask,
        np.nan,
        1.0,
    )

    grey_cmap = ListedColormap(
        ["lightgray"]
    )

    axis.pcolormesh(
        x_edges,
        z_edges,
        inactive_overlay,
        shading="auto",
        cmap=grey_cmap,
        alpha=0.75,
        vmin=0.0,
        vmax=1.0,
    )

    axis.plot(
        x,
        true_rockhead,
        linewidth=2.0,
        label="True rockhead",
    )

    axis.plot(
        x[valid_mask],
        estimated_rockhead[valid_mask],
        linestyle="--",
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        label="Coverage-aware threshold estimate",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Depth below ground surface (m)"
    )

    axis.set_title(
        "GeoRock-2D Coverage-Aware Rockhead Interpretation"
    )

    axis.set_xlim(
        0.0,
        120.0,
    )

    axis.set_ylim(
        45.0,
        0.0,
    )

    axis.legend(
        loc="lower left"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Estimated P-wave velocity (m/s)"
    )

    axis.text(
        0.99,
        0.02,
        "Grey: unconstrained by ray coverage",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
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

    map_output_file = (
        figure_output_folder
        / "10_improved_rockhead_interpretation.png"
    )

    figure.savefig(
        map_output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 2: INTERFACE COMPARISON
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
        x[valid_mask],
        estimated_rockhead[valid_mask],
        linestyle="--",
        marker="o",
        markersize=4.0,
        linewidth=2.0,
        label="Coverage-aware estimate",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Rockhead depth (m)"
    )

    axis.set_title(
        "Improved True versus Estimated Rockhead"
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

    comparison_output_file = (
        figure_output_folder
        / "11_improved_true_vs_estimated_rockhead.png"
    )

    figure.savefig(
        comparison_output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("=" * 72)
    print("GeoRock-2D Improved Rockhead Extraction")
    print("=" * 72)

    print(
        f"\nSelected lambda: "
        f"{selected_lambda:g}"
    )

    print(
        f"Velocity threshold: "
        f"{velocity_threshold_m_s:.1f} m/s"
    )

    print(
        "Accepted depth interval: "
        f"{minimum_depth_m:.1f}–"
        f"{maximum_depth_m:.1f} m"
    )

    print(
        f"Minimum ray density: "
        f"{minimum_ray_density:.2f} m"
    )

    print(
        f"\nValid horizontal positions: "
        f"{metrics['number_of_valid_positions']} "
        f"/ {len(x)}"
    )

    print(
        "Valid-profile coverage: "
        f"{metrics['valid_percentage']:.2f}%"
    )

    print(
        "\nMean rockhead error: "
        f"{metrics['mean_error_m']:.3f} m"
    )

    print(
        "Mean absolute rockhead error: "
        f"{metrics['mean_absolute_error_m']:.3f} m"
    )

    print(
        "Rockhead RMSE: "
        f"{metrics['rmse_m']:.3f} m"
    )

    print(
        "Maximum absolute rockhead error: "
        f"{metrics['maximum_absolute_error_m']:.3f} m"
    )

    print(
        f"\nResult table saved to:\n"
        f"{table_output_file}"
    )

    print(
        f"\nInterpretation map saved to:\n"
        f"{map_output_file}"
    )

    print(
        f"\nComparison figure saved to:\n"
        f"{comparison_output_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()