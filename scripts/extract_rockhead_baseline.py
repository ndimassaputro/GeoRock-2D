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
# PROJECT IMPORT
# ============================================================

from georock2d.model import create_velocity_model


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_rockhead_from_velocity_gradient(
    velocity_model,
    z,
    minimum_depth_m=5.0,
):
    """
    Estimate rockhead using the maximum positive vertical
    velocity gradient in each horizontal model column.

    Parameters
    ----------
    velocity_model : numpy.ndarray
        Two-dimensional estimated velocity model.

    z : numpy.ndarray
        Cell-centre depth coordinates in metres.

    minimum_depth_m : float
        Minimum depth considered during interface extraction.

    Returns
    -------
    estimated_rockhead_m : numpy.ndarray
        Estimated rockhead depth for every horizontal location.
        Columns without sufficient constrained cells are assigned NaN.
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
            column_id,
        ]

        finite_mask = np.isfinite(
            velocity_column
        )

        valid_indices = np.flatnonzero(
            finite_mask
            & (z >= minimum_depth_m)
        )

        if valid_indices.size < 2:
            continue

        # Keep only neighbouring finite cells
        candidate_gradients = []
        candidate_depths = []

        for upper_index in valid_indices[:-1]:

            lower_index = upper_index + 1

            if lower_index >= len(z):
                continue

            if not (
                np.isfinite(
                    velocity_column[upper_index]
                )
                and np.isfinite(
                    velocity_column[lower_index]
                )
            ):
                continue

            depth_difference = (
                z[lower_index]
                - z[upper_index]
            )

            velocity_difference = (
                velocity_column[lower_index]
                - velocity_column[upper_index]
            )

            vertical_gradient = (
                velocity_difference
                / depth_difference
            )

            candidate_gradients.append(
                vertical_gradient
            )

            candidate_depths.append(
                (
                    z[upper_index]
                    + z[lower_index]
                )
                / 2.0
            )

        if not candidate_gradients:
            continue

        candidate_gradients = np.asarray(
            candidate_gradients,
            dtype=float,
        )

        candidate_depths = np.asarray(
            candidate_depths,
            dtype=float,
        )

        maximum_gradient_index = int(
            np.argmax(
                candidate_gradients
            )
        )

        maximum_gradient = candidate_gradients[
            maximum_gradient_index
        ]

        # A positive velocity increase with depth is expected
        if maximum_gradient > 0:
            estimated_rockhead_m[
                column_id
            ] = candidate_depths[
                maximum_gradient_index
            ]

    return estimated_rockhead_m


def calculate_rockhead_metrics(
    true_rockhead,
    estimated_rockhead,
):
    """
    Calculate rockhead error metrics for valid estimated positions.
    """

    valid_mask = np.isfinite(
        estimated_rockhead
    )

    if np.count_nonzero(valid_mask) == 0:
        raise RuntimeError(
            "No valid rockhead estimates were produced."
        )

    errors = (
        estimated_rockhead[valid_mask]
        - true_rockhead[valid_mask]
    )

    mean_absolute_error = float(
        np.mean(
            np.abs(
                errors
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors**2
            )
        )
    )

    maximum_absolute_error = float(
        np.max(
            np.abs(
                errors
            )
        )
    )

    return {
        "valid_mask": valid_mask,
        "errors_m": errors,
        "mean_absolute_error_m": mean_absolute_error,
        "rmse_m": rmse,
        "maximum_absolute_error_m": maximum_absolute_error,
        "valid_percentage": (
            np.count_nonzero(valid_mask)
            / len(valid_mask)
            * 100.0
        ),
    }


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """
    Extract and evaluate the engineering rockhead from the
    regularized inversion selected at lambda = 3.
    """

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    selected_lambda = 3.0
    reference_velocity_m_s = 1500.0

    # --------------------------------------------------------
    # INPUT MODEL
    # --------------------------------------------------------

    inversion_file = (
        project_root
        / "results"
        / "models"
        / "regularization_tests"
        / "inversion_lambda_3.npz"
    )

    if not inversion_file.exists():
        raise FileNotFoundError(
            f"Selected inversion model was not found:\n"
            f"{inversion_file}"
        )

    inversion_data = np.load(
        inversion_file
    )

    estimated_velocity = inversion_data[
        "estimated_velocity_m_s"
    ]

    active_mask = inversion_data[
        "active_mask"
    ].astype(bool)

    # --------------------------------------------------------
    # TRUE SYNTHETIC MODEL
    # --------------------------------------------------------

    x, z, true_velocity, true_rockhead = (
        create_velocity_model()
    )

    # --------------------------------------------------------
    # EXTRACT ESTIMATED ROCKHEAD
    # --------------------------------------------------------

    estimated_rockhead = (
        extract_rockhead_from_velocity_gradient(
            velocity_model=estimated_velocity,
            z=z,
            minimum_depth_m=5.0,
        )
    )

    metrics = calculate_rockhead_metrics(
        true_rockhead=true_rockhead,
        estimated_rockhead=estimated_rockhead,
    )

    valid_mask = metrics[
        "valid_mask"
    ]

    # --------------------------------------------------------
    # SAVE ROCKHEAD TABLE
    # --------------------------------------------------------

    rockhead_table = pd.DataFrame(
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
        / "rockhead_extraction_lambda_3.csv"
    )

    rockhead_table.to_csv(
        table_output_file,
        index=False,
    )

    # --------------------------------------------------------
    # CREATE FULL-DOMAIN DISPLAY MODEL
    # --------------------------------------------------------

    display_velocity = np.where(
        active_mask,
        estimated_velocity,
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
    # FIGURE 1: FULL-DOMAIN INTERPRETATION
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

    # Mark inactive cells using semi-transparent grey
    inactive_overlay = np.where(
        active_mask,
        np.nan,
        1.0,
    )

    axis.pcolormesh(
        x_edges,
        z_edges,
        inactive_overlay,
        shading="auto",
        cmap="Greys",
        alpha=0.35,
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
        linewidth=2.0,
        label="Gradient-derived rockhead",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Depth below ground surface (m)"
    )

    axis.set_title(
        "GeoRock-2D Full-Domain Rockhead Interpretation "
        "(λ = 3)"
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
        "Grey overlay: unconstrained by ray coverage",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
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

    full_domain_figure_file = (
        figure_output_folder
        / "08_full_domain_rockhead_interpretation.png"
    )

    figure.savefig(
        full_domain_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # FIGURE 2: ROCKHEAD COMPARISON
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
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Estimated rockhead",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Rockhead depth (m)"
    )

    axis.set_title(
        "True versus Estimated Engineering Rockhead"
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

    comparison_figure_file = (
        figure_output_folder
        / "09_true_vs_estimated_rockhead.png"
    )

    figure.savefig(
        comparison_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("=" * 72)
    print("GeoRock-2D Engineering Rockhead Extraction")
    print("=" * 72)

    print(
        f"\nSelected regularization lambda: "
        f"{selected_lambda:g}"
    )

    print(
        "Extraction criterion: "
        "maximum positive vertical velocity gradient"
    )

    print(
        f"\nValid horizontal positions: "
        f"{np.count_nonzero(valid_mask)} / {len(x)}"
    )

    print(
        "Valid-profile coverage: "
        f"{metrics['valid_percentage']:.2f}%"
    )

    print(
        "\nRockhead mean absolute error: "
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
        f"\nRockhead table saved to:\n"
        f"{table_output_file}"
    )

    print(
        f"\nFull-domain figure saved to:\n"
        f"{full_domain_figure_file}"
    )

    print(
        f"\nComparison figure saved to:\n"
        f"{comparison_figure_file}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()