from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


from georock2d.model import create_velocity_model
from georock2d.acquisition import (
    create_acquisition_geometry,
    create_source_receiver_pairs,
)
from georock2d.forward import (
    sample_v_shaped_ray,
    calculate_v_ray_travel_time,
)


def main():
    x, z, velocity, rockhead_depth = create_velocity_model()

    sources, receivers = create_acquisition_geometry()

    pairs = create_source_receiver_pairs(
        sources,
        receivers,
        minimum_offset=20.0,
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

    figure, axis = plt.subplots(
        figsize=(11, 5),
    )

    image = axis.pcolormesh(
        x_edges,
        z_edges,
        velocity,
        shading="auto",
        cmap="viridis",
    )

    selected_pairs = pairs[::15]

    travel_times = []

    for source, receiver in selected_pairs:

        ray_x, ray_z = sample_v_shaped_ray(
            source,
            receiver,
        )

        axis.plot(
            ray_x,
            ray_z,
            linewidth=0.7,
            alpha=0.5,
        )

        travel_time = calculate_v_ray_travel_time(
            source,
            receiver,
            x,
            z,
            velocity,
        )

        travel_times.append(
            travel_time
        )

    axis.scatter(
        sources[:, 0],
        sources[:, 1],
        marker="*",
        s=90,
        label="Sources",
    )

    axis.scatter(
        receivers[:, 0],
        receivers[:, 1],
        marker="v",
        s=30,
        label="Receivers",
    )

    axis.plot(
        x,
        rockhead_depth,
        linewidth=2,
        label="True rockhead",
    )

    axis.set_xlabel(
        "Horizontal distance (m)"
    )

    axis.set_ylabel(
        "Depth below ground surface (m)"
    )

    axis.set_title(
        "GeoRock-2D Conceptual V-Shaped Ray Coverage"
    )

    axis.set_xlim(
        0,
        120,
    )

    axis.set_ylim(
        45,
        0,
    )

    axis.legend(
        loc="lower left"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "P-wave velocity (m/s)"
    )

    output_folder = (
        project_root
        / "results"
        / "figures"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_folder
        / "02_conceptual_ray_paths.png"
    )

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Number of sources: {len(sources)}"
    )

    print(
        f"Number of receivers: {len(receivers)}"
    )

    print(
        f"Total valid pairs: {len(pairs)}"
    )

    print(
        f"Displayed rays: {len(selected_pairs)}"
    )

    print(
        "Travel-time range for displayed rays: "
        f"{min(travel_times):.4f}–"
        f"{max(travel_times):.4f} s"
    )

    print(
        f"Figure saved to: {output_file}"
    )

    plt.show()


if __name__ == "__main__":
    main()