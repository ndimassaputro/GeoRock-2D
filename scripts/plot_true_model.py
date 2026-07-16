from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# Add the src folder to the Python import path
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from georock2d.model import create_velocity_model


def main():
    x, z, velocity, rockhead_depth = create_velocity_model()

    dx = x[1] - x[0]
    dz = z[1] - z[0]

    x_edges = np.arange(0.0, x[-1] + dx, dx)
    z_edges = np.arange(0.0, z[-1] + dz, dz)

    figure, axis = plt.subplots(figsize=(10, 5))

    image = axis.pcolormesh(
        x_edges,
        z_edges,
        velocity,
        shading="auto",
        cmap="viridis",
    )

    axis.plot(
        x,
        rockhead_depth,
        linewidth=2,
        label="True engineering rockhead",
    )

    axis.set_xlabel("Horizontal distance (m)")
    axis.set_ylabel("Depth below ground surface (m)")
    axis.set_title("GeoRock-2D Synthetic Velocity Model")

    axis.set_xlim(0, 120)
    axis.set_ylim(45, 0)

    axis.legend()

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("P-wave velocity (m/s)")

    output_folder = project_root / "results" / "figures"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "01_true_velocity_model.png"

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print("Synthetic velocity model created successfully.")
    print(f"Velocity model shape: {velocity.shape}")
    print(
        f"Rockhead depth range: "
        f"{rockhead_depth.min():.2f}–"
        f"{rockhead_depth.max():.2f} m"
    )
    print(f"Figure saved to: {output_file}")

    plt.show()
    plt.show(block=False)
plt.pause(3)
plt.close()


if __name__ == "__main__":
    main()