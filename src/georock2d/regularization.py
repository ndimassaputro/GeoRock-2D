import numpy as np
from scipy import sparse


def build_first_order_smoothness_matrix(
    active_mask,
):
    """
    Build a first-order spatial smoothness matrix.

    The matrix penalizes slowness differences between horizontally
    and vertically adjacent active model cells.

    Parameters
    ----------
    active_mask : numpy.ndarray
        Two-dimensional Boolean array. True indicates a model cell
        crossed by at least one ray.

    Returns
    -------
    smoothness_matrix : scipy.sparse.csr_matrix
        Sparse first-order difference matrix.

    active_flat_indices : numpy.ndarray
        Flat indices of active cells in the full model.
    """

    if active_mask.ndim != 2:
        raise ValueError(
            "active_mask must be a two-dimensional array."
        )

    active_flat_indices = np.flatnonzero(
        active_mask.ravel()
    )

    if active_flat_indices.size == 0:
        raise ValueError(
            "No active cells were found."
        )

    # Map full-model flat indices to active-model indices
    full_to_active = {
        int(full_index): active_index
        for active_index, full_index
        in enumerate(active_flat_indices)
    }

    number_of_rows, number_of_columns = active_mask.shape

    matrix_rows = []
    matrix_columns = []
    matrix_values = []

    equation_id = 0

    for row in range(number_of_rows):
        for column in range(number_of_columns):

            if not active_mask[row, column]:
                continue

            current_full_index = np.ravel_multi_index(
                (row, column),
                active_mask.shape,
            )

            current_active_index = full_to_active[
                int(current_full_index)
            ]

            # Horizontal neighbour
            if (
                column + 1 < number_of_columns
                and active_mask[row, column + 1]
            ):
                neighbour_full_index = np.ravel_multi_index(
                    (row, column + 1),
                    active_mask.shape,
                )

                neighbour_active_index = full_to_active[
                    int(neighbour_full_index)
                ]

                matrix_rows.extend(
                    [equation_id, equation_id]
                )

                matrix_columns.extend(
                    [
                        current_active_index,
                        neighbour_active_index,
                    ]
                )

                matrix_values.extend(
                    [1.0, -1.0]
                )

                equation_id += 1

            # Vertical neighbour
            if (
                row + 1 < number_of_rows
                and active_mask[row + 1, column]
            ):
                neighbour_full_index = np.ravel_multi_index(
                    (row + 1, column),
                    active_mask.shape,
                )

                neighbour_active_index = full_to_active[
                    int(neighbour_full_index)
                ]

                matrix_rows.extend(
                    [equation_id, equation_id]
                )

                matrix_columns.extend(
                    [
                        current_active_index,
                        neighbour_active_index,
                    ]
                )

                matrix_values.extend(
                    [1.0, -1.0]
                )

                equation_id += 1

    smoothness_matrix = sparse.coo_matrix(
        (
            matrix_values,
            (
                matrix_rows,
                matrix_columns,
            ),
        ),
        shape=(
            equation_id,
            active_flat_indices.size,
        ),
        dtype=float,
    ).tocsr()

    return (
        smoothness_matrix,
        active_flat_indices,
    )