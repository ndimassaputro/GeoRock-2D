import numpy as np


def create_acquisition_geometry():
    """
    Create synthetic surface source and receiver locations.

    Returns
    -------
    sources : numpy.ndarray
        Source coordinates with shape (n_sources, 2).
        Columns represent x and z coordinates in metres.

    receivers : numpy.ndarray
        Receiver coordinates with shape (n_receivers, 2).
    """

    source_x = np.array(
        [5.0, 25.0, 45.0, 65.0, 85.0, 105.0, 115.0]
    )

    receiver_x = np.arange(
        2.0,
        120.0,
        4.0,
    )

    source_z = np.zeros_like(source_x)
    receiver_z = np.zeros_like(receiver_x)

    sources = np.column_stack(
        (
            source_x,
            source_z,
        )
    )

    receivers = np.column_stack(
        (
            receiver_x,
            receiver_z,
        )
    )

    return sources, receivers


def create_source_receiver_pairs(
    sources,
    receivers,
    minimum_offset=4.0,
):
    """
    Create source-receiver pairs.

    Parameters
    ----------
    sources : numpy.ndarray
        Source coordinates.

    receivers : numpy.ndarray
        Receiver coordinates.

    minimum_offset : float
        Minimum horizontal source-receiver distance in metres.

    Returns
    -------
    pairs : list
        List of source-receiver coordinate pairs.
    """

    pairs = []

    for source in sources:
        for receiver in receivers:

            offset = abs(
                receiver[0] - source[0]
            )

            if offset >= minimum_offset:
                pairs.append(
                    (
                        source.copy(),
                        receiver.copy(),
                    )
                )

    return pairs