import numpy as np
import torch
import math


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")
else:
    device = torch.device("cpu")
    print("MPS not available -- using CPU")


def psk_constellation(M):
    k = np.arange(M)
    const = np.exp(1j * 2 * np.pi * k / M)
    return const

def qam_constellation(M):
    m = int(np.sqrt(M))
    a = np.arange(-(m-1), m, 2)
    xx, yy = np.meshgrid(a, a)
    const = xx + 1j * yy
    const /= np.sqrt(np.mean(np.abs(const)**2))
    return const.flatten()

def qam_cross_constellation(M, normalize=True):
    """
    Generate a standard Cross-M-QAM constellation.

    Supported family:
        M = 32, 128, 512, 2048, ...
          = 32 * 4^k

    Returns
    -------
    pts : ndarray of complex
        Constellation points.
    """

    N = int(round(np.sqrt(9 * M / 8)))

    if N * N * 8 != 9 * M:
        raise ValueError(
            f"M={M} is not a member of the standard cross-QAM family"
        )

    # Odd PAM levels
    levels = np.arange(-(N - 1), N, 2)

    # Number of outer levels removed on each side
    corner_levels = N // 6

    # Threshold beyond which points belong to a corner block
    threshold = levels[-corner_levels]

    pts = []

    for x in levels:
        for y in levels:

            # Remove corner blocks
            if abs(x) >= threshold and abs(y) >= threshold:
                continue

            pts.append(x + 1j * y)

    pts = np.array(pts)

    if len(pts) != M:
        raise RuntimeError(
            f"Generated {len(pts)} points instead of {M}"
        )

    if normalize:
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))

    return pts



def constellation_to_mps(constellation):
    return (
        torch.tensor(
            np.real(constellation),
            dtype=torch.float32,
            device=device
        ),
        torch.tensor(
            np.imag(constellation),
            dtype=torch.float32,
            device=device
        )
    )