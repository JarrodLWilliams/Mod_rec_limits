# functionality for estimating the saddle point, computing the Chernoff information
# and required cumulants.

import numpy as np
import torch
import math


def log_likelihood_mps(x_re, x_im, constellation_re, constellation_im, sigma2):
    """
    Log likelihood of complex observations x = x_re + i x_im
    under a uniform Gaussian mixture whose constellation points
    are constellation_re + i constellation_im.

    p(x) = (1/M) sum_m [1/(pi sigma2)]
                     exp(-|x-a_m|^2 / sigma2)

    Everything is performed on the selected PyTorch device.
    """

    # x_re: [N]
    # constellation_re: [M]

    d_re = x_re[:, None] - constellation_re[None, :]
    d_im = x_im[:, None] - constellation_im[None, :]

    d2 = d_re ** 2 + d_im ** 2

    log_terms = -d2 / sigma2

    return (
            torch.logsumexp(log_terms, dim=1)
            - math.log(len(constellation_re))
            - math.log(math.pi * sigma2)
    )

@torch.no_grad()
def generate_batch_mps(
    constellation0_re,
    constellation0_im,
    constellation1_re,
    constellation1_im,
    sigma2,
    N
):
    """
    Generate N samples under constellation0 and return

        L = log p1(x) - log p0(x)

    entirely on the GPU.
    """

    M0 = len(constellation0_re)

    # Random symbol indices
    indices = torch.randint(
        low=0,
        high=M0,
        size=(N,),
        device=device
    )

    symbols_re = constellation0_re[indices]
    symbols_im = constellation0_im[indices]

    # AWGN
    noise_scale = math.sqrt(sigma2 / 2.0)

    noise_re = noise_scale * torch.randn(
        N,
        device=device,
        dtype=torch.float32
    )

    noise_im = noise_scale * torch.randn(
        N,
        device=device,
        dtype=torch.float32
    )

    x_re = symbols_re + noise_re
    x_im = symbols_im + noise_im

    log_p0 = log_likelihood_mps(
        x_re,
        x_im,
        constellation0_re,
        constellation0_im,
        sigma2
    )

    log_p1 = log_likelihood_mps(
        x_re,
        x_im,
        constellation1_re,
        constellation1_im,
        sigma2
    )

    return log_p1 - log_p0

@torch.no_grad()
def K_derivatives_mps(L, s):
    """
    Return

        K(s)
        K'(s)
        K''(s)

    for the empirical cumulant generating function.
    """

    s = torch.as_tensor(
        s,
        dtype=L.dtype,
        device=L.device
    )

    z = s * L

    log_Z = torch.logsumexp(z, dim=0)

    log_N = math.log(L.numel())

    K = log_Z - log_N

    # Normalized tilted weights
    log_w = z - log_Z
    w = torch.exp(log_w)

    mean = torch.sum(w * L)

    centered = L - mean

    variance = torch.sum(w * centered**2)

    return K, mean, variance

@torch.no_grad()
def find_s_star_mps(
    L,
    s_initial=0.5,
    max_iter=20,
    tolerance=1e-7
):
    """
    Find s* satisfying K'(s*) = 0.

    Uses Newton's method entirely on the GPU.

    The solution is constrained to [0,1].
    """

    s = torch.tensor(
        s_initial,
        dtype=L.dtype,
        device=L.device
    )

    for _ in range(max_iter):

        K, mean, variance = K_derivatives_mps(
            L,
            s
        )

        # Avoid division by zero
        if variance.item() < 1e-14:
            break

        step = mean / variance

        s_new = s - step

        # Keep saddle point inside [0,1]
        s_new = torch.clamp(
            s_new,
            min=0.0,
            max=1.0
        )

        if torch.abs(s_new - s).item() < tolerance:
            s = s_new
            break

        s = s_new

    return s

@torch.no_grad()
def estimate_from_L_mps(
    L,
    s_initial=0.5
):
    """
    Estimate

        u = (C_star, s_star, cumulants)

    from accumulated LLR samples.
    """

    N = L.numel()

    # --------------------------------------------------
    # Find saddle point
    # --------------------------------------------------

    s_star = find_s_star_mps(
        L,
        s_initial=s_initial
    )

    # --------------------------------------------------
    # Tilted weights
    # --------------------------------------------------

    z = s_star * L

    log_Z = torch.logsumexp(z, dim=0)

    log_w = z - log_Z

    w = torch.exp(log_w)

    # --------------------------------------------------
    # Chernoff information
    # --------------------------------------------------

    K_star = log_Z - math.log(N)

    C_star = -K_star

    # --------------------------------------------------
    # Tilted cumulants
    # --------------------------------------------------

    mean_L = torch.sum(w * L)

    centered = L - mean_L

    var_L = torch.sum(
        w * centered**2
    )

    third = torch.sum(
        w * centered**3
    )

    fourth = torch.sum(
        w * centered**4
    )

    fourth_c = (
        fourth
        - 3.0 * var_L**2
    )

    # --------------------------------------------------
    # Move only six numbers back to CPU
    # --------------------------------------------------

    C_star_cpu = C_star.item()
    s_star_cpu = s_star.item()

    cumulants_cpu = np.array([
        mean_L.item(),
        var_L.item(),
        third.item(),
        fourth_c.item()
    ])

    return (
        C_star_cpu,
        s_star_cpu,
        cumulants_cpu
    )

def normalize_reverse(u_reverse):

    C_reverse, s_reverse, kappa_reverse = u_reverse

    s_forward = 1.0 - s_reverse

    kappa_forward = np.array([
        -kappa_reverse[0],
         kappa_reverse[1],
        -kappa_reverse[2],
         kappa_reverse[3]
    ])

    return (
        C_reverse,
        s_forward,
        kappa_forward
    )

def flatten_u(u):

    C_star, s_star, cumulants = u

    return np.concatenate([
        np.array([C_star, s_star]),
        np.asarray(cumulants)
    ])


def agreement_metrics(u0, u1):

    a = flatten_u(u0)
    b = flatten_u(u1)

    difference = a - b

    return difference

def chernoff_information_batch_mps(
    constellation0,
    constellation1,
    sigma2,
    batch_size=100_000,
    max_batches=100,
    rtol=1e-3,
    # atol=1e-5,
    min_batches=2,
    verbose=True
):
    """
    Apple-Silicon/MPS implementation of the accumulating
    Chernoff-information calculation.

    Two independent Monte-Carlo calculations are performed:

        Forward:
            X ~ p0
            L = log(p1/p0)

        Reverse:
            X ~ p1
            L = log(p0/p1)

    Samples are accumulated between batches.

    Returns
    -------

    u_avg
        Average of the two final estimates.

    u0
        Forward estimate.

    u1
        Reverse estimate converted to forward convention.

    total_N
        Number of samples per direction.

    history
        History of convergence.
    """

    # --------------------------------------------------
    # Convert constellations to GPU representation
    # --------------------------------------------------

    c0_re, c0_im = constellation_to_mps(
        constellation0
    )

    c1_re, c1_im = constellation_to_mps(
        constellation1
    )

    # --------------------------------------------------
    # Preallocate accumulated LLR storage
    # --------------------------------------------------

    max_N = batch_size * max_batches

    L0_all = torch.empty(
        max_N,
        dtype=torch.float32,
        device=device
    )

    L1_all = torch.empty(
        max_N,
        dtype=torch.float32,
        device=device
    )

    history = []

    # Previous saddle-point estimates.
    # These make Newton convergence faster.
    s0_previous = 0.5
    s1_previous = 0.5

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    for batch in range(1, max_batches + 1):

        start = (batch - 1) * batch_size
        end = batch * batch_size

        # --------------------------------------------------
        # Forward direction
        #
        # X ~ p0
        # L = log(p1/p0)
        # --------------------------------------------------

        L0_batch = generate_batch_mps(
            c0_re,
            c0_im,
            c1_re,
            c1_im,
            sigma2,
            batch_size
        )

        # --------------------------------------------------
        # Reverse direction
        #
        # X ~ p1
        # L = log(p0/p1)
        # --------------------------------------------------

        L1_batch = generate_batch_mps(
            c1_re,
            c1_im,
            c0_re,
            c0_im,
            sigma2,
            batch_size
        )

        # --------------------------------------------------
        # Store accumulated samples
        # --------------------------------------------------

        L0_all[start:end] = L0_batch
        L1_all[start:end] = L1_batch

        L0 = L0_all[:end]
        L1 = L1_all[:end]

        total_N = end

        # --------------------------------------------------
        # Estimate forward
        # --------------------------------------------------

        u0 = estimate_from_L_mps(
            L0,
            s_initial=s0_previous
        )

        s0_previous = u0[1]

        # --------------------------------------------------
        # Estimate reverse
        # --------------------------------------------------

        u1_reverse = estimate_from_L_mps(
            L1,
            s_initial=s1_previous
        )

        s1_previous = u1_reverse[1]

        # Convert reverse convention
        u1 = normalize_reverse(
            u1_reverse
        )

        # --------------------------------------------------
        # Compare
        # --------------------------------------------------

        difference = agreement_metrics(
            u0,
            u1
        )

        a = flatten_u(u0)
        b = flatten_u(u1)

        # let's remove the first cumulant, which goes to zero
        agreement = np.allclose(
            [a[0], a[1], a[3], a[4], a[5]],
            [b[0], b[1], b[3], b[4], b[5]],
            rtol=rtol,
            # atol=atol
        )

        # --------------------------------------------------
        # History
        # --------------------------------------------------

        history.append({
            "N": total_N,
            "u0": u0,
            "u1": u1,
            "difference": difference.copy(),
            "agreement": agreement
        })

        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        if verbose:

            print(
                f"\nBatch {batch}"
                f"    N = {total_N:,}"
            )

            print("\nForward:")
            print("  C* =", u0[0])
            print("  s* =", u0[1])
            print("  cumulants =", u0[2])

            print("\nReverse (converted):")
            print("  C* =", u1[0])
            print("  s* =", u1[1])
            print("  cumulants =", u1[2])

            print("\nAbsolute difference:")
            print(difference)

            # A more sensible relative difference:
            # denominator uses max(|a|, |b|)
            denominator = np.maximum(
                np.maximum(np.abs(a), np.abs(b)),
                1e-14
            )

            relative_difference = (
                (a - b)
                / denominator
            )

            print("\nRelative difference:")
            print(relative_difference)

            print(
                "\nAgreement:",
                agreement
            )

        # --------------------------------------------------
        # Convergence
        # --------------------------------------------------

        if (
            batch >= min_batches
            and agreement
        ):

            if verbose:
                print(
                    f"\nConverged after {batch} batches."
                )

                print(
                    f"Total samples per direction: "
                    f"{total_N:,}"
                )

            # --------------------------------------------------
            # Average the two independent estimates
            # --------------------------------------------------

            avg = 0.5 * (a + b)

            u_avg = (
                avg[0],
                avg[1],
                avg[2:]
            )

            return (
                u_avg,
                u0,
                u1,
                total_N,
                history
            )

    raise RuntimeError(
        "Failed to reach requested agreement "
        f"after {max_batches} batches."
    )