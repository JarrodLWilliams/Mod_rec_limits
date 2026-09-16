# ============================================================
# BIVARIATE SADDLEPOINT FUNCTIONS
#
# We have samples under P0:
#
#     L10 = log(p1 / p0)
#     L20 = log(p2 / p0)
#
# and want
#
#     P0(L10 > 0, L20 > 0)
#
# ============================================================

import math
import torch
import numpy as np


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
def K_derivatives_bivariate_mps(L1, L2, s, t):
    """
    Empirical bivariate CGF and its gradient/Hessian.

    K(s,t) = log E_0[ exp(s L1 + t L2) ]

    Returns
    -------
    K       : scalar
    grad    : [2] tensor
              [K_s, K_t]
    Hessian : [2,2] tensor
              [[K_ss, K_st],
               [K_st, K_tt]]
    """

    s = torch.as_tensor(
        s,
        dtype=L1.dtype,
        device=L1.device
    )

    t = torch.as_tensor(
        t,
        dtype=L1.dtype,
        device=L1.device
    )

    z = s * L1 + t * L2

    log_Z = torch.logsumexp(z, dim=0)
    log_N = math.log(L1.numel())

    K = log_Z - log_N

    # --------------------------------------------------------
    # Tilted probabilities
    # --------------------------------------------------------

    log_w = z - log_Z
    w = torch.exp(log_w)

    # --------------------------------------------------------
    # Tilted mean
    # --------------------------------------------------------

    mean1 = torch.sum(w * L1)
    mean2 = torch.sum(w * L2)

    grad = torch.stack([
        mean1,
        mean2
    ])

    # --------------------------------------------------------
    # Tilted covariance
    # --------------------------------------------------------

    d1 = L1 - mean1
    d2 = L2 - mean2

    var11 = torch.sum(w * d1 * d1)
    var22 = torch.sum(w * d2 * d2)
    cov12 = torch.sum(w * d1 * d2)

    Hessian = torch.stack([
        torch.stack([var11, cov12]),
        torch.stack([cov12, var22])
    ])

    return K, grad, Hessian


@torch.no_grad()
def find_bivariate_s_star_mps(
    L1,
    L2,
    s_initial=0.5,
    t_initial=0.5,
    max_iter=30,
    tolerance=1e-7
):
    """
    Solve

        grad K(s,t) = 0

    using 2D Newton iteration.

    This finds the interior saddlepoint associated with
    the boundary (L1,L2) = (0,0).

    Returns
    -------
    s_star, t_star
    """

    s = torch.tensor(
        s_initial,
        dtype=L1.dtype,
        device=L1.device
    )

    t = torch.tensor(
        t_initial,
        dtype=L1.dtype,
        device=L1.device
    )

    for _ in range(max_iter):

        K, grad, H = K_derivatives_bivariate_mps(
            L1,
            L2,
            s,
            t
        )

        # ----------------------------------------------------
        # Check that Hessian is usable
        # ----------------------------------------------------

        det_H = (
            H[0, 0] * H[1, 1]
            - H[0, 1] * H[1, 0]
        )

        if det_H.item() <= 1e-14:
            raise RuntimeError(
                "Bivariate Hessian is singular or nearly singular."
            )

        # ----------------------------------------------------
        # Newton step:
        #
        #     H delta = grad
        #
        #     x_new = x - delta
        # ----------------------------------------------------

        delta = torch.linalg.solve(H, grad)

        s_new = s - delta[0]
        t_new = t - delta[1]

        # ----------------------------------------------------
        # The useful saddlepoint for the upper-right quadrant
        # should normally have positive coordinates.
        #
        # Do NOT impose the simplex constraint s+t <= 1 here.
        #
        # This is a tail saddlepoint, not the three-way
        # Chernoff mixture saddlepoint.
        # ----------------------------------------------------

        s_new = torch.clamp(
            s_new,
            min=1e-8
        )

        t_new = torch.clamp(
            t_new,
            min=1e-8
        )

        step_size = torch.maximum(
            torch.abs(s_new - s),
            torch.abs(t_new - t)
        )

        s = s_new
        t = t_new

        if step_size.item() < tolerance:
            break

    return s, t


@torch.no_grad()
def estimate_bivariate_quadrant_mps(
    L1,
    L2,
    s_initial=0.5,
    t_initial=0.5
):
    """
    Estimate

        P0(L1 > 0, L2 > 0)

    using the leading multivariate saddlepoint approximation.

    The saddlepoint is defined by

        grad K(s*,t*) = 0.

    The approximation used is

        P ~= exp(K*) /
             [2*pi*s*t*sqrt(det(H))]

    where H is the tilted covariance matrix.

    Returns
    -------
    result : dict
        Contains saddlepoint, CGF, covariance, correlation,
        Chernoff/rate quantity, and probability estimate.
    """

    N = L1.numel()

    # --------------------------------------------------------
    # Find saddlepoint
    # --------------------------------------------------------

    s_star, t_star = find_bivariate_s_star_mps(
        L1,
        L2,
        s_initial=s_initial,
        t_initial=t_initial
    )

    breakpoint()
    # --------------------------------------------------------
    # Evaluate CGF and derivatives at saddlepoint
    # --------------------------------------------------------

    K_star, grad_star, H_star = (
        K_derivatives_bivariate_mps(
            L1,
            L2,
            s_star,
            t_star
        )
    )

    # --------------------------------------------------------
    # Determinant
    # --------------------------------------------------------

    det_H = (
        H_star[0, 0] * H_star[1, 1]
        - H_star[0, 1] * H_star[1, 0]
    )

    if det_H.item() <= 0:
        raise RuntimeError(
            "Non-positive Hessian determinant."
        )

    # --------------------------------------------------------
    # Correlation under the tilted distribution
    # --------------------------------------------------------

    rho = (
        H_star[0, 1]
        / torch.sqrt(
            H_star[0, 0] * H_star[1, 1]
        )
    )

    # --------------------------------------------------------
    # Large-deviation / saddlepoint rate
    #
    # I(0,0) = -K(s*,t*)
    # --------------------------------------------------------

    I_star = -K_star

    # --------------------------------------------------------
    # Leading bivariate quadrant approximation
    # --------------------------------------------------------

    denominator = (
        2.0
        * math.pi
        * s_star
        * t_star
        * torch.sqrt(det_H)
    )

    p_joint = torch.exp(K_star) / denominator

    # --------------------------------------------------------
    # Guard against tiny numerical excursions
    # --------------------------------------------------------

    p_joint = torch.clamp(
        p_joint,
        min=0.0,
        max=1.0
    )

    return {
        "p_joint": p_joint.item(),

        "log_p_joint": torch.log(
            torch.clamp(p_joint, min=1e-30)
        ).item(),

        "I_star": I_star.item(),

        "s_star": s_star.item(),
        "t_star": t_star.item(),

        "K_star": K_star.item(),

        "grad_s": grad_star[0].item(),
        "grad_t": grad_star[1].item(),

        "var_L1": H_star[0, 0].item(),
        "cov_L1_L2": H_star[0, 1].item(),
        "var_L2": H_star[1, 1].item(),

        "rho": rho.item(),

        "det_H": det_H.item()
    }


@torch.no_grad()
def generate_bivariate_batch_mps(
    constellation0_re,
    constellation0_im,
    constellation1_re,
    constellation1_im,
    constellation2_re,
    constellation2_im,
    sigma2,
    N
):
    """
    Generate N samples under P0 and return

        L10 = log(p1 / p0)
        L20 = log(p2 / p0)

    using the SAME observations for both LLRs.
    """

    M0 = len(constellation0_re)

    # --------------------------------------------------------
    # Draw symbols from P0
    # --------------------------------------------------------

    indices = torch.randint(
        low=0,
        high=M0,
        size=(N,),
        device=device
    )

    symbols_re = constellation0_re[indices]
    symbols_im = constellation0_im[indices]

    # --------------------------------------------------------
    # AWGN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Three likelihoods
    # --------------------------------------------------------

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

    log_p2 = log_likelihood_mps(
        x_re,
        x_im,
        constellation2_re,
        constellation2_im,
        sigma2
    )

    # --------------------------------------------------------
    # LLRs relative to P0
    # --------------------------------------------------------

    L10 = log_p1 - log_p0
    L20 = log_p2 - log_p0

    return L10, L20


@torch.no_grad()
def bivariate_quadrant_batch_mps(
    constellation0,
    constellation1,
    constellation2,
    sigma2,
    batch_size=100_000,
    max_batches=100,
    min_batches=2,
    rtol=1e-3,
    verbose=True
):
    """
    Estimate

        P0(L10 > 0, L20 > 0)

    where

        L10 = log(p1/p0)
        L20 = log(p2/p0).

    Samples are accumulated between batches.
    """

    # --------------------------------------------------------
    # Convert constellations to MPS
    # --------------------------------------------------------

    c0_re, c0_im = constellation_to_mps(
        constellation0
    )

    c1_re, c1_im = constellation_to_mps(
        constellation1
    )

    c2_re, c2_im = constellation_to_mps(
        constellation2
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    max_N = batch_size * max_batches

    L10_all = torch.empty(
        max_N,
        dtype=torch.float32,
        device=device
    )

    L20_all = torch.empty(
        max_N,
        dtype=torch.float32,
        device=device
    )

    s_previous = 0.5
    t_previous = 0.5

    history = []

    previous_p = None

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for batch in range(1, max_batches + 1):

        start = (batch - 1) * batch_size
        end = batch * batch_size

        # ----------------------------------------------------
        # Generate samples under P0
        # ----------------------------------------------------

        L10_batch, L20_batch = (
            generate_bivariate_batch_mps(
                c0_re,
                c0_im,
                c1_re,
                c1_im,
                c2_re,
                c2_im,
                sigma2,
                batch_size
            )
        )

        L10_all[start:end] = L10_batch
        L20_all[start:end] = L20_batch

        L10 = L10_all[:end]
        L20 = L20_all[:end]

        total_N = end

        # ----------------------------------------------------
        # Estimate
        # ----------------------------------------------------

        result = estimate_bivariate_quadrant_mps(
            L10,
            L20,
            s_initial=s_previous,
            t_initial=t_previous
        )

        s_previous = result["s_star"]
        t_previous = result["t_star"]

        p_current = result["p_joint"]

        # ----------------------------------------------------
        # Monte-Carlo convergence diagnostic
        # ----------------------------------------------------

        if previous_p is None:
            breakpoint()
            relative_change = np.nan
        else:
            relative_change = (
                abs(p_current - previous_p)
                / max(abs(p_current), 1e-30)
            )

        previous_p = p_current

        result["N"] = total_N
        result["relative_change"] = relative_change

        history.append(result.copy())

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        if verbose:

            print(
                f"\nBatch {batch}"
                f"    N = {total_N:,}"
            )

            print(
                "  P0(L10>0,L20>0) =",
                result["p_joint"]
            )

            print(
                "  I* =",
                result["I_star"]
            )

            print(
                "  s* =",
                result["s_star"]
            )

            print(
                "  t* =",
                result["t_star"]
            )

            print(
                "  rho* =",
                result["rho"]
            )

            print(
                "  grad K =",
                (
                    result["grad_s"],
                    result["grad_t"]
                )
            )

            print(
                "  relative change =",
                relative_change
            )

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        if (
            batch >= min_batches
            and not np.isnan(relative_change)
            and relative_change < rtol
        ):

            if verbose:
                print(
                    f"\nConverged after {batch} batches."
                )

            return (
                result,
                total_N,
                history
            )

    return (
        result,
        total_N,
        history
    )


# ============================================================
# Example
# ============================================================


if __name__=='__main__':

    PSNR = 30

    sigma2 = 10 ** (-PSNR / 20)

    constellation0 = qam_cross_constellation(32)
    constellation1 = qam_constellation(16)

    # Example third hypothesis.
    # Replace this with whatever your actual H2 is.
    constellation2 = qam_constellation(64)


    result, total_N, history = (
        bivariate_quadrant_batch_mps(
            constellation0,
            constellation1,
            constellation2,
            sigma2,
            batch_size=100_000,
            max_batches=100,
            min_batches=2,
            rtol=1e-10,
            verbose=True
        )
    )


    print("\n======================================")
    print("Final bivariate saddlepoint estimate")
    print("======================================")

    print(
        "P0(L10 > 0, L20 > 0) =",
        result["p_joint"]
    )

    print(
        "I* =",
        result["I_star"]
    )

    print(
        "s* =",
        result["s_star"]
    )

    print(
        "t* =",
        result["t_star"]
    )

    print(
        "rho* =",
        result["rho"]
    )