# this script attempts to compute the discrepancy between the LR approximation and
# the true value of P(S_n \geq 0), using the techniques given in 
# "SADDLE POINT APPROXIMATION FOR THE DISTRIBUTION OF THE SUM OF INDEPENDENT RANDOM VARIABLES" 

import math
import numpy as np
from numpy.polynomial.hermite import hermgauss
import scipy.special as sp
from src.mod_rec_limits import chernoff_information_batch_mps
from src.mod_rec_limits.constellation_schemes import qam_cross_constellation, qam_constellation


def lugannani_rice_error_bound_m1(
    N,
    C_star,
    s_star,
    cumulants,
):
    """
    Section 5, Eq. (31) of Lugannani & Rice.

    Bounds the error of

        Q_A(1)
          = 1/2 erfc(sqrt(-f0))
            + A0 - B0

    for

        P(sum_{i=1}^N L_i >= 0).

    Requires cumulants kappa_2, kappa_3, kappa_4
    under the saddlepoint-tilted distribution.

    Returns
    -------
    bound : upper bound on |Q_A(1) - Q_N|
    details : dictionary of intermediate quantities
    """

    kappa2 = cumulants[1]
    kappa3 = cumulants[2]
    kappa4 = cumulants[3]

    if kappa2 <= 0:
        raise ValueError("kappa2 must be positive")

    if s_star <= 0:
        raise ValueError(
            "This bound assumes the non-central saddlepoint "
            "case s_star > 0."
        )

    f0 = -N * C_star

    if f0 >= 0:
        raise ValueError("Expected f0 = -N*C_star < 0")

    # --------------------------------------------------
    # Normalised cumulants
    # --------------------------------------------------

    theta3 = (
        kappa3
        / (6.0 * kappa2**1.5)
    )

    theta4 = (
        kappa4
        / (24.0 * kappa2**2)
    )

    # Paper's mu
    mu = 1.0 / (
        s_star * math.sqrt(kappa2)
    )

    # --------------------------------------------------
    # A0, B0
    # --------------------------------------------------

    exp_f0 = math.exp(f0)

    A0 = (
        mu
        / math.sqrt(2.0 * math.pi * N)
        * exp_f0
    )

    B0 = (
        1.0
        / (2.0 * math.sqrt(-math.pi * f0))
        * exp_f0
    )

    # --------------------------------------------------
    # A1, B1
    # --------------------------------------------------

    bracket = (
        (1.0 / 3.0) * mu**2
        + mu * theta3
        + 0.5 * (
            5.0 * theta3**2
            - 2.0 * theta4
        )
    )

    A1 = (
        -3.0
        * A0
        / N
        * bracket
    )

    B1 = B0 / (2.0 * f0)

    remainder_coefficient = A1 - B1

    # --------------------------------------------------
    # Section 5:
    #
    # |EP_1| <=
    # [N/(N-sigma_1)]^(3/2) |A1-B1|
    # --------------------------------------------------

    # sigma_1 must be supplied by the path calculation.
    details = {
        "f0": f0,
        "mu": mu,
        "theta3": theta3,
        "theta4": theta4,
        "A0": A0,
        "B0": B0,
        "A1": A1,
        "B1": B1,
        "A1_minus_B1": remainder_coefficient,
    }

    return details



def make_gh_grid(n_gh=30):
    """
    2D Gauss-Hermite quadrature nodes and weights.
    """

    nodes, weights = hermgauss(n_gh)

    xx, yy = np.meshgrid(
        nodes,
        nodes,
        indexing="ij"
    )

    ww = np.outer(weights, weights)

    return (
        xx.ravel(),
        yy.ravel(),
        ww.ravel()
    )

def gh_samples_for_component(
    constellation_point,
    sigma2,
    n_gh=30
):

    xh, yh, wh = make_gh_grid(n_gh)

    scale = math.sqrt(sigma2 / 2.0)

    x_re = (
        constellation_point.real
        + math.sqrt(2.0) * scale * xh
    )

    x_im = (
        constellation_point.imag
        + math.sqrt(2.0) * scale * yh
    )

    # hermgauss integrates exp(-x^2) f(x)
    # so expectation under standard Gaussian
    # has factor 1/pi in 2D.

    weights = wh / math.pi

    return x_re, x_im, weights

def log_likelihood_np(
    x_re,
    x_im,
    constellation,
    sigma2
):

    c_re = np.real(constellation)
    c_im = np.imag(constellation)

    d_re = (
        x_re[:, None]
        - c_re[None, :]
    )

    d_im = (
        x_im[:, None]
        - c_im[None, :]
    )

    d2 = d_re**2 + d_im**2

    log_terms = -d2 / sigma2

    return (
        sp.logsumexp(
            log_terms,
            axis=1
        )
        - math.log(len(constellation))
        - math.log(math.pi * sigma2)
    )

def complex_mgf_gauss_hermite(
    t,
    constellation0,
    constellation1,
    sigma2,
    n_gh=30
):
    """
    Deterministic approximation to

        M(t) = E_p0[ exp(t L) ]

    for complex t.
    """

    total_M = 0.0 + 0.0j

    for a in constellation0:

        x_re, x_im, w = (
            gh_samples_for_component(
                a,
                sigma2,
                n_gh
            )
        )

        log_p0 = log_likelihood_np(
            x_re,
            x_im,
            constellation0,
            sigma2
        )

        log_p1 = log_likelihood_np(
            x_re,
            x_im,
            constellation1,
            sigma2
        )

        L = log_p1 - log_p0

        total_M += (
            np.sum(
                w * np.exp(t * L)
            )
            / len(constellation0)
        )

    return total_M

def complex_K_and_derivative(
    t,
    constellation0,
    constellation1,
    sigma2,
    n_gh=30
):

    M = 0.0 + 0.0j
    M1 = 0.0 + 0.0j

    for a in constellation0:

        x_re, x_im, w = (
            gh_samples_for_component(
                a,
                sigma2,
                n_gh
            )
        )

        log_p0 = log_likelihood_np(
            x_re,
            x_im,
            constellation0,
            sigma2
        )

        log_p1 = log_likelihood_np(
            x_re,
            x_im,
            constellation1,
            sigma2
        )

        L = log_p1 - log_p0

        e = np.exp(t * L)

        M += np.sum(w * e) / len(constellation0)

        M1 += (
            np.sum(w * L * e)
            / len(constellation0)
        )

    K = np.log(M)

    K1 = M1 / M

    return K, K1


def trace_steepest_path(
    t0,
    constellation0,
    constellation1,
    sigma2,
    n_gh=30,
    delta=1e-3,
    max_steps=10000,
    sink_tol=1e-10
):
    """
    Numerically trace the steepest-descent path starting
    from the saddlepoint t0.

    Returns a list containing t, K(t), K'(t), and tau.
    """

    path = []

    # Saddlepoint value
    K0, K1_0 = complex_K_and_derivative(
        t0,
        constellation0,
        constellation1,
        sigma2,
        n_gh
    )

    gamma0 = K0.real

    # Start just above the saddle in the transformed plane
    t = complex(t0, delta)

    for j in range(max_steps):

        K, K1 = complex_K_and_derivative(
            t,
            constellation0,
            constellation1,
            sigma2,
            n_gh
        )

        # tau = gamma0 - gamma(t)
        tau = gamma0 - K

        path.append({
            "t": t,
            "K": K,
            "K1": K1,
            "tau": tau
        })

        # Stop if we have gone sufficiently far
        # from the saddle / reached numerical trouble.
        if abs(K1) < 1e-14:
            break

        M = np.exp(K)

        if abs(M) < sink_tol:
            break

        # Steepest-descent step
        t = t - delta * abs(K1) / K1

    return path

def C0_lugannani_rice(
    C_star,
    s_star,
    kappa2
):

    gamma0 = -C_star

    mu = 1.0 / (
        s_star * math.sqrt(kappa2)
    )

    return (
        mu / math.sqrt(2.0)
        - 1.0 / (
            2.0 * math.sqrt(-gamma0)
        )
    )

def F1_from_path_point(
    t,
    K1,
    tau,
    gamma0,
    C0
):

    # First term
    term1 = (
        0.5
        / np.sqrt(-gamma0 * tau)
        / (1.0 - tau / gamma0)
    )

    # Second term
    term2 = -np.real(
        1j / (t * K1)
    )

    # Third term
    term3 = (
        C0
        / np.sqrt(tau)
    )

    return (
        term1
        + term2
        + term3
    )


def sigma1_profile(
    path,
    C_star,
    C0
):

    gamma0 = -C_star

    profile = []

    for p in path:

        t = p["t"]
        tau = p["tau"]
        K1 = p["K1"]

        if abs(tau) < 1e-12:
            continue

        F1 = F1_from_path_point(
            t,
            K1,
            tau,
            gamma0,
            C0
        )

        value = (
            abs(F1)
            / (
                abs(C0)
                * abs(tau)**1.5
            )
        )

        profile.append(
            (tau.real, value)
        )

    return np.asarray(profile)

def section5_bound_m1(
    N,
    C_star,
    s_star,
    cumulants,
    sigma1
):

    details = lugannani_rice_error_bound_m1(
        N,
        C_star,
        s_star,
        cumulants
    )

    A1_minus_B1 = (
        details["A1_minus_B1"]
    )

    factor = (
        N / (N - sigma1)
    ) ** 1.5

    bound = (
        factor
        * abs(A1_minus_B1)
    )

    details["sigma1"] = sigma1
    details["section5_factor"] = factor
    details["error_bound"] = bound

    return bound, details


if __name__=='__main__':

    PSNR = 20
    sigma2 = 10 ** (-PSNR / 20)

    const0 = qam_cross_constellation(32)
    const1 = qam_constellation(64)

    u_avg, u0, u1, N, history = (
        chernoff_information_batch_mps(
            const0,
            const1,
            sigma2,
            batch_size=100_000,
            max_batches=200,
            rtol=5e-3,
            # atol=1e-5
        )
    )

    C_star, s_star, cumulants = u_avg

    breakpoint()

    # sanity check
    M_star = complex_mgf_gauss_hermite(
        s_star,
        const0,
        const1,
        sigma2,
        n_gh=30
    )

    print("M(s*) =", M_star)
    print("-log M(s*) =", -np.log(M_star.real))
    print("Monte Carlo C* =", C_star)

    breakpoint()

    path = trace_steepest_path(
        s_star,
        const0,
        const1,
        sigma2,
        n_gh=30,
        delta=5e-4,
        max_steps=500
    )

    print("Number of path points:", len(path))

    for p in path[:10]:
        print(
            "t =", p["t"],
            "K =", p["K"],
            "K' =", p["K1"],
            "tau =", p["tau"]
        )

    breakpoint()

    kappa2 = cumulants[1]

    C0 = C0_lugannani_rice(
        C_star,
        s_star,
        kappa2
    )

    # another sanity check

    # gamma0 = -C_star
    # mu = 1.0 / (
    #         s_star * np.sqrt(kappa2)
    # )
    #
    # C0_check = (
    #         mu / np.sqrt(2.0)
    #         - 1.0 / (
    #                 2.0 * np.sqrt(-gamma0)
    #         )
    # )
    #
    # print(C0)
    # print(C0_check)

    breakpoint()

    profile = sigma1_profile(
        path,
        C_star,
        C0
    )

    idx = np.argmax(profile[:, 1])

    print(
        "sigma1 =",
        profile[idx, 1]
    )

    print(
        "attained near tau =",
        profile[idx, 0]
    )

    breakpoint()

    sigma1 = profile[idx, 1]
    N = 512  # number of samples
    bound, details = section5_bound_m1(
        N,
        C_star,
        s_star,
        cumulants,
        sigma1
    )

    print("\nSection 5 result")
    print("----------------")
    print("N =", N)
    print("C* =", C_star)
    print("s* =", s_star)
    print("kappa2 =", cumulants[1])
    print("kappa3 =", cumulants[2])
    print("kappa4 =", cumulants[3])
    print("sigma1 =", sigma1)

    print("\nA0 =", details["A0"])
    print("B0 =", details["B0"])
    print("A1 =", details["A1"])
    print("B1 =", details["B1"])
    print("|A1-B1| =", abs(details["A1_minus_B1"]))

    print("\nSection 5 multiplicative factor =",
          details["section5_factor"])

    print("\nFINAL ERROR BOUND =",
          details["error_bound"])

    breakpoint()