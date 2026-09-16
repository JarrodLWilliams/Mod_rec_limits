# functionality for computing the Lugannani--Rice (LR) expansion.


import numpy as np
import scipy as sp
import math


def lugannani_rice_probability(
    N,
    C_star,
    s_star,
    cumulants,
    order=0,
):
    """
    LR approximation for

        P(sum L_i >= 0)

    using quantities returned by
    chernoff_information_with_K().

    cumulants = mean, variance, third cumulant, fourth cumulant  (of tilted measure)
    """

    # var_L is equal to phi^(2) in the notation of the paper
    var_L = cumulants[1]

    # calculating the normalised cumulants
    theta3 = cumulants[2] / (6 * math.sqrt(var_L**3))
    theta4 = cumulants[3] / (24 * var_L ** 2)

    # w = np.sqrt(2.0 * N * C_star)    # where is the 2 from?
    mu = 1/(s_star * math.sqrt(var_L))
    f0 = -N*C_star

    A0 = (mu/math.sqrt(2*math.pi*N))*np.exp(f0)
    A1 = -3*(A0/N)*((1/3)*mu**2 + mu*theta3 + (1/2)*(5*theta3**2 - 2*theta4))
    B0 = (1/(2*math.sqrt(-math.pi*f0)))*np.exp(f0)
    B1 = (1/(4*math.sqrt(-math.pi*f0)*f0))*np.exp(f0)

    if order==0:
        out = (1/2)*sp.special.erfc(math.sqrt(-f0)) + A0 - B0

    elif order==1:
        out = (1/2)*sp.special.erfc(math.sqrt(-f0)) + A0 - B0 + A1 - B1

    else:
        AssertionError("Order not supported.")

    return out