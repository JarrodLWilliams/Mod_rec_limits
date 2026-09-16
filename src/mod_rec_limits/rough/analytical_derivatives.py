# implementations of the derivatives of various functions. The idea was that 
# various quantities such as the optimal s^* and Chernoff information 
# can be shown to satisfy ODEs that would allow one to compute how to update them 
# for increasing levels of noise. 


import numpy as np
from scipy.optimize import minimize_scalar

# -----------------------------
# 1. Define constellations
# -----------------------------
def BPSK():
    return np.array([-1, 1])

def QPSK():
    return np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

# -----------------------------
# 2. Gaussian log-likelihood
# -----------------------------
def log_likelihood(x, constellation, sigma2):
    # Mixture of Gaussians log-sum-exp
    dists = np.abs(x[:, None] - constellation[None, :])**2
    log_probs = -dists / (2*sigma2)
    max_log = np.max(log_probs, axis=1, keepdims=True)
    return (max_log + np.log(np.mean(np.exp(log_probs - max_log), axis=1))).squeeze()

# -----------------------------
# 3. Decision-boundary sampling
# -----------------------------
def sample_near_boundary(const0, const1, N, sigma2, seed=None):
    rng = np.random.default_rng(seed)
    x_samples = []
    for _ in range(N):
        # pick nearest pair of points
        s0 = const0[rng.integers(len(const0))]
        s1 = const1[rng.integers(len(const1))]
        u = rng.uniform()
        x = s0 + u*(s1 - s0)  # line segment connecting nearest neighbors
        # add Gaussian noise
        x += np.sqrt(sigma2/2)*(rng.standard_normal() + 1j*rng.standard_normal())
        x_samples.append(x)
    return np.array(x_samples)

# -----------------------------
# 4. Compute L(x)
# -----------------------------
def L_x(x, const0, const1, sigma2):
    return log_likelihood(x, const0, sigma2) - log_likelihood(x, const1, sigma2)

# -----------------------------
# 5. Tilted expectations
# -----------------------------
def tilted_expectations(L, t):
    exp_tL = np.exp(t*L)
    mean_L = np.sum(exp_tL * L) / np.sum(exp_tL)
    var_L = np.sum(exp_tL * (L - mean_L)**2) / np.sum(exp_tL)
    return mean_L, var_L

# -----------------------------
# 6. Derivative wrt sigma2 (SNR)
# -----------------------------
def dL_dsigma2(x, const0, const1, sigma2):
    # derivative of log-sum-exp
    def dlogsum(constellation):
        dists = np.abs(x[:, None] - constellation[None, :])**2
        weights = np.exp(-dists/(2*sigma2))
        weights /= np.sum(weights, axis=1, keepdims=True)
        return np.sum(weights * dists, axis=1) / (2*sigma2**2)
    return dlogsum(const0) - dlogsum(const1)

def tilted_snr_derivatives(L, dL, t):
    exp_tL = np.exp(t*L)
    mean_dL = np.sum(exp_tL * dL) / np.sum(exp_tL)
    cov_L_dL = np.sum(exp_tL * (L - np.sum(exp_tL*L)/np.sum(exp_tL)) * (dL - mean_dL)) / np.sum(exp_tL)
    return mean_dL, cov_L_dL

# -----------------------------
# 7. Compute Chernoff info, t*, prefactor
# -----------------------------
def compute_chernoff(const0, const1, sigma2, N=20000, seed=0):
    x_samples = sample_near_boundary(const0, const1, N, sigma2, seed)
    L = L_x(x_samples, const0, const1, sigma2)
    dL = dL_dsigma2(x_samples, const0, const1, sigma2)

    # optimize t*
    def objective(t):
        mean_L, _ = tilted_expectations(L, t)
        return np.abs(mean_L)
    res = minimize_scalar(objective, bounds=(0,1), method='bounded')
    t_star = res.x

    # Chernoff info
    mean_L, var_L = tilted_expectations(L, t_star)
    I = np.log(np.mean(np.exp(t_star * L)))

    # SNR derivative
    mean_dL, cov_L_dL = tilted_snr_derivatives(L, dL, t_star)

    # Bahadur-Rao prefactor
    sigma2_L = var_L
    prefactor = 1 / np.sqrt(2*np.pi*N*sigma2_L)

    return I, t_star, prefactor

# -----------------------------
# 8. Example usage
# -----------------------------
if __name__ == "__main__":
    sigma2 = 0.1  # example SNR
    I, t_star, prefactor = compute_chernoff(BPSK(), QPSK(), sigma2)
    print(f"Chernoff I = {I:.4f}, t* = {t_star:.4f}, prefactor = {prefactor:.4e}")