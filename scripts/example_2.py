import numpy as np
from src.mod_rec_limits import chernoff_information_batch_mps
from src.mod_rec_limits.constellation_schemes import qam_cross_constellation, qam_constellation


np.random.seed(0)
PSNRs = np.linspace(-2, 30, 4)
sigma2s = 10**(-PSNRs/20)


schemes = ["QAM cross 32", "QAM 64", "QAM 256"]
constellations = [qam_cross_constellation(32), qam_constellation(64), qam_constellation(256)]
lengths = [16 * i for i in range(4, 128 * 128)]

Optimal_tilts_per_noise = {}
Chernoff_per_noise = {}
Prefactors_per_noise = {}
Cumulants_per_noise = {}

for k, sigma2 in tqdm(enumerate(sigma2s)):
    Optimal_tilts = np.zeros((len(constellations), len(constellations)))
    Chernoffs = np.zeros((len(constellations), len(constellations)))
    Prefactors = np.zeros((len(constellations), len(constellations)))
    Cumulants = np.zeros((len(constellations), len(constellations), 4))

    for i in range(len(constellations)):
        const0 = constellations[i]
        for j in range(i + 1, len(constellations)):
            const1 = constellations[j]
            C_star, s_star, cumulants = chernoff_information(const0, const1, sigma2, N=1_000_000)
            Optimal_tilts[i, j] = s_star
            Chernoffs[i, j] = C_star
            Cumulants[i, j] = cumulants

    Optimal_tilts_per_noise[str(k)] = Optimal_tilts
    Chernoff_per_noise[str(k)] = Chernoffs
    Cumulants_per_noise[str(k)] = Cumulants

for i in range(len(constellations)):
    for j in range(i + 1, len(constellations)):
        for s, sigma2 in enumerate(sigma2s):

            C_star = Chernoff_per_noise[str(s)][i, j]
            s_star = Optimal_tilts_per_noise[str(s)][i, j]
            cum = Cumulants_per_noise[str(s)][i, j]

            LR_probs_0 = []
            LR_probs_1 = []

            for l in lengths:
                LR_probs_0.append(lugannani_rice_probability(l, C_star, s_star, cum))
                LR_probs_1.append(lugannani_rice_probability(l, C_star, s_star, cum, order=1))

            print("Schemes: ", schemes[i] + " " + schemes[j])
            print("Sigma: ", sigma2)
            print("Largest diff (abs): ", np.max(np.abs(np.asarray(LR_probs_1) - np.asarray(LR_probs_0))))
            print("Largest diff (rel): ", np.max(np.abs((np.asarray(LR_probs_1) - np.asarray(LR_probs_0))/np.asarray(LR_probs_0))))
            eps = 1e-14

            plt.plot(np.log2(lengths), np.log10(np.asarray(LR_probs_0) + eps),
                        label=f"LR 0, Sigma: {sigma2:.2f}, PSNR: {PSNRs[s]:.2f}", linestyle="solid")
            plt.plot(np.log2(lengths), np.log10(np.asarray(LR_probs_1) + eps),
                        label=f"LR 1, Sigma: {sigma2:.2f}, PSNR: {PSNRs[s]:.2f}", linestyle="dashed")

        plt.ylim(bottom=-2.5)
        plt.axhline(np.log10(1 / 2), linestyle='--', color="k", linewidth=0.5)  # , label="p=1/2")
        plt.axhline(np.log10(1 / 4), linestyle='--', color="k", linewidth=0.5)  # , label="p=1/4")
        plt.axhline(np.log10(1 / 8), linestyle='--', color="k", linewidth=0.5)  # , label="p=1/8")
        plt.axhline(np.log10(1 / 16), linestyle='--', color="k", linewidth=0.5)  # , label="p=1/16")
        plt.axhline(np.log10(1 / 32), linestyle='--', color="k", linewidth=0.5)  # , label="p=1/32")
        plt.axhline(np.log10(1 / 64), linestyle='--', color="k", linewidth=0.5)  # , label="p=1/64")
        plt.axhline(np.log10(1 / 128), linestyle='--', color="k", linewidth=0.5)  # , label="p=1/64")
        plt.xlabel("Log2 of sequence length")
        plt.ylabel("Log10 of error probability estimate")
        plt.title("Zeroth/first-order LR approximations of error probability as a function of \n signal length, "
                    + schemes[i] + " versus " + schemes[j])
        plt.legend()
        plt.savefig("Zeroth_first_order_LR_error_probs_" + schemes[i] + " versus " + schemes[j] + ".png")
        plt.close()