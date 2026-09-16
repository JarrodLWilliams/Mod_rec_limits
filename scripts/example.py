from src.mod_rec_limits.constellation_schemes import qam_constellation, qam_cross_constellation
from src.mod_rec_limits.chernoff import chernoff_information_batch_mps


PSNR = 5
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

print("\nFINAL RESULT")
print("C* =", C_star)
print("s* =", s_star)
print("cumulants =", cumulants)
print("N =", N)