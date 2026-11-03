"""Checks for the Gaussian pointer field (Proposition on Gaussian records).

Pointers xi_t, t in R, in a pure Gaussian state with position covariance Sigma (real wave function),
marks U_t^(a) = exp(-i lambda mu_a pi_t).  Then
   F[a,a'] = exp(-(1/8) (d-d')^T Sigma^{-1} (d-d'))  with d = lambda mu(a),
           = exp(-(lambda^2/2) Delta^T Pi Delta)       with Pi = <pi pi^T> = Sigma^{-1}/4,
and the second form holds for any (also mixed, classical) Gaussian momentum distribution with
covariance Pi.  Checks: (1) the overlap formula by numerical quadrature in two dimensions,
(2) Pi = Sigma^{-1}/4 for the pure state, (3) the stationary (AR(1)) kernel: persistent versus
alternating differences, white-noise and ledger limits, (4) the spectral (filter-function) form.
"""
import numpy as np

worst = {}


def record(name, val):
    worst[name] = max(worst.get(name, 0.0), float(val))


# ---------------------------------------------------------------- (1) overlap by quadrature
print("(1) Overlap of displaced two-dimensional Gaussians versus exp(-(1/8) D^T Sigma^{-1} D)")
rng = np.random.default_rng(5)
for trial in range(5):
    A = rng.normal(size=(2, 2))
    Sigma = A @ A.T + 0.5 * np.eye(2)
    Sinv = np.linalg.inv(Sigma)
    L = 12.0
    n = 801
    xs = np.linspace(-L, L, n)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    pts = np.stack([X, Y], axis=-1)

    def psi(shift):
        z = pts - shift
        q = np.einsum('...i,ij,...j->...', z, Sinv, z)
        return np.exp(-q / 4)

    norm = np.sum(psi(np.zeros(2)) ** 2) * (xs[1] - xs[0]) ** 2
    d = rng.normal(size=2) * 1.5
    dp = rng.normal(size=2) * 1.5
    ov = np.sum(psi(d) * psi(dp)) * (xs[1] - xs[0]) ** 2 / norm
    expect = np.exp(-(d - dp) @ Sinv @ (d - dp) / 8)
    record("overlap quadrature", abs(ov - expect))
print(f"   ok: {worst['overlap quadrature']:.1e}")

# ---------------------------------------------------------------- (2) momentum covariance
print("(2) Momentum covariance of the pure Gaussian state equals Sigma^{-1}/4")
for trial in range(3):
    A = rng.normal(size=(2, 2))
    Sigma = A @ A.T + 0.5 * np.eye(2)
    Sinv = np.linalg.inv(Sigma)
    L, n = 12.0, 801
    xs = np.linspace(-L, L, n)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    pts = np.stack([X, Y], axis=-1)
    q = np.einsum('...i,ij,...j->...', pts, Sinv, pts)
    psi0 = np.exp(-q / 4)
    psi0 /= np.sqrt(np.sum(psi0 ** 2) * h * h)
    gx, gy = np.gradient(psi0, h, h)
    Pi = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gy * gx), np.sum(gy * gy)]]) * h * h
    record("Pi - Sigma^{-1}/4", np.max(np.abs(Pi - Sinv / 4)))
print(f"   ok: {worst['Pi - Sigma^{-1}/4']:.1e}")

# ---------------------------------------------------------------- (3) AR(1) kernel
print("(3) Stationary kernel Pi_tt' = pi0^2 r^|t-t'|: damping of persistent and alternating differences")
lam2pi2 = 2.0 * np.log(2.0)  # lambda^2 pi0^2 chosen so that a single-epoch difference gives F = 1/2
m = 4
pers = np.ones(m)
alt = np.array([1, -1, 1, -1], dtype=float)
single = np.array([1, 0, 0, 0], dtype=float)
print("     r      F(single)  F(persistent)  F(alternating)   1-lambda_min(F) for 2 categories")
import itertools
seqs = list(itertools.product([0, 1], repeat=m))
for r in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    Pi = np.array([[r ** abs(t - s) for s in range(m)] for t in range(m)])
    def F_of(delta):
        return np.exp(-0.5 * lam2pi2 * delta @ Pi @ delta)
    Fmat = np.array([[F_of(np.array(a) - np.array(b)) for b in seqs] for a in seqs])
    lam = np.min(np.linalg.eigvalsh(Fmat))
    print(f"   {r:5.2f}   {F_of(single):8.4f}   {F_of(pers):10.4f}    {F_of(alt):10.4f}       {1 - lam:8.4f}")
    if r == 0.0:
        record("white noise: product", abs(F_of(pers) - F_of(single) ** m))
        record("white noise: lambda_min = (1-kappa)^m", abs(lam - (1 - F_of(single)) ** m))
    if r == 1.0:
        record("ledger: alternating undamped", abs(F_of(alt) - 1))
        record("ledger: persistent = chi(4)", abs(F_of(pers) - F_of(single) ** 16))
print("   (white noise: product of single-epoch factors; ledger r = 1: F depends on the tally only)")

# ---------------------------------------------------------------- (4) spectral form
print("(4) Spectral form Delta^T Pi Delta = (1/2pi) int S(w) |Delta^(w)|^2 dw for a stationary kernel")
r = 0.6
m = 6
gamma = lambda tau: r ** abs(tau)
Pi = np.array([[gamma(t - s) for s in range(m)] for t in range(m)])
w = np.linspace(-np.pi, np.pi, 20001)
S = np.real(sum(gamma(tau) * np.exp(-1j * w * tau) for tau in range(-60, 61)))  # S(w) = (1-r^2)/(1-2r cos w+r^2)
record("S(w) closed form", np.max(np.abs(S - (1 - r ** 2) / (1 - 2 * r * np.cos(w) + r ** 2))))
for delta in [np.ones(m), np.array([1, -1, 1, -1, 1, -1.0]), rng.normal(size=m)]:
    Dhat = sum(delta[t] * np.exp(-1j * w * t) for t in range(m))
    integral = np.trapezoid(S * np.abs(Dhat) ** 2, w) / (2 * np.pi)
    record("spectral form", abs(integral - delta @ Pi @ delta))
print(f"   ok: {worst['spectral form']:.1e}; S(w) = (1-r^2)/(1-2r cos w+r^2) with error {worst['S(w) closed form']:.1e}")

print("\nWORST DISCREPANCIES")
for k, v in worst.items():
    print(f"   {k}: {v:.2e}")
ok = all((v < 1e-3) if k.startswith("Pi - ") else (v < 1e-6) for k, v in worst.items())
print("ALL PASS" if ok else "CHECK FAILED")
