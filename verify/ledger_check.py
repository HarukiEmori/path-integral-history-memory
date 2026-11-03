"""Checks for the ledger (reused pointer with commuting marks): F depends on the tallies only,
order blindness, and the closed forms of the minimal example (N = 2, Hadamard steps, calN = 3,
records at epochs 1 and 2, valuation difference theta at both epochs):

  fresh probes, overlap kappa :  P_kappa(1) = 1/2 - (kappa^2/4)(1 - cos 2 theta)
  ledger with characteristic function chi :  P(1) = 1/4 + (chi(2)/4) cos 2 theta
  (sharp counter D >= 3: P(1) = 1/4 ; parity counter D = 2: P(1) = P_q(1) = 1/4 + cos(2 theta)/4)
  all-or-nothing loss with probability kappa :  P(1) = 1/2 - (kappa/4)(1 - cos 2 theta)
"""
import itertools
import numpy as np

from pimh_memory import (RecordProcess, haar_unitary, fresh_probe_process, ledger_process,
                         rotation_ledger_process, correlated_failure_process, tv)

rng = np.random.default_rng(3)
worst = {}


def record(name, val):
    worst[name] = max(worst.get(name, 0.0), float(val))


H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def steps(theta, calN=3):
    V = np.diag([1.0, np.exp(1j * theta)])
    return [H @ V for _ in range(calN)]


# ---------------------------------------------------------------- 1. tally structure
print("1. Ledger: F[a,a'] = chi(Sigma(a) - Sigma(a')) and order blindness")
for D in [2, 3, 5]:
    cats = {1: np.array([0, 1]), 2: np.array([0, 1]), 3: np.array([0, 1])}
    proc = ledger_process(2, cats, D)
    seqs, F = proc.influence_kernel()
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            expect = 1.0 if (sum(a) - sum(b)) % D == 0 else 0.0
            record(f"counter D={D}", abs(F[i, j] - expect))
# rotation ledger: chi(s) = cos(s theta / 2)
for th in [0.4, 1.3, np.pi]:
    proc = rotation_ledger_process(2, {1: np.array([0, 1]), 2: np.array([0, 1]), 3: np.array([0, 1])}, th)
    seqs, F = proc.influence_kernel()
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            record("rotation ledger", abs(F[i, j] - np.cos((sum(a) - sum(b)) * th / 2)))
print("   ok:", {k: f"{v:.1e}" for k, v in worst.items()})

# ---------------------------------------------------------------- 2. minimal example closed forms
print("2. Minimal example closed forms")
cats = {1: np.array([0, 1]), 2: np.array([0, 1])}
for theta in np.linspace(0, 2 * np.pi, 13):
    Ws = steps(theta)
    for kappa in [0.0, 0.3, 0.8, 1.0]:
        P = fresh_probe_process(2, cats, kappa).simulate(Ws, 0)
        record("fresh P(1)", abs(P[1] - (0.5 - kappa ** 2 / 4 * (1 - np.cos(2 * theta)))))
    # counters
    P3 = ledger_process(2, cats, 3).simulate(Ws, 0)
    record("counter D=3 P(1)=1/4", abs(P3[1] - 0.25))
    P2 = ledger_process(2, cats, 2).simulate(Ws, 0)
    record("parity counter P(1)", abs(P2[1] - (0.25 + np.cos(2 * theta) / 4)))
    # rotation ledger, chi(2) = cos(theta_r)
    for th_r in [0.5, 2.0]:
        Pr = rotation_ledger_process(2, cats, th_r).simulate(Ws, 0)
        record("rotation ledger P(1)", abs(Pr[1] - (0.25 + np.cos(th_r) * np.cos(2 * theta) / 4)))
    # all-or-nothing loss
    for kappa in [0.2, 0.6]:
        mu = {frozenset(): 1 - kappa, frozenset({1, 2}): kappa}
        comps = correlated_failure_process(2, cats, mu)
        P = sum(p * pr.simulate(Ws, 0) for p, pr in comps)
        record("all-or-nothing P(1)", abs(P[1] - (0.5 - kappa / 4 * (1 - np.cos(2 * theta)))))
        # independent failures with the same marginal: mixture over subsets
        mu_ind = {frozenset(S): kappa ** len(S) * (1 - kappa) ** (2 - len(S))
                  for r in range(3) for S in itertools.combinations([1, 2], r)}
        comps = correlated_failure_process(2, cats, mu_ind)
        P = sum(p * pr.simulate(Ws, 0) for p, pr in comps)
        record("independent mixture = fresh kappa", abs(P[1] - (0.5 - kappa ** 2 / 4 * (1 - np.cos(2 * theta)))))
print("   ok:", {k: f"{v:.1e}" for k, v in worst.items() if 'P(1)' in k or 'mixture' in k})

# ---------------------------------------------------------------- 3. correlated failures: F = mu(S contains D(a,a'))
print("3. Correlated intermittent records: F[a,a'] = mu{S : S contains the set of differing epochs}")
cats = {1: np.array([0, 1]), 2: np.array([0, 1]), 3: np.array([0, 1])}
subsets = [frozenset(S) for r in range(4) for S in itertools.combinations([1, 2, 3], r)]
w = rng.dirichlet(np.ones(len(subsets)))
mu = dict(zip(subsets, w))
comps = correlated_failure_process(2, cats, mu)
seqs = comps[0][1].sequences()
Fmix = sum(p * pr.influence_kernel()[1] for p, pr in comps)
for i, a in enumerate(seqs):
    for j, b in enumerate(seqs):
        Dset = frozenset(t for t, (u, v) in zip([1, 2, 3], zip(a, b)) if u != v)
        expect = sum(p for S, p in mu.items() if Dset <= S)
        record("correlated F", abs(Fmix[i, j] - expect))
# latitude bound 1 - mu(empty) and the general bound 1 - lambda_min
Ws = [haar_unitary(2, rng) for _ in range(4)]
P = sum(p * pr.simulate(Ws, 0) for p, pr in comps)
P0 = fresh_probe_process(2, cats, 0.0).simulate(Ws, 0)
lam = np.min(np.linalg.eigvalsh(Fmix))
print(f"   ok ({worst['correlated F']:.1e}); TV = {tv(P, P0):.3f} <= 1 - mu(empty) = {1 - mu[frozenset()]:.3f}; lambda_min = {lam:.3f} >= mu(empty) = {mu[frozenset()]:.3f}")
assert tv(P, P0) <= 1 - mu[frozenset()] + 1e-9 and lam >= mu[frozenset()] - 1e-9

print("\nWORST DISCREPANCIES")
for k, v in worst.items():
    print(f"   {k}: {v:.2e}")
print("ALL PASS" if max(worst.values()) < 1e-8 else "CHECK FAILED")
