"""Checks for correlated preparations of the probes.

(a) Entangled pair: two recorded epochs, probes in the maximally entangled state of two d-level
    systems, marks X^a on the respective halves.  Influence kernel F = [a_2 - a_1 = a'_2 - a'_1 mod d]
    (a record of the change between the epochs).  The single record dephases completely, the second
    record restores the coherence of courses that differ at both epochs (recoherence), the trace
    distance between two system states increases between the records (no CP-divisible, indeed no
    linear-map, description), and the general two-epoch formula F = <Phi0| G (x) G' |Phi0> holds.
(b) Chain of pairs: the sequence of changes is recorded and nothing else.
(c) Classically correlated intermittent records: the latitude bound 1 - mu(empty) and its range over
    couplings with given marginals (Frechet bounds).
"""
import itertools
import numpy as np

from pimh_memory import (RecordProcess, haar_unitary, random_state, hadamard_matrix, tv,
                         trace_distance, fresh_probe_process, bell_pair_process,
                         bell_chain_process, correlated_failure_process)

rng = np.random.default_rng(11)
worst = {}


def record(name, val):
    worst[name] = max(worst.get(name, 0.0), float(val))


# ---------------------------------------------------------------- (a) entangled pair
print("(a) Entangled pair of probes")
for d in [2, 3, 4]:
    N = d
    cats = {1: np.arange(d), 2: np.arange(d)}
    proc = bell_pair_process(N, cats, d)
    seqs, F = proc.influence_kernel()
    for i, (a1, a2) in enumerate(seqs):
        for j, (b1, b2) in enumerate(seqs):
            expect = 1.0 if (a2 - a1 - (b2 - b1)) % d == 0 else 0.0
            record(f"pair d={d} F=[change equal]", abs(F[i, j] - expect))
    # single-record kernel (prefix): F_1[a,a'] = delta
    for i, (a1, a2) in enumerate(seqs):
        for j, (b1, b2) in enumerate(seqs):
            if a2 == b2:
                record(f"pair d={d} first record sharp", abs(F[i, j] - (1.0 if a1 == b1 else 0.0)))
    # Theorem 1 with dynamics
    Ws = [haar_unitary(N, rng) for _ in range(3)]
    record("pair thm1", np.max(np.abs(proc.simulate(Ws, 0) - proc.pair_sum(Ws, 0))))
print("   ok:", {k: f"{v:.1e}" for k, v in worst.items()})

# general two-epoch formula with random marks and random (entangled) preparation
print("   general two-epoch formula F = <Phi0| U1^(a1')^dag U1^(a1) (x) U2^(a2')^dag U2^(a2) |Phi0>")
d1, d2, nc = 3, 2, 2
U1 = [haar_unitary(d1, rng) for _ in range(nc)]
U2 = [haar_unitary(d2, rng) for _ in range(nc)]
Phi0 = random_state(d1 * d2, rng)
proc = RecordProcess(2, {1: np.array([0, 1]), 2: np.array([0, 1])}, d1 * d2, Phi0,
                     {1: [np.kron(u, np.eye(d2)) for u in U1], 2: [np.kron(np.eye(d1), u) for u in U2]})
seqs, F = proc.influence_kernel()
for i, (a1, a2) in enumerate(seqs):
    for j, (b1, b2) in enumerate(seqs):
        G1 = U1[b1].conj().T @ U1[a1]
        G2 = U2[b2].conj().T @ U2[a2]
        expect = Phi0.conj() @ np.kron(G1, G2) @ Phi0
        record("two-epoch formula", abs(F[i, j] - expect))
# maximally entangled: F = Tr[G1 G2^T]/d
bell = np.zeros(4, dtype=complex); bell[0] = bell[3] = 1 / np.sqrt(2)
U1 = [haar_unitary(2, rng) for _ in range(2)]
U2 = [haar_unitary(2, rng) for _ in range(2)]
proc = RecordProcess(2, {1: np.array([0, 1]), 2: np.array([0, 1])}, 4, bell,
                     {1: [np.kron(u, np.eye(2)) for u in U1], 2: [np.kron(np.eye(2), u) for u in U2]})
seqs, F = proc.influence_kernel()
for i, (a1, a2) in enumerate(seqs):
    for j, (b1, b2) in enumerate(seqs):
        G1 = U1[b1].conj().T @ U1[a1]
        G2 = U2[b2].conj().T @ U2[a2]
        record("Bell formula Tr[G1 G2^T]/d", abs(F[i, j] - np.trace(G1 @ G2.T) / 2))
print("   ok:", f"{worst['two-epoch formula']:.1e}", f"{worst['Bell formula Tr[G1 G2^T]/d']:.1e}")

# recoherence and trace-distance backflow: no dynamics between the records
print("   recoherence: trace distance between |+> and |-> after each epoch (W_k = identity)")
cats = {1: np.array([0, 1]), 2: np.array([0, 1])}
plus = np.array([1, 1]) / np.sqrt(2)
minus = np.array([1, -1]) / np.sqrt(2)
I2 = np.eye(2)
for label, proc in [("Bell pair", bell_pair_process(2, cats, 2)), ("fresh sharp probes", fresh_probe_process(2, cats, 0.0))]:
    _, sp, _ = proc.simulate([I2, I2, I2], 0, return_states=True, psi_i=plus)
    _, sm, _ = proc.simulate([I2, I2, I2], 0, return_states=True, psi_i=minus)
    dists = [trace_distance(a, b) for a, b in zip(sp, sm)]
    print(f"     {label}: " + ", ".join(f"{x:.3f}" for x in dists))
    if label == "Bell pair":
        assert abs(dists[1] - 0) < 1e-12 and abs(dists[2] - 1) < 1e-12
        # no linear map: the two states coincide after epoch 1 and differ after epoch 2
        record("rho_1(+) - rho_1(-)", np.max(np.abs(sp[1] - sm[1])))
        assert np.max(np.abs(sp[2] - sm[2])) > 0.9
# partial revival with a rotation between the records
print("   partial revival with W_2 = R_y(phi) between the records (trace distance after epoch 2):")
for phi in [0.0, 0.5, 1.0, np.pi / 2]:
    R = np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])
    proc = bell_pair_process(2, cats, 2)
    _, sp, _ = proc.simulate([I2, R, I2], 0, return_states=True, psi_i=plus)
    _, sm, _ = proc.simulate([I2, R, I2], 0, return_states=True, psi_i=minus)
    print(f"     phi = {phi:.2f}: D_1 = {trace_distance(sp[1], sm[1]):.3f}, D_2 = {trace_distance(sp[2], sm[2]):.3f} (cos^2(phi/2) = {np.cos(phi/2)**2:.3f})")

# ---------------------------------------------------------------- (b) chain of pairs
print("(b) Chain of pairs: F = [all consecutive changes equal]")
for d, m in [(2, 3), (2, 4), (3, 3)]:
    cats = {t: np.arange(d) for t in range(1, m + 1)}
    proc = bell_chain_process(d, cats, d)
    seqs, F = proc.influence_kernel()
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            expect = 1.0 if all((a[k + 1] - a[k] - (b[k + 1] - b[k])) % d == 0 for k in range(m - 1)) else 0.0
            record(f"chain d={d} m={m}", abs(F[i, j] - expect))
    Ws = [haar_unitary(d, rng) for _ in range(m + 1)]
    record("chain thm1", np.max(np.abs(proc.simulate(Ws, 0) - proc.pair_sum(Ws, 0))))
    assert proc.kernel_restriction_causal()
print("   ok:", {k: f"{v:.1e}" for k, v in worst.items() if 'chain' in k})

# ---------------------------------------------------------------- (c) correlated failures: Frechet range
print("(c) Correlated intermittent records: bound 1 - mu(empty) between max kappa_t and min(1, sum kappa_t)")
epochs = [1, 2, 3]
subsets = [frozenset(S) for r in range(4) for S in itertools.combinations(epochs, r)]
for trial in range(200):
    w = rng.dirichlet(np.ones(len(subsets)) * rng.uniform(0.2, 2))
    mu = dict(zip(subsets, w))
    marg = [sum(p for S, p in mu.items() if t in S) for t in epochs]
    bound = 1 - mu[frozenset()]
    assert max(marg) - 1e-12 <= bound <= min(1, sum(marg)) + 1e-12
# the two extremes
kappas = [0.3, 0.2, 0.1]
como = 1 - (1 - max(kappas))  # all-or-nothing with the largest marginal... build comonotone coupling
# comonotone coupling: S = {t : kappa_t >= u} for uniform u
grid = np.linspace(0, 1, 100001)
mu_c = {}
for u in grid[:-1]:
    S = frozenset(t for t, k in zip(epochs, kappas) if k > u)
    mu_c[S] = mu_c.get(S, 0) + 1 / (len(grid) - 1)
print(f"   marginals {kappas}: comonotone bound = {1 - mu_c.get(frozenset(), 0):.3f} (= max kappa = {max(kappas)}),"
      f" independent = {1 - np.prod([1 - k for k in kappas]):.3f}, disjoint = {min(1, sum(kappas)):.3f}")

print("\nWORST DISCREPANCIES")
for k, v in worst.items():
    print(f"   {k}: {v:.2e}")
print("ALL PASS" if max(worst.values()) < 1e-8 else "CHECK FAILED")
