"""Identification of the influence functional from the realised distributions, and hidden memory.

Checks for the manuscript (Sec. 4.4, identification theorem, and the blind archive with memory):

  A. The two-configuration archive "flag with a phase" (a qubit flipped by configuration 1 at epochs 1 and 3,
     phase-marked at epoch 2, plus a sharp record of epoch 2) has F != delta but leaves the course blind:
     its realised distribution equals the classical sum (the complete sharp chronicle) for every choice of
     the steps, and does not depend on the valuations.
  B. Rank of the real-linear map  F -> (realised distributions under all steps):
       N >= 3: the map is injective on Hermitian kernels (full rank), with injective and with coarse resolutions;
       N = 2 : the null space is spanned by the "hidden" tensors predicted by the parity rule
               (two off-diagonal factors enclosing an odd number of d^- factors).
  C. The explicit multi-block interferometer (sinks) for N = 3 and two blocks of consecutive differences:
       exactly the 2^n courses x_s reach x_f, P(x_f; theta) = 4^{-n} sum_{s,s'} F[x_s, x_s'] e^{i(s-s').theta}, and the
       discrete Fourier transform over the 3^n grid of phases recovers F[a, a'] (n = 2).

Only numpy is required.
"""
from __future__ import annotations

import itertools
import numpy as np

from pimh_memory import RecordProcess, haar_unitary, tv, fresh_probe_process

rng = np.random.default_rng(20261104)
TOL = 1e-10
report = []


def check(name, ok, detail=""):
    report.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))


# ----------------------------------------------------------------------------- helpers


def chronicle_amplitudes(Ws, x_i, N, cats):
    """Psi[x_f, idx(chronicle)] = sum of the amplitudes of the paths with that chronicle.
    cats: dict epoch -> array of categories; all intermediate epochs must be in cats."""
    calN = len(Ws)
    epochs = sorted(cats)
    assert epochs == list(range(1, calN))
    ncat = [int(np.max(cats[t])) + 1 for t in epochs]
    chrons = list(itertools.product(*[range(n) for n in ncat]))
    idx = {c: i for i, c in enumerate(chrons)}
    Psi = np.zeros((N, len(chrons)), dtype=complex)
    for p in itertools.product(range(N), repeat=calN - 1):
        path = (x_i,) + p
        amp = 1 + 0j
        for k in range(1, calN):
            amp *= Ws[k - 1][path[k], path[k - 1]]
        c = tuple(int(cats[t][path[t]]) for t in epochs)
        for x_f in range(N):
            Psi[x_f, idx[c]] += amp * Ws[calN - 1][x_f, path[calN - 1]]
    return chrons, Psi


def real_row(C):
    """The functional F -> sum_{a,a'} F[a,a'] C[a,a'] on Hermitian F, as a real row vector in the
    coordinates (diag F, Re F[a<a'], Im F[a<a'])."""
    M = C.shape[0]
    iu = np.triu_indices(M, 1)
    return np.concatenate([np.real(np.diag(C)), 2 * np.real(C[iu]), -2 * np.imag(C[iu])])


def hermitian_coords(H):
    M = H.shape[0]
    iu = np.triu_indices(M, 1)
    return np.concatenate([np.real(np.diag(H)), np.real(H[iu]), np.imag(H[iu])])


def data_matrix(N, calN, cats, n_exp):
    rows = []
    for _ in range(n_exp):
        Ws = [haar_unitary(N, rng) for _ in range(calN)]
        x_i = int(rng.integers(N))
        chrons, Psi = chronicle_amplitudes(Ws, x_i, N, cats)
        for x_f in range(N):
            C = np.outer(Psi[x_f], Psi[x_f].conj())  # C[a,a'] = Psi[a] conj(Psi[a'])
            rows.append(real_row(C))
    return chrons, np.array(rows)


def classical_sum(Ws, x_i):
    N = Ws[0].shape[0]
    p = np.zeros(N)
    p[x_i] = 1.0
    for W in Ws:
        p = (np.abs(W) ** 2) @ p
    return p


# ----------------------------------------------------------------------------- A. the blind archive

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
cats2 = {1: [0, 1], 2: [0, 1], 3: [0, 1]}
marks_flag = {1: [np.kron(I2, I2), np.kron(X, I2)],
              2: [np.kron(I2, I2), np.kron(Z, X)],   # phase on the flag, sharp record of epoch 2 on the second qubit
              3: [np.kron(I2, I2), np.kron(X, I2)]}
Phi0 = np.zeros(4, dtype=complex)
Phi0[0] = 1.0
flag = RecordProcess(2, cats2, 4, Phi0, marks_flag)
seqs, F = flag.influence_kernel()
idx = {s: i for i, s in enumerate(seqs)}
Fpred = np.eye(len(seqs), dtype=complex)
for (a1, b, a3) in seqs:
    Fpred[idx[(a1, b, a3)], idx[(1 - a1, b, 1 - a3)]] = (-1) ** b
check("A1 flag archive: F = delta + (-1)^b on the pairs (a1,b,a3),(1-a1,b,1-a3)", np.allclose(F, Fpred, atol=TOL))
check("A1' flag archive: F differs from delta (lambda_min = %.2f)" % np.min(np.linalg.eigvalsh(F)),
      not np.allclose(F, np.eye(len(seqs))))
sharp = fresh_probe_process(2, cats2, 0.0)
worst = 0.0
worst_val = 0.0
for _ in range(40):
    Ws = [haar_unitary(2, rng) for _ in range(4)]
    x_i = int(rng.integers(2))
    P = flag.simulate(Ws, x_i)
    worst = max(worst, tv(P, classical_sum(Ws, x_i)), tv(P, sharp.simulate(Ws, x_i)))
    # valuation independence for fixed accessibilities
    As = Ws
    Ps = []
    for _ in range(4):
        v = rng.uniform(0, 2 * np.pi, size=(4, 2))
        Ps.append(flag.simulate([A @ np.diag(np.exp(1j * v[k])) for k, A in enumerate(As)], x_i))
    worst_val = max(worst_val, max(tv(Ps[0], Pj) for Pj in Ps[1:]))
check("A2 flag archive is blind: P = classical sum = sharp chronicle for 40 random step sequences", worst < 1e-12,
      "max TV %.1e" % worst)
check("A3 flag archive: no dependence on the valuations", worst_val < 1e-12, "max TV %.1e" % worst_val)

# ----------------------------------------------------------------------------- B. ranks


def hidden_tensors_N2(m):
    """Basis tensors h in {d, d-, o01, o10}^m whose off-diagonal factors enclose an odd number of d- factors."""
    fac = {
        'd': np.array([[1, 0], [0, 1]], dtype=complex),
        'dm': np.array([[1, 0], [0, -1]], dtype=complex),
        'o01': np.array([[0, 1], [0, 0]], dtype=complex),
        'o10': np.array([[0, 0], [1, 0]], dtype=complex),
    }
    hidden = []
    for h in itertools.product(['d', 'dm', 'o01', 'o10'], repeat=m):
        pos = [k for k, f in enumerate(h) if f in ('o01', 'o10')]
        if len(pos) < 2:
            continue
        bad = False
        for p, q in zip(pos, pos[1:]):
            if sum(1 for k in range(p + 1, q) if h[k] == 'dm') % 2 == 1:
                bad = True
        if bad:
            T = np.array([[1.0 + 0j]])
            for f in h:
                T = np.kron(T, fac[f])   # index order: chronicle a = (a_1..a_m) with a_1 most significant
            hidden.append(T)
    return hidden


def rank_report(N, calN, cats, n_exp, expect_nullity=None, label=""):
    chrons, R = data_matrix(N, calN, cats, n_exp)
    M = len(chrons)
    dim = M * M
    s = np.linalg.svd(R, compute_uv=False)
    rank = int(np.sum(s > 1e-9 * s[0]))
    nullity = dim - rank
    print(f"   {label}: M = {M} chronicles, real dimension {dim}, rows {R.shape[0]}, rank {rank}, nullity {nullity}")
    return chrons, R, rank, nullity


# N = 3, injective, calN = 4
_, _, rank, nullity = rank_report(3, 4, {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}, 330, label="N=3 injective, 3 epochs")
check("B1 N = 3, injective resolutions: F is determined (null space trivial)", nullity == 0)
# N = 3, coarse resolution at epoch 2
_, _, rank, nullity = rank_report(3, 4, {1: [0, 1, 2], 2: [0, 0, 1], 3: [0, 1, 2]}, 160, label="N=3 coarse at epoch 2")
check("B2 N = 3, coarse resolution at one epoch: F is determined", nullity == 0)
# N = 2, two epochs: everything determined
_, _, rank, nullity = rank_report(2, 3, {1: [0, 1], 2: [0, 1]}, 40, label="N=2, 2 epochs")
check("B3 N = 2, two recorded epochs: F is determined", nullity == 0)
# N = 2, three epochs: nullity 4, spanned by o (x) d- (x) o
chrons, R, rank, nullity = rank_report(2, 4, cats2, 80, label="N=2, 3 epochs")
hidden = hidden_tensors_N2(3)
check("B4 N = 2, three recorded epochs: nullity = number of hidden tensors = %d" % len(hidden), nullity == len(hidden))
# chronicle order of data_matrix is itertools.product -> a_1 most significant, same as kron order
resid = max(np.linalg.norm(R @ hermitian_coords((H + H.conj().T) / 2)) + np.linalg.norm(R @ hermitian_coords(1j * (H - H.conj().T) / 2))
            for H in hidden)
check("B5 the hidden tensors are invisible to all realised distributions", resid < 1e-9, "residual %.1e" % resid)
# N = 2, four epochs: nullity predicted by the parity rule
chrons, R, rank, nullity = rank_report(2, 5, {1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [0, 1]}, 320, label="N=2, 4 epochs")
hidden4 = hidden_tensors_N2(4)
resid4 = max(np.linalg.norm(R @ hermitian_coords((H + H.conj().T) / 2)) for H in hidden4)
check("B6 N = 2, four recorded epochs: nullity = %d hidden tensors (parity rule), all invisible" % len(hidden4),
      nullity == len(hidden4) and resid4 < 1e-9, "residual %.1e" % resid4)
# the visible part: the average of F over the configurations of a gap is determined (one explicit functional)
# pairs (0,b,0),(1,b,1): the functional F[(0,0,0),(1,0,1)] + F[(0,1,0),(1,1,1)] must be in the row space,
# the difference must not.
chrons3, R3, _, _ = rank_report(2, 4, cats2, 80, label="N=2, 3 epochs (again)")
i3 = {c: i for i, c in enumerate(chrons3)}
def unit(a, b):
    H = np.zeros((8, 8), dtype=complex); H[i3[a], i3[b]] = 1; H[i3[b], i3[a]] = 1
    return hermitian_coords(H)
v_sum = unit((0, 0, 0), (1, 0, 1)) + unit((0, 1, 0), (1, 1, 1))
v_dif = unit((0, 0, 0), (1, 0, 1)) - unit((0, 1, 0), (1, 1, 1))
proj = np.linalg.pinv(R3) @ R3   # projector on the row space (in coordinates)
in_row = lambda v: np.linalg.norm(proj @ v - v) < 1e-8
check("B7 N = 2: the sum over the gap configuration of F[(0,b,0),(1,b,1)] is determined, the difference is not",
      in_row(v_sum) and not in_row(v_dif))

# ----------------------------------------------------------------------------- C. the multi-block interferometer


def interferometer_two_blocks_N3(theta1, theta2):
    """Steps for N = 3 and the pair a = (0,0,2,0,0), a' = (1,1,2,1,1): split at epoch 1 with phase theta1,
    merge at epoch 3 into the common configuration 2 (the antisymmetric branch goes to the sink 0),
    split again at epoch 4 with phase theta2 (the antisymmetric combination is sourced from the empty
    configuration 1, the sink moves to 2), merge at the end into x_f = 0 (sink to 1, old sink stays at 2)."""
    r = 1 / np.sqrt(2)
    e1, e2 = np.exp(1j * theta1), np.exp(1j * theta2)
    W1 = np.array([[r, r, 0], [r * e1, -r * e1, 0], [0, 0, 1]], dtype=complex)   # columns: images of |0>,|1>,|2>
    W2 = np.eye(3, dtype=complex)
    W3 = np.array([[r, -r, 0], [0, 0, 1], [r, r, 0]], dtype=complex)             # |0>->(|2>+|0>)/r2, |1>->(|2>-|0>)/r2, |2>->|1>
    W4 = np.array([[0, r, r], [0, -r * e2, r * e2], [1, 0, 0]], dtype=complex)  # |2>->(|0>+e2|1>)/r2, |1>->(|0>-e2|1>)/r2, |0>->|2>
    W5 = np.eye(3, dtype=complex)
    W6 = np.array([[r, r, 0], [r, -r, 0], [0, 0, 1]], dtype=complex)
    for W in (W1, W3, W4, W6):
        assert np.allclose(W.conj().T @ W, np.eye(3))
    return [W1, W2, W3, W4, W5, W6]


cats3 = {t: [0, 1, 2] for t in range(1, 6)}
D = 3
marks = {t: [haar_unitary(D, rng) for _ in range(3)] for t in range(1, 6)}
Phi0 = np.zeros(D, dtype=complex)
Phi0[0] = 1
proc3 = RecordProcess(3, cats3, D, Phi0, marks)
seqs3, F3 = proc3.influence_kernel()
i3 = {s: i for i, s in enumerate(seqs3)}
# the four courses reaching x_f = 0: branch choices s = (s1, s2)
course = {(0, 0): (0, 0, 2, 0, 0), (0, 1): (0, 0, 2, 1, 1), (1, 0): (1, 1, 2, 0, 0), (1, 1): (1, 1, 2, 1, 1)}
a, ap = course[(0, 0)], course[(1, 1)]
Fa = F3[i3[a], i3[ap]]
worst_four = 0.0
grid = [0, 2 * np.pi / 3, 4 * np.pi / 3]
acc = 0j
for th1 in grid:
    for th2 in grid:
        P = proc3.simulate(interferometer_two_blocks_N3(th1, th2), 0)
        pred = 0.0
        for s, y in course.items():
            for s2, y2 in course.items():
                ph = np.exp(1j * ((s[0] - s2[0]) * th1 + (s[1] - s2[1]) * th2))
                pred += (1 / 16) * F3[i3[y], i3[y2]] * ph
        worst_four = max(worst_four, abs(P[0] - np.real(pred)))
        acc += np.exp(1j * (th1 + th2)) * P[0]
F_recovered = (16 / 9) * acc
check("C1 N = 3, two blocks: exactly the four courses x_s reach x_f, P = 4^{-2} sum_{s,s'} F e^{i(s-s').theta}",
      worst_four < 1e-12, "max error %.1e" % worst_four)
check("C1' the discrete Fourier transform over the 3 x 3 grid of phases recovers F[a,a'] (|F| = %.3f)" % abs(Fa),
      abs(F_recovered - Fa) < 1e-12, "error %.1e" % abs(F_recovered - Fa))
# the same pair would be invisible to a two-path circuit on N = 2? (sanity of the one-block formula for N = 2)
cats2b = {1: [0, 1], 2: [0, 1]}
proc2 = RecordProcess(2, cats2b, 3, Phi0, {t: [haar_unitary(3, rng) for _ in range(2)] for t in (1, 2)})
seqs2, F2 = proc2.influence_kernel()
j2 = {s: i for i, s in enumerate(seqs2)}
worst = 0.0
for theta in np.linspace(0, 2 * np.pi, 7):
    r = 1 / np.sqrt(2)
    W1 = np.array([[r, r], [r * np.exp(1j * theta), -r * np.exp(1j * theta)]])
    W2 = np.eye(2)
    W3 = np.array([[r, r], [r, -r]])
    P = proc2.simulate([W1, W2, W3], 0)
    pred = 0.5 * (1 + np.real(np.exp(-1j * theta) * F2[j2[(0, 0)], j2[(1, 1)]]))
    worst = max(worst, abs(P[0] - pred))
check("C2 one block (two-path interferometer of Proposition 3): P = (1/2)[1 + Re(e^{-i theta} F)]", worst < 1e-12)

print()
print("ALL PASS" if all(ok for _, ok in report) else "SOME CHECKS FAILED")
