"""Checks for Theorem 1 (recording with memory), Theorem 2 (characterisation of influence
kernels), the latitude bound 1 - lambda_min(F), the finite-archive bound, and the invariance of
the realised distribution under post-processing of the records.

Run:  python influence_functional_check.py
All discrepancies printed are of the order of machine precision when the checks pass.
"""
import itertools
import numpy as np

from pimh_memory import (RecordProcess, haar_unitary, random_state, hadamard_matrix, tv,
                         fresh_probe_process, ledger_process, bell_pair_process,
                         chained_qubit_process, process_from_kernel, is_psd)

rng = np.random.default_rng(20261103)
worst = {}


def record(name, val):
    worst[name] = max(worst.get(name, 0.0), float(val))


def random_process(N, epochs, D, ncat, rng):
    cats = {t: rng.integers(0, ncat, size=N) for t in epochs}
    for t in epochs:  # make every category occur
        cats[t][:ncat] = np.arange(ncat)
        rng.shuffle(cats[t])
    marks = {t: [haar_unitary(D, rng) for _ in range(ncat)] for t in epochs}
    return RecordProcess(N, cats, D, random_state(D, rng), marks)


# ---------------------------------------------------------------- 1. Theorem 1 for random processes
print("1. Theorem 1: pair sum = explicit simulation (random marks, reused probe space)")
for trial in range(40):
    N = int(rng.integers(2, 4))
    calN = int(rng.integers(2, 5))
    inter = list(range(1, calN))
    m = int(rng.integers(1, min(3, len(inter)) + 1)) if inter else 0
    epochs = sorted(rng.choice(inter, size=m, replace=False).tolist()) if m else []
    if not epochs:
        continue
    ncat = int(rng.integers(2, N + 1))
    D = int(rng.integers(2, 5))
    proc = random_process(N, epochs, D, ncat, rng)
    Ws = [haar_unitary(N, rng) for _ in range(calN)]
    x_i = int(rng.integers(0, N))
    P_sim = proc.simulate(Ws, x_i)
    P_pair = proc.pair_sum(Ws, x_i)
    record("thm1 |P_sim - P_pair|", np.max(np.abs(P_sim - P_pair)))
    record("thm1 |sum P - 1|", abs(P_pair.sum() - 1))
    seqs, F = proc.influence_kernel()
    record("F unit diagonal", np.max(np.abs(np.diag(F) - 1)))
    record("F hermitian", np.max(np.abs(F - F.conj().T)))
    assert is_psd(F), "F not PSD"
    assert proc.kernel_restriction_causal(), "F not causal"
print("   ok:", {k: f"{v:.2e}" for k, v in worst.items()})

# ---------------------------------------------------------------- 2. fresh probes reproduce kappa^d
print("2. Fresh independent probes: F = kappa^{d_c}")
for kappa in [0.0, 0.3, 0.75, 1.0]:
    N, calN = 3, 4
    cats = {1: np.array([0, 1, 1]), 2: np.array([0, 1, 2]), 3: np.array([1, 0, 0])}
    proc = fresh_probe_process(N, cats, kappa)
    seqs, F = proc.influence_kernel()
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            d = sum(int(u != v) for u, v in zip(a, b))
            record("fresh F - kappa^d", abs(F[i, j] - (kappa ** d if not (kappa == 0 and d == 0) else 1.0)))
    Ws = [haar_unitary(N, rng) for _ in range(calN)]
    record("fresh thm1", np.max(np.abs(proc.simulate(Ws, 0) - proc.pair_sum(Ws, 0))))
print("   ok:", f"{worst['fresh F - kappa^d']:.2e}", f"{worst['fresh thm1']:.2e}")

# ---------------------------------------------------------------- 3. converse construction
print("3. Theorem 2 (converse): rebuild a process from its kernel and compare")
for trial in range(25):
    N = 3
    calN = 4
    epochs = [1, 2, 3][: int(rng.integers(1, 4))]
    ncat = 2
    D = int(rng.integers(2, 4))
    proc = random_process(N, epochs, D, ncat, rng)
    seqs, F = proc.influence_kernel()
    ncats = [proc.ncat[t] for t in proc.epochs]
    rebuilt = process_from_kernel(seqs, F, ncats, N, proc.cats)
    seqs2, F2 = rebuilt.influence_kernel()
    assert seqs2 == seqs
    record("converse |F2 - F|", np.max(np.abs(F2 - F)))
    Ws = [haar_unitary(N, rng) for _ in range(calN)]
    record("converse |P2 - P|", np.max(np.abs(rebuilt.simulate(Ws, 1) - proc.simulate(Ws, 1))))
    # the rebuilt probe space has dimension rank(F) <= |C|^m
    assert rebuilt.D <= len(seqs)
print("   ok:", f"{worst['converse |F2 - F|']:.2e}", f"{worst['converse |P2 - P|']:.2e}")

# a hand-made causal kernel that is a mixture of two processes' kernels (convex combinations are causal)
print("   mixture kernel (not itself produced by either process) is realised:")
N, calN, epochs = 2, 4, [1, 2, 3]
cats = {t: np.array([0, 1]) for t in epochs}
p1 = ledger_process(N, cats, D=3)
p2 = fresh_probe_process(N, cats, 0.4)
s1, F1 = p1.influence_kernel()
s2, F2 = p2.influence_kernel()
Fmix = 0.35 * F1 + 0.65 * F2
pm = process_from_kernel(s1, Fmix, [2, 2, 2], N, cats)
sm, Fm = pm.influence_kernel()
Ws = [hadamard_matrix(2) @ np.diag(np.exp(1j * rng.normal(size=2))) for _ in range(calN)]
P_mix_direct = 0.35 * p1.simulate(Ws, 0) + 0.65 * p2.simulate(Ws, 0)
record("mixture |Fm - Fmix|", np.max(np.abs(Fm - Fmix)))
record("mixture |P - P_mix|", np.max(np.abs(pm.simulate(Ws, 0) - P_mix_direct)))
print("   ok:", f"{worst['mixture |Fm - Fmix|']:.2e}", f"{worst['mixture |P - P_mix|']:.2e}")

# ---------------------------------------------------------------- 4. latitude bound
print("4. Latitude bound TV(P_F, P_delta) <= 1 - lambda_min(F)")
viol = 0
maxratio = 0.0
for trial in range(60):
    N = 2 if trial % 2 == 0 else 3
    calN = 4
    epochs = [1, 2, 3]
    cats = {t: np.arange(N) for t in epochs}  # injective resolution at all intermediate epochs
    kind = trial % 4
    if kind == 0:
        proc = random_process(N, epochs, int(rng.integers(2, 5)), N, rng)
        for t in epochs:
            proc.cats[t] = np.arange(N)
    elif kind == 1:
        proc = ledger_process(N, cats, D=int(rng.integers(2, 6)))
    elif kind == 2:
        proc = chained_qubit_process(2, {t: np.array([0, 1]) for t in epochs}, kappa=rng.uniform(), phi=rng.uniform(0, np.pi / 2))
        N = 2
    else:
        proc = fresh_probe_process(N, cats, rng.uniform())
    Ws = [haar_unitary(N, rng) for _ in range(calN)]
    x_i = 0
    P_F = proc.simulate(Ws, x_i)
    sharp = fresh_probe_process(N, {t: proc.cats[t] for t in epochs}, 0.0)
    P_delta = sharp.simulate(Ws, x_i)
    seqs, F = proc.influence_kernel()
    lam = np.min(np.linalg.eigvalsh((F + F.conj().T) / 2))
    bound = 1 - max(lam, 0.0)
    d = tv(P_F, P_delta)
    if d > bound + 1e-9:
        viol += 1
    if bound > 1e-12:
        maxratio = max(maxratio, d / bound)
    if kind == 3:  # product kernel: lambda_min = (1-kappa)^m exactly
        kappa = proc.influence_kernel()[1][0, 1].real if False else None
print(f"   violations: {viol}, max TV/bound = {maxratio:.3f}")
# product kernel eigenvalue
for kappa in [0.2, 0.5, 0.9]:
    proc = fresh_probe_process(3, {1: np.arange(3), 2: np.arange(3)}, kappa)
    seqs, F = proc.influence_kernel()
    lam = np.min(np.linalg.eigvalsh(F))
    record("product lambda_min - (1-kappa)^m", abs(lam - (1 - kappa) ** 2))
print("   product kernel: lambda_min = (1-kappa)^|R| with error", f"{worst['product lambda_min - (1-kappa)^m']:.2e}")

# ---------------------------------------------------------------- 5. finite archive
print("5. Finite archive: rank F <= D and off-diagonal mass bound")
for trial in range(30):
    N = 2
    epochs = [1, 2, 3]
    ncat = 2
    D = int(rng.integers(2, 6))
    proc = random_process(N, epochs, D, ncat, rng)
    seqs, F = proc.influence_kernel()
    M = len(seqs)
    rank = np.linalg.matrix_rank(F, tol=1e-9)
    assert rank <= D
    off = np.sum(np.abs(F) ** 2) - M
    bound = M * (M / D - 1)
    assert off >= bound - 1e-9, (off, bound)
    if D < M:
        assert np.min(np.linalg.eigvalsh(F)) < 1e-9
print("   ok")

# ---------------------------------------------------------------- 6. post-processing invariance
print("6. Operations on the probes after the last record leave P unchanged; between records they do not")
N, calN, epochs = 3, 4, [1, 2]
cats = {t: np.array([0, 1, 1]) for t in epochs}
proc = random_process(N, epochs, 3, 2, rng)
Ws = [haar_unitary(N, rng) for _ in range(calN)]
V = haar_unitary(3, rng)
P0 = proc.simulate(Ws, 0)
P1 = proc.simulate(Ws, 0, probe_post=V)
record("post-processing |P1 - P0|", np.max(np.abs(P1 - P0)))
# between records: modify the marks of epoch 2 by a common probe unitary before them
proc2 = RecordProcess(N, cats, 3, proc.Phi0, {1: proc.marks[1], 2: [u @ V for u in proc.marks[2]]})
P2 = proc2.simulate(Ws, 0)
print(f"   after last record: {worst['post-processing |P1 - P0|']:.2e};  between records: {np.max(np.abs(P2 - P0)):.3f} (nonzero expected)")

# ---------------------------------------------------------------- 7. blindness needs a complete sharp chronicle
print("7. Blindness: a kernel entry F[a,a'] != 0 (a != a') is exposed by a two-path interferometer")
# N = 2, epochs 1,2 recorded with c injective; choose steps so that exactly two paths with
# records a = (0,0) and a' = (1,1) reach x_f = 0, with a valuation phase theta between them.
H = hadamard_matrix(2)
for proc, label in [(ledger_process(2, {1: np.array([0, 1]), 2: np.array([0, 1])}, D=2), "parity ledger"),
                    (fresh_probe_process(2, {1: np.array([0, 1]), 2: np.array([0, 1])}, 0.0), "complete sharp chronicle")]:
    sens = 0.0
    for theta in np.linspace(0, 2 * np.pi, 9):
        W1 = H
        W2 = np.diag([1.0, np.exp(1j * theta)])  # stay put, valuation phase on configuration 1
        W3 = H
        P = proc.simulate([W1, W2, W3], 0)
        sens = max(sens, abs(P[0] - 0.5))
    print(f"   {label}: max |P(0) - 1/2| over valuations = {sens:.3f}")

# ---------------------------------------------------------------- 8. identification of F by two-path dynamics
print("8. Identification: F[a,a'] from two-path dynamics when every category has >= 2 configurations")


def two_path_steps(proc, a, b, x_i, x_f, theta, calN):
    """Steps W_1..W_calN such that exactly two paths with record sequences a and b reach x_f,
    with amplitudes 1/2 and e^{i theta}/2 (branch at step 1, recombine at step calN)."""
    N = proc.N
    reps = {}
    xs, ys = [x_i], [x_i]
    for k in range(1, calN):
        if k in proc.cats:
            i = proc.epochs.index(k)
            ca = np.where(proc.cats[k] == a[i])[0]
            cb = np.where(proc.cats[k] == b[i])[0]
            xk = int(ca[0])
            yk = int(cb[0]) if cb[0] != xk else int(cb[1])
        else:
            xk, yk = 0, 1
        xs.append(xk); ys.append(yk)
    xs.append(x_f); ys.append(x_f)
    Ws = []
    for k in range(1, calN + 1):
        W = np.zeros((N, N), dtype=complex)
        if k == 1:
            W[xs[1], x_i] = 1 / np.sqrt(2); W[ys[1], x_i] = np.exp(1j * theta) / np.sqrt(2)
            # complete: other inputs -> remaining outputs, plus the orthogonal combination
            others_in = [z for z in range(N) if z != x_i]
            others_out = [z for z in range(N) if z not in (xs[1], ys[1])]
            v2 = np.zeros(N, dtype=complex); v2[xs[1]] = 1 / np.sqrt(2); v2[ys[1]] = -np.exp(1j * theta) / np.sqrt(2)
            cols = [v2] + [np.eye(N)[:, z] for z in others_out]
            for z, col in zip(others_in, cols):
                W[:, z] = col
        elif k == calN:
            W[x_f, xs[k - 1]] = 1 / np.sqrt(2); W[x_f, ys[k - 1]] = 1 / np.sqrt(2)
            yf = [z for z in range(N) if z != x_f][0]
            W[yf, xs[k - 1]] = 1 / np.sqrt(2); W[yf, ys[k - 1]] = -1 / np.sqrt(2)
            others_in = [z for z in range(N) if z not in (xs[k - 1], ys[k - 1])]
            others_out = [z for z in range(N) if z not in (x_f, yf)]
            for z, o in zip(others_in, others_out):
                W[o, z] = 1
        else:
            W[xs[k], xs[k - 1]] = 1; W[ys[k], ys[k - 1]] = 1
            others_in = [z for z in range(N) if z not in (xs[k - 1], ys[k - 1])]
            others_out = [z for z in range(N) if z not in (xs[k], ys[k])]
            for z, o in zip(others_in, others_out):
                W[o, z] = 1
        assert np.allclose(W.conj().T @ W, np.eye(N)), "not unitary"
        Ws.append(W)
    return Ws


for trial in range(12):
    N = 4
    calN = 4
    epochs = [1, 2, 3]
    cats = {t: np.array([0, 0, 1, 1]) for t in epochs}  # every category has two configurations
    D = int(rng.integers(2, 4))
    marks = {t: [haar_unitary(D, rng) for _ in range(2)] for t in epochs}
    proc = RecordProcess(N, cats, D, random_state(D, rng), marks)
    seqs, F = proc.influence_kernel()
    for i, a in enumerate(seqs):
        for j, b in enumerate(seqs):
            if i >= j:
                continue
            vals = []
            for theta in [0.0, np.pi / 2]:
                Ws = two_path_steps(proc, a, b, 0, 0, theta, calN)
                vals.append(proc.simulate(Ws, 0)[0])
            # P = (1/2)(1 + Re(e^{-i theta} F[a,b]))  with path a having amplitude 1/2 and path b e^{i theta}/2
            est = complex(2 * vals[0] - 1, 2 * vals[1] - 1)
            record("identification |F_est - F|", abs(est - F[i, j]))
print(f"   ok: {worst['identification |F_est - F|']:.1e}")

print("\nWORST DISCREPANCIES")
for k, v in worst.items():
    print(f"   {k}: {v:.2e}")
print("ALL PASS" if max(v for k, v in worst.items() if 'between' not in k) < 1e-8 and viol == 0 else "CHECK FAILED")
