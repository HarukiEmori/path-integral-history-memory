"""Checks for the section on forward and backward amplitudes.

Fresh probes with the intermittent readout: at a recorded epoch the pointer is read and gives
'no record' with probability kappa or the category a with probability 1 - kappa (operations
Omega_0(rho) = kappa rho and Omega_a(rho) = (1-kappa) Pi_a rho Pi_a).

(1) Causality: the reduced state of the system at epoch k, and the forward distribution of the
    readout at epoch k, do not depend on the steps, valuations, records or probe operations after k.
(2) Retrodiction (past quantum state): Pr(r_t = r | r_{s != t}, x_f) is proportional to
    Tr[E_t Omega_r(rho_t)] with the forward state rho_t (conditioned on the earlier readouts) and
    the backward effect E_t (the final projector propagated backwards through the later steps and
    the adjoints of the later operations).
(3) At kappa = 0 with injective resolution and every intermediate epoch recorded, the retrodicted
    distribution is the forward-backward smoothing of the Markov chain and is independent of the
    valuations; at kappa > 0 it depends on the valuations of epochs after t, whereas the forward
    distribution does not.
(4) The ABL rule for an unrecorded epoch equals the conditional distribution of a sharp record made
    there (consistency of the definitions).
"""
import itertools
import numpy as np

from pimh_memory import haar_unitary, hadamard_matrix, fresh_probe_process, tv

rng = np.random.default_rng(7)
worst = {}


def record(name, val):
    worst[name] = max(worst.get(name, 0.0), float(val))


def readout_joint(N, cats, kappa, Ws, x_i):
    """Joint distribution of the intermittent readouts (0 = no record, a+1 = category a) at the
    recorded epochs and of the final configuration, by explicit operations on density matrices."""
    epochs = sorted(cats)
    calN = len(Ws)
    outcomes = {}
    ncat = {t: int(np.max(cats[t])) + 1 for t in epochs}
    for rs in itertools.product(*[range(ncat[t] + 1) for t in epochs]):
        rho = np.zeros((N, N), dtype=complex)
        rho[x_i, x_i] = 1.0
        for k, W in enumerate(Ws, start=1):
            rho = W @ rho @ W.conj().T
            if k in cats:
                r = rs[epochs.index(k)]
                if r == 0:
                    rho = kappa * rho
                else:
                    Pi = np.diag((cats[k] == r - 1).astype(float))
                    rho = (1 - kappa) * Pi @ rho @ Pi
        for x_f in range(N):
            outcomes[rs + (x_f,)] = float(np.real(rho[x_f, x_f]))
    return outcomes


# ---------------------------------------------------------------- (1) causality
print("(1) Causality: states and forward readout statistics at epoch k do not depend on later operations")
N, calN = 3, 4
cats = {1: np.array([0, 1, 1]), 2: np.array([0, 0, 1])}
kappa = 0.4
Ws = [haar_unitary(N, rng) for _ in range(calN)]
proc = fresh_probe_process(N, cats, kappa)
_, states, _ = proc.simulate(Ws, 0, return_states=True)
Ws2 = Ws[:2] + [haar_unitary(N, rng) for _ in range(2)]  # different later steps
proc2 = fresh_probe_process(N, {1: cats[1], 2: cats[2], 3: np.array([0, 1, 2])}, kappa)  # an extra later record
_, states2, _ = proc2.simulate(Ws2, 0, return_states=True)
for k in range(3):
    record("rho_k independent of the future", np.max(np.abs(states[k] - states2[k])))
J1 = readout_joint(N, cats, kappa, Ws, 0)
J2 = readout_joint(N, {1: cats[1], 2: cats[2], 3: np.array([0, 1, 2])}, kappa, Ws2, 0)
# forward distribution of the readout at epoch 1
p1 = np.zeros(3)
p2 = np.zeros(3)
for key, v in J1.items():
    p1[key[0]] += v
for key, v in J2.items():
    p2[key[0]] += v
record("forward readout independent of the future", np.max(np.abs(p1 - p2)))
print(f"   ok: {worst['rho_k independent of the future']:.1e}, {worst['forward readout independent of the future']:.1e}")

# ---------------------------------------------------------------- (2) past quantum state
print("(2) Past quantum state formula against the brute-force conditional distribution")
N, calN = 3, 5
cats = {1: np.array([0, 1, 1]), 2: np.array([0, 1, 2]), 3: np.array([1, 0, 0])}
epochs = sorted(cats)
for kappa in [0.0, 0.35, 0.8]:
    Ws = [haar_unitary(N, rng) for _ in range(calN)]
    J = readout_joint(N, cats, kappa, Ws, 0)
    for t in epochs:
        ncat = int(np.max(cats[t])) + 1
        for key in list(J.keys()):
            if key[-1] != 1:
                continue
            x_f = key[-1]
            others = {s: key[epochs.index(s)] for s in epochs if s != t}
            # brute force
            denom = sum(J[k2] for k2 in J if k2[-1] == x_f and all(k2[epochs.index(s)] == others[s] for s in others))
            if denom < 1e-12:
                continue
            brute = np.array([J[tuple(others.get(s, r) if s != t else r for s in epochs) + (x_f,)] / denom for r in range(ncat + 1)])
            # forward state up to and including the step at t (before the record at t)
            rho = np.zeros((N, N), dtype=complex); rho[0, 0] = 1
            for k in range(1, t + 1):
                rho = Ws[k - 1] @ rho @ Ws[k - 1].conj().T
                if k in cats and k != t:
                    r = others[k]
                    if r == 0:
                        rho = kappa * rho
                    else:
                        Pi = np.diag((cats[k] == r - 1).astype(float)); rho = (1 - kappa) * Pi @ rho @ Pi
            # backward effect from the final projector through the later steps and operations
            E = np.zeros((N, N), dtype=complex); E[x_f, x_f] = 1
            for k in range(calN, t, -1):
                if k in cats and k != t:
                    r = others[k]
                    if r == 0:
                        E = kappa * E
                    else:
                        Pi = np.diag((cats[k] == r - 1).astype(float)); E = (1 - kappa) * Pi @ E @ Pi
                E = Ws[k - 1].conj().T @ E @ Ws[k - 1]
            # Omega_r(rho_t)
            vals = []
            for r in range(ncat + 1):
                if r == 0:
                    om = kappa * rho
                else:
                    Pi = np.diag((cats[t] == r - 1).astype(float)); om = (1 - kappa) * Pi @ rho @ Pi
                vals.append(np.real(np.trace(E @ om)))
            vals = np.array(vals)
            pqs = vals / vals.sum()
            record("past quantum state", np.max(np.abs(pqs - brute)))
print(f"   ok: {worst['past quantum state']:.1e}")

# ---------------------------------------------------------------- (3) valuation dependence
print("(3) Dependence of the forward and the retrodicted distributions on later valuations")
N, calN = 2, 4
H = hadamard_matrix(2)
cats = {1: np.array([0, 1]), 2: np.array([0, 1]), 3: np.array([0, 1])}
epochs = [1, 2, 3]
A = np.array([[np.cos(0.7), np.sin(0.7)], [-np.sin(0.7), np.cos(0.7)]])  # a non-Hadamard accessibility


def steps(th1, th2, th3, th4):
    out = []
    for th in (th1, th2, th3, th4):
        out.append(A @ np.diag([1, np.exp(1j * th)]))
    return out


for kappa in [0.0, 0.5]:
    base = steps(0.3, 0.9, 1.4, 0.2)
    later = steps(0.3, 0.9, 2.6, 2.9)  # valuations of epochs 3 and 4 changed
    J0 = readout_joint(N, cats, kappa, base, 0)
    J1 = readout_joint(N, cats, kappa, later, 0)
    t = 1
    # forward distribution of r_1
    f0 = np.zeros(3); f1 = np.zeros(3)
    for key, v in J0.items():
        f0[key[0]] += v
    for key, v in J1.items():
        f1[key[0]] += v
    # retrodicted distribution of r_1 given x_f = 0 (marginal over the other readouts)
    def retro(J):
        num = np.zeros(3)
        for key, v in J.items():
            if key[-1] == 0:
                num[key[0]] += v
        return num / num.sum()
    r0, r1 = retro(J0), retro(J1)
    print(f"   kappa = {kappa}: forward Pr(r_1) changes by {np.max(np.abs(f0 - f1)):.2e}; retrodicted Pr(r_1 | x_f=0) changes by {np.max(np.abs(r0 - r1)):.3f}")
    record("forward r_1 independent of later valuations", np.max(np.abs(f0 - f1)))
    if kappa == 0.0:
        record("kappa=0 retrodiction independent of valuations", np.max(np.abs(r0 - r1)))
        # equals forward-backward smoothing of the Markov chain
        M = np.abs(A) ** 2  # transition matrix (column y -> row x)
        # Pr(x_1 | x_f = 0) with x_1 = r_1 - 1 (sharp records at epochs 1,2,3), chain of 4 steps from x_0 = 0
        fwd = M[:, 0]  # Pr(x_1)
        # backward: Pr(x_f = 0 | x_1) = (M^3)[0, x_1]
        bwd = np.linalg.matrix_power(M, 3)[0, :]
        smooth = fwd * bwd / np.sum(fwd * bwd)
        record("kappa=0 smoothing", np.max(np.abs(r0[1:] - smooth)))
    else:
        assert np.max(np.abs(r0 - r1)) > 1e-3
print("   ok")

# ---------------------------------------------------------------- (4) ABL rule
print("(4) ABL rule equals the conditional distribution of a sharp record at the unrecorded epoch")
N = 3
Ws = [haar_unitary(N, rng) for _ in range(4)]
t = 2
U_before = Ws[1] @ Ws[0]
U_after = Ws[3] @ Ws[2]
for x_f in range(N):
    abl = np.array([abs(U_after[x_f, x] * U_before[x, 0]) ** 2 for x in range(N)])
    abl /= abl.sum()
    J = readout_joint(N, {t: np.arange(N)}, 0.0, Ws, 0)
    cond = np.array([J[(x + 1, x_f)] for x in range(N)])
    cond /= cond.sum()
    record("ABL", np.max(np.abs(abl - cond)))
print(f"   ok: {worst['ABL']:.1e}")

print("\nWORST DISCREPANCIES")
for k, v in worst.items():
    print(f"   {k}: {v:.2e}")
print("ALL PASS" if max(worst.values()) < 1e-8 else "CHECK FAILED")
