"""The account of an epoch under an archive with memory (joint past quantum state).

For a record process (Definition 1), the account of the category a at a recorded epoch t, given the final
configuration x_f and the outcome m of a measurement of the archive after the last record, is

   Pr(a | m, x_f) = <psi_t| (Pi_a (x) 1) E_t (Pi_a (x) 1) |psi_t> / <psi_t| E_t |psi_t>,

where psi_t is the joint state of system and archive after the step and the record of epoch t and E_t is the
joint effect  |x_f><x_f| (x) |m><m|  carried backward through the later steps and records by their adjoints.
It is the probability that a sharp record of the category, inserted at t, would have shown a.

Checks:
  1. The formula equals the brute-force conditional probability obtained by inserting an actual sharp probe at
     epoch t, before or after the record of epoch t (both agree, by the non-demolition of the records).
  2. Summed over m and x_f it is the forward probability of a at epoch t (no dependence on later epochs).
  3. For the sharp fresh archive (F = delta) read in its pointer basis, the account is the smoothing of the
     Markov chain and does not depend on the valuations; read in a scrambled basis it does.
  4. For a counter (memory) the account depends on the basis in which the archive is read.
Only numpy is required.
"""
from __future__ import annotations

import numpy as np

from pimh_memory import RecordProcess, haar_unitary, fresh_probe_process, ledger_process

rng = np.random.default_rng(411)
report = []


def check(name, ok, detail=""):
    report.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))


def forward_states(proc, Ws, x_i):
    """Joint pure states psi_k after the step and the record of epoch k (k = 0..calN)."""
    N, D = proc.N, proc.D
    psi = np.zeros(N * D, dtype=complex)
    psi[x_i * D:(x_i + 1) * D] = proc.Phi0
    states = [psi]
    for k, W in enumerate(Ws, start=1):
        psi = np.kron(W, np.eye(D)) @ psi
        if k in proc.cats:
            psi = proc.recording_unitary(k) @ psi
        states.append(psi)
    return states


def backward_effect(proc, Ws, t, x_f, basis_vector):
    """E_t: the projector |x_f><x_f| (x) |m><m| carried back through W_calN, U_{calN-1}, ..., U_{t+1}, W_{t+1}."""
    N, D = proc.N, proc.D
    ex = np.zeros(N); ex[x_f] = 1
    E = np.outer(np.kron(ex, basis_vector), np.kron(ex, basis_vector).conj())
    calN = len(Ws)
    for k in range(calN, t, -1):
        if k in proc.cats and k < calN:
            U = proc.recording_unitary(k)
            E = U.conj().T @ E @ U
        Wk = np.kron(Ws[k - 1], np.eye(D))
        E = Wk.conj().T @ E @ Wk
    return E


def account_formula(proc, Ws, x_i, t, basis):
    """Returns table[a, m, x_f] = Pr(a, m, x_f) (joint, unnormalised conditional numerators) for a sharp
    counterfactual record of the configuration at t (injective resolution assumed: a = configuration)."""
    N, D = proc.N, proc.D
    psi_t = forward_states(proc, Ws, x_i)[t]
    table = np.zeros((N, D, N))
    for m in range(D):
        for x_f in range(N):
            E = backward_effect(proc, Ws, t, x_f, basis[:, m])
            for a in range(N):
                Pa = np.zeros((N, N)); Pa[a, a] = 1
                proj = np.kron(Pa, np.eye(D))
                v = proj @ psi_t
                table[a, m, x_f] = np.real(v.conj() @ E @ v)
    return table


def account_bruteforce(proc, Ws, x_i, t, basis, insert_before_record):
    """Insert an actual sharp probe (N-level, shift marks) at epoch t and read everything at the end."""
    N, D = proc.N, proc.D
    Dp = D * N  # archive (x) probe
    psi = np.zeros(N * Dp, dtype=complex)
    e0 = np.zeros(N, dtype=complex); e0[0] = 1
    psi[x_i * Dp:(x_i + 1) * Dp] = np.kron(proc.Phi0, e0)
    # sharp probe unitary: |x>|0> -> |x>|x>
    Up = np.zeros((N * Dp, N * Dp), dtype=complex)
    for x in range(N):
        S = np.roll(np.eye(N), x, axis=0)
        Up[x * Dp:(x + 1) * Dp, x * Dp:(x + 1) * Dp] = np.kron(np.eye(D), S)
    for k, W in enumerate(Ws, start=1):
        psi = np.kron(W, np.eye(Dp)) @ psi
        if k == t and insert_before_record:
            psi = Up @ psi
        if k in proc.cats:
            U = proc.recording_unitary(k)  # on system (x) archive; extend by the probe
            Ue = np.zeros((N * Dp, N * Dp), dtype=complex)
            for x in range(N):
                for y in range(N):
                    Ue[y * Dp:(y + 1) * Dp, x * Dp:(x + 1) * Dp] = np.kron(U[y * D:(y + 1) * D, x * D:(x + 1) * D], np.eye(N))
            psi = Ue @ psi
        if k == t and not insert_before_record:
            psi = Up @ psi
    # read: system in configuration basis, archive in `basis`, probe in computational basis
    amp = psi.reshape(N, D, N)                      # [x_f, archive, probe]
    amp = np.einsum('xdp,dm->xmp', amp, basis.conj())  # archive in the given basis
    table = np.zeros((N, D, N))
    for a in range(N):
        for m in range(D):
            for x_f in range(N):
                table[a, m, x_f] = abs(amp[x_f, m, a]) ** 2
    return table


def conditional(table):
    tot = table.sum(axis=0, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        c = np.where(tot > 1e-12, table / tot, np.nan)
    return c


N, calN = 2, 4
cats = {1: [0, 1], 2: [0, 1], 3: [0, 1]}
procs = {'counter D=3': ledger_process(N, cats, 3), 'sharp fresh (kappa=0)': fresh_probe_process(N, cats, 0.0),
         'fresh kappa=0.6': fresh_probe_process(N, cats, 0.6)}

# 1 and 2
worst_formula = 0.0
worst_forward = 0.0
for name, proc in procs.items():
    for _ in range(4):
        Ws = [haar_unitary(N, rng) for _ in range(calN)]
        x_i = int(rng.integers(N))
        basis = haar_unitary(proc.D, rng)
        for t in (1, 2, 3):
            tf = account_formula(proc, Ws, x_i, t, basis)
            tb = account_bruteforce(proc, Ws, x_i, t, basis, True)
            ta = account_bruteforce(proc, Ws, x_i, t, basis, False)
            worst_formula = max(worst_formula, np.nanmax(np.abs(tf - tb)), np.nanmax(np.abs(tf - ta)))
            psi_t = forward_states(proc, Ws, x_i)[t]
            fwd = np.array([np.sum(np.abs(psi_t.reshape(N, proc.D)[a]) ** 2) for a in range(N)])
            worst_forward = max(worst_forward, np.max(np.abs(tf.sum(axis=(1, 2)) - fwd)))
check("1 account formula = brute force with an inserted sharp probe, before and after the record (3 archives)",
      worst_formula < 1e-12, "max diff %.1e" % worst_formula)
check("2 summed over the readout and the final configuration it is the forward probability", worst_forward < 1e-12,
      "max diff %.1e" % worst_forward)

# 3 sharp fresh archive: pointer basis -> Markov smoothing, no valuation dependence; scrambled basis -> dependence
proc = procs['sharp fresh (kappa=0)']
As = [haar_unitary(N, rng) for _ in range(calN)]
x_i = 0
t = 2


def account_with_valuations(proc, basis, v):
    Ws = [A @ np.diag(np.exp(1j * v[k])) for k, A in enumerate(As)]
    return conditional(account_formula(proc, Ws, x_i, t, basis))


vals = [rng.uniform(0, 2 * np.pi, size=(calN, N)) for _ in range(5)]
pointer = np.eye(proc.D)
accs = [account_with_valuations(proc, pointer, v) for v in vals]
dev_pointer = max(np.nanmax(np.abs(accs[0] - a)) for a in accs[1:])
# Markov smoothing: Pr(x_t | x_{t-1}, x_{t+1}) proportional to M_{t+1}(x_{t+1}|x_t) M_t(x_t|x_{t-1})
Ws0 = [A @ np.diag(np.exp(1j * vals[0][k])) for k, A in enumerate(As)]
M = [np.abs(W) ** 2 for W in Ws0]
# pointer basis index m encodes the three sharp records (a1, a2, a3) in the fresh-probe process; find them
# by reading the archive vectors
_, _, seqs = None, None, proc.sequences()
smooth_ok = True
for m in range(proc.D):
    # which record sequence does pointer state m correspond to? compare with probe vectors
    match = [s for s in seqs if abs(proc.probe_vector(s)[m]) > 0.5]
    if not match:
        continue
    (a1, a2, a3) = match[0]
    for x_f in range(N):
        tf = accs[0][:, m, x_f]
        if np.all(np.isnan(tf)):
            continue
        pred = np.array([M[2][a3, a] * M[1][a, a1] for a in range(N)])
        pred = pred / pred.sum()
        # the account of epoch 2 must be concentrated on a2 (the record is sharp) -- the smoothing formula
        # applies to the counterfactual at an epoch; with a sharp record at t the readout already fixes a2
        if abs(tf[a2] - 1) > 1e-9:
            smooth_ok = False
check("3a sharp archive read in its pointer basis: the account repeats the record and does not depend on the valuations",
      smooth_ok and dev_pointer < 1e-12, "max deviation %.1e" % dev_pointer)
# an unrecorded epoch under the sharp archive at the other epochs: Markov smoothing
cats_gap = {1: [0, 1], 3: [0, 1]}
proc_gap = fresh_probe_process(N, cats_gap, 0.0)
accs_gap = [conditional(account_formula(proc_gap, [A @ np.diag(np.exp(1j * v[k])) for k, A in enumerate(As)], x_i, 2, np.eye(proc_gap.D))) for v in vals]
dev_gap = max(np.nanmax(np.abs(accs_gap[0] - a)) for a in accs_gap[1:])
smooth_ok = True
seqs_gap = proc_gap.sequences()
for m in range(proc_gap.D):
    match = [s for s in seqs_gap if abs(proc_gap.probe_vector(s)[m]) > 0.5]
    if not match:
        continue
    (a1, a3) = match[0]
    for x_f in range(N):
        tf = accs_gap[0][:, m, x_f]
        if np.all(np.isnan(tf)):
            continue
        pred = np.array([M[2][a3, a] * M[1][a, a1] for a in range(N)])
        pred = pred / pred.sum()
        smooth_ok &= np.allclose(tf, pred, atol=1e-10)
check("3b unrecorded epoch between sharp records: the account is the Markov smoothing, valuation-free",
      smooth_ok and dev_gap < 1e-12, "max deviation %.1e" % dev_gap)
scr = haar_unitary(proc.D, rng)
accs_s = [account_with_valuations(proc, scr, v) for v in vals]
dev_scr = max(np.nanmax(np.abs(accs_s[0] - a)) for a in accs_s[1:])
check("3c the same sharp archive read in a scrambled basis: the account depends on the valuations", dev_scr > 0.05,
      "max deviation %.3f" % dev_scr)
# 4 counter: dependence on the reading basis
proc = procs['counter D=3']
fourier = np.exp(2j * np.pi * np.outer(np.arange(3), np.arange(3)) / 3) / np.sqrt(3)
acc_c = account_with_valuations(proc, np.eye(3), vals[0])
acc_f = account_with_valuations(proc, fourier, vals[0])
dev_basis = np.nanmax(np.abs(acc_c - acc_f))
check("4 counter with memory: the account depends on the basis in which the archive is read", dev_basis > 0.05,
      "max difference %.3f" % dev_basis)

# 5 analytic example: N = 2, calN = 3, a sharp record at epoch 1 only, Hadamard accessibility, valuation phase theta
# on configuration 1 at step 2; the record is read in the basis (|0> +- |1>)/sqrt2; the account of the unrecorded
# epoch 2 given the outcome +- is Pr(a_2 = 0 | +-, x_f) = (1 +- cos theta)/2, whereas read in the pointer basis it is
# the Markov smoothing 1/2 (Proposition 12).
cats1 = {1: [0, 1]}
proc1 = fresh_probe_process(2, cats1, 0.0)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
worst = 0.0
worst_pointer = 0.0
for theta in (0.3, 1.1, 2.0, 2.9):
    Ws = [H, H @ np.diag([1, np.exp(1j * theta)]), H]
    B1 = np.eye(3, dtype=complex)
    B1[:, 1] = np.array([0, 1, 1]) / np.sqrt(2)   # +
    B1[:, 2] = np.array([0, 1, -1]) / np.sqrt(2)  # -
    tab = conditional(account_formula(proc1, Ws, 0, 2, B1))
    for sign, m in ((+1, 1), (-1, 2)):
        for x_f in range(2):
            val = tab[0, m, x_f]
            if not np.isnan(val):
                worst = max(worst, abs(val - (1 + sign * np.cos(theta)) / 2))
    tabp = conditional(account_formula(proc1, Ws, 0, 2, np.eye(3, dtype=complex)))
    for m in (1, 2):
        for x_f in range(2):
            val = tabp[0, m, x_f]
            if not np.isnan(val):
                worst_pointer = max(worst_pointer, abs(val - 0.5))
check("5 analytic example: Pr(a_2 = 0 | +-, x_f) = (1 +- cos theta)/2 for the record read in the +- basis, 1/2 in its pointer basis",
      worst < 1e-12 and worst_pointer < 1e-12, "max errors %.1e / %.1e" % (worst, worst_pointer))

print()
print("ALL PASS" if all(ok for _, ok in report) else "SOME CHECKS FAILED")
