"""Shared tools for the numerical checks of the manuscript
"Records with Memory: A Non-Markovian Extension of a Path Integral Model of History"
(H. Emori and A. Iriki).

The basic model (Emori and Iriki, "A Path Integral Model of History"): a system on C^X with
|X| = N, one-step unitaries W_k, and records of the category c_t(x) of the configuration at the
recorded epochs t in R.  Here a *record process* is given by

  * a probe (archive) space C^D with an initial unit vector Phi0,
  * for every recorded epoch t and every category a a unitary U_t^{(a)} on C^D,

and the recording unitary at epoch t is  sum_a Pi_a^{(t)} (x) U_t^{(a)},  where Pi_a^{(t)} projects on
the configurations of category a.  Fresh independent probes, reused registers (ledgers),
correlated preparations and probe-probe interactions between records are all special cases.

Main formula (Theorem 1 of the manuscript):
  P(x_f) = sum_{paths x, x'} F[c(x), c(x')] phi[x] conj(phi[x']),
  F[a, a'] = <Phi_{a'} | Phi_a>,   Phi_a = U_{t_m}^{(a_m)} ... U_{t_1}^{(a_1)} Phi0.

Only numpy is required.  Dimensions are kept small, all paths are enumerated.
"""
from __future__ import annotations

import itertools
import numpy as np

# ----------------------------------------------------------------------------- utilities


def haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def random_state(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    return v / np.linalg.norm(v)


def hadamard_matrix(N: int) -> np.ndarray:
    """Discrete Fourier transform, a complex Hadamard matrix of size N."""
    x = np.arange(N)
    return np.exp(2j * np.pi * np.outer(x, x) / N) / np.sqrt(N)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(rho - sigma)
    return 0.5 * float(np.sum(np.abs(ev)))


def is_psd(M: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.min(np.linalg.eigvalsh((M + M.conj().T) / 2)) > -tol)


# ----------------------------------------------------------------------------- record processes


class RecordProcess:
    """A record process on a probe space of dimension D.

    N        : number of configurations
    cats     : dict epoch -> integer array of length N with the category of each configuration
    D        : probe dimension
    Phi0     : initial probe vector (length D)
    marks    : dict epoch -> list (over categories) of D x D unitaries
    """

    def __init__(self, N, cats, D, Phi0, marks):
        self.N = N
        self.cats = {int(t): np.asarray(v, dtype=int) for t, v in cats.items()}
        self.epochs = sorted(self.cats)
        self.D = D
        self.Phi0 = np.asarray(Phi0, dtype=complex)
        self.marks = {int(t): [np.asarray(u, dtype=complex) for u in us] for t, us in marks.items()}
        self.ncat = {t: len(self.marks[t]) for t in self.epochs}
        for t in self.epochs:
            assert self.cats[t].max() < self.ncat[t]

    # --- record sequences ----------------------------------------------------------------
    def sequences(self, upto: int | None = None):
        eps = self.epochs if upto is None else [t for t in self.epochs if t <= upto]
        return list(itertools.product(*[range(self.ncat[t]) for t in eps]))

    def probe_vector(self, seq) -> np.ndarray:
        v = self.Phi0.copy()
        for t, a in zip(self.epochs, seq):
            v = self.marks[t][a] @ v
        return v

    def influence_kernel(self):
        """Return (sequences, F) with F[i, j] = <Phi_{seq_j} | Phi_{seq_i}>."""
        seqs = self.sequences()
        V = np.array([self.probe_vector(s) for s in seqs])  # rows = Phi_a
        F = V.conj() @ V.T  # F[i,j] = sum_k conj(V[i,k]) V[j,k] = <Phi_i|Phi_j>
        # we want F[a, a'] = <Phi_{a'}|Phi_a>  = conj(<Phi_a|Phi_{a'}>)
        return seqs, F.conj()

    # --- explicit joint simulation ----------------------------------------------------------
    def recording_unitary(self, t: int) -> np.ndarray:
        N, D = self.N, self.D
        U = np.zeros((N * D, N * D), dtype=complex)
        for x in range(N):
            a = self.cats[t][x]
            U[x * D:(x + 1) * D, x * D:(x + 1) * D] = self.marks[t][a]
        return U

    def simulate(self, Ws, x_i, probe_post=None, return_states=False, psi_i=None):
        """Explicit evolution of system (x) probe.  Ws[k-1] is the step W_k, k = 1..calN.
        Records are applied after the step of the same epoch.  Returns the final distribution of
        the configuration and, optionally, the reduced system states after every epoch.
        The initial system state is the configuration x_i, or the vector psi_i if given."""
        N, D = self.N, self.D
        if psi_i is None:
            psi_i = np.zeros(N, dtype=complex)
            psi_i[x_i] = 1.0
        psi = np.kron(np.asarray(psi_i, dtype=complex), self.Phi0)
        states = [self._reduced(psi)]
        for k, W in enumerate(Ws, start=1):
            psi = np.kron(W, np.eye(D)) @ psi
            if k in self.cats:
                psi = self.recording_unitary(k) @ psi
            states.append(self._reduced(psi))
        if probe_post is not None:
            psi = np.kron(np.eye(N), probe_post) @ psi
        P = np.array([np.sum(np.abs(psi[x * D:(x + 1) * D]) ** 2) for x in range(N)])
        if return_states:
            return P, states, psi
        return P

    def _reduced(self, psi):
        M = psi.reshape(self.N, self.D)
        return M @ M.conj().T

    # --- pair sum -----------------------------------------------------------------------
    def pair_sum(self, Ws, x_i: int, F_override=None):
        """P(x_f) from Theorem 1.  Paths are enumerated; the amplitude of each path is grouped
        by its final configuration and its record sequence."""
        N = self.N
        calN = len(Ws)
        seqs, F = self.influence_kernel() if F_override is None else F_override
        index = {s: i for i, s in enumerate(seqs)}
        Psi = np.zeros((N, len(seqs)), dtype=complex)  # Psi[x_f, a] = sum of amplitudes
        for inner in itertools.product(range(N), repeat=calN - 1):
            path = (x_i,) + inner
            amp_common = 1.0 + 0j
            for k in range(1, calN):
                amp_common *= Ws[k - 1][path[k], path[k - 1]]
            seq = tuple(int(self.cats[t][path[t]]) for t in self.epochs)
            for x_f in range(N):
                amp = amp_common * Ws[calN - 1][x_f, path[calN - 1]]
                Psi[x_f, index[seq]] += amp
        P = np.real(np.einsum('xa,ab,xb->x', Psi, F, Psi.conj()))
        return P

    # --- divisibility helpers ---------------------------------------------------------------
    def kernel_restriction_causal(self, tol=1e-10) -> bool:
        """Check (K2): F[(b,d),(b',d)] does not depend on the common tail d."""
        seqs, F = self.influence_kernel()
        idx = {s: i for i, s in enumerate(seqs)}
        m = len(self.epochs)
        for i in range(1, m):
            for b in itertools.product(*[range(self.ncat[t]) for t in self.epochs[:i]]):
                for b2 in itertools.product(*[range(self.ncat[t]) for t in self.epochs[:i]]):
                    vals = []
                    for d in itertools.product(*[range(self.ncat[t]) for t in self.epochs[i:]]):
                        vals.append(F[idx[b + d], idx[b2 + d]])
                    if np.max(np.abs(np.array(vals) - vals[0])) > tol:
                        return False
        return True


# ----------------------------------------------------------------------------- constructors


def fresh_probe_process(N, cats, kappa):
    """The record process of the basic model: one fresh probe per recorded epoch with pointer
    states <eta_a|eta_b> = kappa for a != b (realised as eta_a = sqrt(kappa)|0> + sqrt(1-kappa)|a>)."""
    epochs = sorted(cats)
    ncat = {t: int(np.max(cats[t])) + 1 for t in epochs}
    dims = [ncat[t] + 1 for t in epochs]
    D = int(np.prod(dims))
    # pointer unitary R^(a) on one probe: maps |0> -> eta_a, extended to a unitary
    def local_mark(nc, a):
        d = nc + 1
        eta = np.zeros(d, dtype=complex)
        eta[0] = np.sqrt(kappa)
        eta[a + 1] = np.sqrt(1 - kappa)
        # unitary whose first column is eta: complete eta by an orthonormal basis of its complement
        u, s, vh = np.linalg.svd(eta.reshape(d, 1), full_matrices=True)
        comp = u[:, 1:]  # orthonormal basis of the orthogonal complement of eta
        q = np.hstack([eta.reshape(d, 1), comp])
        assert np.allclose(q.conj().T @ q, np.eye(d))
        return q
    marks = {}
    for i, t in enumerate(epochs):
        us = []
        for a in range(ncat[t]):
            ops = [np.eye(dims[j], dtype=complex) for j in range(len(epochs))]
            ops[i] = local_mark(ncat[t], a)
            U = ops[0]
            for o in ops[1:]:
                U = np.kron(U, o)
            us.append(U)
        marks[t] = us
    Phi0 = np.zeros(D, dtype=complex)
    Phi0[0] = 1.0
    return RecordProcess(N, cats, D, Phi0, marks)


def ledger_process(N, cats, D, mu=None):
    """A reused D-level counter: the mark of category a is the shift T^{mu_a} (mod D)."""
    epochs = sorted(cats)
    T = np.roll(np.eye(D), 1, axis=0)  # T|j> = |j+1 mod D>
    marks = {}
    for t in epochs:
        nc = int(np.max(cats[t])) + 1
        m = list(range(nc)) if mu is None else mu
        marks[t] = [np.linalg.matrix_power(T, int(m[a]) % D) for a in range(nc)]
    Phi0 = np.zeros(D, dtype=complex)
    Phi0[0] = 1.0
    return RecordProcess(N, cats, D, Phi0, marks)


def rotation_ledger_process(N, cats, theta):
    """A reused qubit rotated by R_y(theta) for every recorded epoch in category 1 (two categories)."""
    epochs = sorted(cats)
    Ry = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    marks = {t: [np.eye(2, dtype=complex), Ry] for t in epochs}
    return RecordProcess(N, cats, 2, np.array([1, 0], dtype=complex), marks)


def bell_pair_process(N, cats, d=2):
    """Two recorded epochs; the probes are two d-level systems in the maximally entangled state and
    the mark of category a is the shift X^a on the half of the respective epoch."""
    epochs = sorted(cats)
    assert len(epochs) == 2
    X = np.roll(np.eye(d), 1, axis=0)
    I = np.eye(d)
    Phi0 = np.zeros(d * d, dtype=complex)
    for j in range(d):
        Phi0[j * d + j] = 1 / np.sqrt(d)
    marks = {
        epochs[0]: [np.kron(np.linalg.matrix_power(X, a), I) for a in range(d)],
        epochs[1]: [np.kron(I, np.linalg.matrix_power(X, a)) for a in range(d)],
    }
    return RecordProcess(N, cats, d * d, Phi0, marks)


def chained_qubit_process(N, cats, kappa, phi):
    """Fresh qubit probes (two categories) with a partial swap exp(-i phi SWAP) between the probe
    of the previous recorded epoch and the new probe before the new probe records."""
    epochs = sorted(cats)
    m = len(epochs)
    D = 2 ** m
    Phi0 = np.zeros(D, dtype=complex)
    Phi0[0] = 1.0
    eta1 = np.array([kappa, np.sqrt(1 - kappa ** 2)], dtype=complex)
    R = np.array([[eta1[0], -eta1[1]], [eta1[1], eta1[0]]], dtype=complex)  # R|0> = eta1, R|1> = ...
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    S = np.cos(phi) * np.eye(4) - 1j * np.sin(phi) * SWAP

    def embed(op, qubits):
        # op acts on the listed qubits (ordered), identity elsewhere; qubit 0 is the most significant
        full = np.zeros((D, D), dtype=complex)
        others = [q for q in range(m) if q not in qubits]
        for idx in range(D):
            bits = [(idx >> (m - 1 - q)) & 1 for q in range(m)]
            sub_in = 0
            for q in qubits:
                sub_in = 2 * sub_in + bits[q]
            for sub_out in range(2 ** len(qubits)):
                amp = op[sub_out, sub_in]
                if amp == 0:
                    continue
                nb = bits.copy()
                for j, q in enumerate(reversed(qubits)):
                    nb[q] = (sub_out >> j) & 1
                out = 0
                for q in range(m):
                    out = 2 * out + nb[q]
                full[out, idx] += amp
        return full

    marks = {}
    for i, t in enumerate(epochs):
        pre = np.eye(D, dtype=complex) if i == 0 else embed(S, [i - 1, i])
        marks[t] = [pre, embed(R, [i]) @ pre]
    return RecordProcess(N, cats, D, Phi0, marks)


def correlated_failure_process(N, cats, mu):
    """Classically correlated intermittent records realised as a *mixture*: mu is a dict
    frozenset(unrecorded epochs) -> probability.  Returns a list of (prob, RecordProcess), each
    process having sharp records at the recorded epochs and inert probes elsewhere (two categories:
    marks X^a on probes prepared in |0> (sharp) or |+> (inert))."""
    epochs = sorted(cats)
    m = len(epochs)
    D = 2 ** m
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    comps = []
    for S, p in mu.items():
        Phi0 = np.array([1.0 + 0j])
        for t in epochs:
            Phi0 = np.kron(Phi0, np.array([1, 1]) / np.sqrt(2) if t in S else np.array([1, 0]))
        marks = {}
        for i, t in enumerate(epochs):
            us = []
            for a in range(2):
                ops = [I2] * m
                ops[i] = np.linalg.matrix_power(X, a)
                U = ops[0]
                for o in ops[1:]:
                    U = np.kron(U, o)
                us.append(U)
            marks[t] = us
        comps.append((p, RecordProcess(N, cats, D, Phi0, marks)))
    return comps


# ----------------------------------------------------------------------------- converse construction


def process_from_kernel(seqs, F, ncats, N, cats, tol=1e-9):
    """Theorem 2 (converse): build a record process whose influence kernel is the given causal
    correlation kernel F (indexed by the record sequences seqs).  Gram vectors of F are taken in
    C^r (r = rank F); prefix vectors use the default tail (0,...,0); the map
    Phi_b -> Phi_{(b,a)} is an isometry on the span of the level-(i-1) vectors and is extended to a
    unitary of C^r."""
    M = len(seqs)
    idx = {s: i for i, s in enumerate(seqs)}
    m = len(seqs[0])
    # Gram vectors: F = V^dagger-ish; we need <Phi_{a'}|Phi_a> = F[a, a'] -> vectors v_a with
    # sum_k conj(v_{a'}[k]) v_a[k] = F[a,a'], i.e. V V^H = F (rows v_a)
    G = F
    w, Q = np.linalg.eigh((G + G.conj().T) / 2)
    keep = w > tol
    r = int(np.sum(keep))
    V = Q[:, keep] * np.sqrt(w[keep])  # rows: v_a (length r), V V^H = G
    epochs = sorted(cats)

    def vec(prefix):
        tail = tuple(0 for _ in range(m - len(prefix)))
        return V[idx[tuple(prefix) + tail]]

    marks = {}
    for i in range(1, m + 1):
        t = epochs[i - 1]
        prefixes = list(itertools.product(*[range(ncats[j]) for j in range(i - 1)]))
        A = np.array([vec(p) for p in prefixes]).T  # columns: Phi_b
        us = []
        for a in range(ncats[i - 1]):
            B = np.array([vec(p + (a,)) for p in prefixes]).T
            # find unitary U with U A = B (A, B have equal Gram matrices)
            # solve on the span: U = B A^+ on range(A), then complete unitarily
            Ua, s, Vh = np.linalg.svd(A, full_matrices=False)
            rank = int(np.sum(s > tol))
            P_range = Ua[:, :rank]
            Apinv = Vh[:rank].conj().T @ np.diag(1 / s[:rank]) @ Ua[:, :rank].conj().T
            Upart = B @ Apinv  # maps range(A) -> range(B); isometric there
            # complete: U = Upart P_range + W (1 - P_range) with W an isometry from range(A)^perp to range(B)^perp
            Qa = Ua[:, :rank]
            Qb_full, _ = np.linalg.qr(np.hstack([B @ Vh[:rank].conj().T, np.eye(r)]))
            # orthonormal basis of range(B)^perp: complement of first `rank` columns
            Qb = Qb_full[:, :rank]
            Pa_perp = np.eye(r) - Qa @ Qa.conj().T
            Pb_perp = np.eye(r) - Qb @ Qb.conj().T
            # bases of the complements
            wa, ea = np.linalg.eigh(Pa_perp)
            wb, eb = np.linalg.eigh(Pb_perp)
            Ea = ea[:, wa > 0.5]
            Eb = eb[:, wb > 0.5]
            U = Upart @ (Qa @ Qa.conj().T) + Eb @ Ea.conj().T
            assert np.allclose(U.conj().T @ U, np.eye(r), atol=1e-8), "not unitary"
            assert np.allclose(U @ A, B, atol=1e-8), "isometry does not map prefixes"
            us.append(U)
        marks[t] = us
    Phi0 = vec(())
    return RecordProcess(N, cats, r, Phi0, marks)


def bell_chain_process(N, cats, d=2):
    """m recorded epochs and m-1 maximally entangled pairs; pair i joins epochs t_i and t_{i+1}.
    At epoch t_i the mark of category a applies X^{a} to the first half of pair i and X^{a} to the
    second half of pair i-1.  Since (X^{a} (x) X^{b}) |Phi+> = sum_j |j+a, j+b> depends on b - a only,
    pair i ends in a state that depends on a_{i+1} - a_i: the record keeps the sequence of changes
    and nothing else."""
    epochs = sorted(cats)
    m = len(epochs)
    assert m >= 2
    X = np.roll(np.eye(d), 1, axis=0)
    Xinv = X.T
    I = np.eye(d)
    bell = np.zeros(d * d, dtype=complex)
    for j in range(d):
        bell[j * d + j] = 1 / np.sqrt(d)
    Phi0 = np.array([1.0 + 0j])
    for _ in range(m - 1):
        Phi0 = np.kron(Phi0, bell)
    D = d ** (2 * (m - 1))

    def op_on(slots):  # slots: dict slot index (0..2(m-1)-1) -> d x d matrix
        U = np.array([[1.0 + 0j]])
        for s_ in range(2 * (m - 1)):
            U = np.kron(U, slots.get(s_, I))
        return U

    marks = {}
    for i, t in enumerate(epochs):
        us = []
        for a in range(d):
            slots = {}
            if i < m - 1:
                slots[2 * i] = np.linalg.matrix_power(X, a)          # first half of pair i
            if i > 0:
                slots[2 * (i - 1) + 1] = np.linalg.matrix_power(X, a)  # second half of pair i-1
            us.append(op_on(slots))
        marks[t] = us
    return RecordProcess(N, cats, D, Phi0, marks)
