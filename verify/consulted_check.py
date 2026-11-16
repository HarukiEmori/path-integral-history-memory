"""Consulted archives: an independent numerical check of the proposition on consulted record processes.

A consulted record process marks an archive as a record process does, and every step after the first
recorded epoch is applied as sum_nu W_k^(nu) (x) Q_nu^(t(k)), where {Q_nu^(t)} is a projective reading of the
archive after the record of epoch t and t(k) is the last recorded epoch before k.  The joint state of
system (x) archive is evolved explicitly (brute force) and compared with the statements.

Checks:
  1. (i)   P(x_f) = sum_{x,x',nu,nu'} <Phi^{nu'}_{c(x')}|Phi^nu_{c(x)}> phi_nu[x] conj(phi_nu'[x']) for random archives,
           readings and consulted steps (unrecorded intermediate epochs included); with every reading copied on a
           fresh probe only the terms nu = nu' remain.
  2. (ii)  Definite readings (counters, entangled pair, permutation register (x) fuzzy probes, copying probes):
           P = sum F[c(x),c(x')] phi_s[x] conj(phi_s[x']); as a record with back-action relative to reference steps
           the transcript kernel is F[a,a'] beta(r) conj(beta(r')).
  3. (iii) TV(P, P_{delta,s}) <= 1 - lambda_min(F) over 300 random definite instances (P_{delta,s}: fresh sharp
           probes read through the same s_t, simulated explicitly).
  4. (iv)  Complete chronicle, injective resolutions: product formula, blindness to the valuations, realisation
           of random laws p on courses as the joint law of (chronicle, final configuration).
  5. Examples (a) consulted counter, (b) records lost together and consulted, (c) copying probes,
     (d) entangled pair with P(0) = (1 + cos theta)/2 against 1/2.
  6. The commutant of {W (x) conj(W)} is span{1, P_Omega} for N = 2, 3, 4 (Haar samples; the proof's generators).
  7. A reading that is not definite restores interference: a sharp probe consulted in its Fourier basis by a
     valuation-only step gives P(0) = (1 + cos theta)/2, against 1/2 for the pointer reading.
Only numpy is required.
"""
from __future__ import annotations

import functools
import itertools
import time
import numpy as np

from pimh_memory import haar_unitary, hadamard_matrix, tv

rng = np.random.default_rng(1115)
report = []
T0 = time.time()
prod = itertools.product

def check(name, ok, detail=""):
    report.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))

kron_all = lambda ops: functools.reduce(np.kron, ops, np.eye(1, dtype=complex))
e = lambda d, j: np.eye(d, dtype=complex)[j]                                  # basis vector |j> of C^d
proj = lambda d, j: np.outer(e(d, j), e(d, j))
shift = lambda d, k=1: np.roll(np.eye(d, dtype=complex), int(k) % d, axis=0)  # |j> -> |j + k mod d>

def step(N, A=None):   # W = A exp(iV): A Haar unless given, V a random valuation
    return (haar_unitary(N, rng) if A is None else A) @ np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, N)))

class Consulted:
    """cats: epoch -> category array; marks: epoch -> [U^(a)]; reads: epoch -> [Q_nu].
    steps[k-1] is a list of W_k^(nu) over the outcomes nu of the reading at t(k) (one entry if k <= t_1)."""

    def __init__(self, N, cats, D, Phi0, marks, reads):
        self.N, self.D, self.Phi0, self.marks, self.reads = N, D, np.asarray(Phi0, dtype=complex), marks, reads
        self.cats = {int(t): np.asarray(c, dtype=int) for t, c in cats.items()}
        self.epochs = sorted(self.cats)
        self.ncat = {t: len(marks[t]) for t in self.epochs}
        self.nout = {t: len(reads[t]) for t in self.epochs}
        assert all(np.allclose(sum(reads[t]), np.eye(D)) for t in self.epochs)

    def t_of(self, k):
        prev = [t for t in self.epochs if t < k]
        return prev[-1] if prev else None

    def random_steps(self, calN):
        return [[step(self.N) for _ in range(1 if self.t_of(k) is None else self.nout[self.t_of(k)])] for k in range(1, calN + 1)]

    def record_op(self, t, extra=1):
        return sum(np.kron(proj(self.N, x), np.kron(self.marks[t][self.cats[t][x]], np.eye(extra))) for x in range(self.N))

    def simulate(self, steps, x_i, copy=False, joint=False):
        """Brute-force evolution of system (x) archive (x) [fresh probes onto which every reading is copied]."""
        N, D, eps = self.N, self.D, self.epochs
        cd = [self.nout[t] for t in eps] if copy else []
        Dc = int(np.prod(cd))
        psi = np.kron(e(N, x_i), np.kron(self.Phi0, e(Dc, 0)))
        for k, Wk in enumerate(steps, start=1):
            t = self.t_of(k)
            Qs = [np.eye(D)] if t is None else self.reads[t]
            psi = sum(np.kron(Wk[nu], np.kron(Q, np.eye(Dc))) for nu, Q in enumerate(Qs)) @ psi
            if k in self.cats:
                psi = self.record_op(k, Dc) @ psi
                if copy:
                    i = eps.index(k)
                    C = sum(np.kron(Q, kron_all([shift(cd[j], nu) if j == i else np.eye(cd[j]) for j in range(len(cd))]))
                            for nu, Q in enumerate(self.reads[k]))
                    psi = np.kron(np.eye(N), C) @ psi
        M = psi.reshape(N, D * Dc)
        return (np.sum(np.abs(M) ** 2, axis=1), M) if joint else np.sum(np.abs(M) ** 2, axis=1)

    def chron(self, path):  # path = (x_1, ..., x_calN)
        return tuple(int(self.cats[t][path[t - 1]]) for t in self.epochs)

    def amp(self, steps, path, nu, x_i):
        A, prev = 1 + 0j, x_i
        for k, xk in enumerate(path, start=1):
            t = self.t_of(k)
            A *= steps[k - 1][0 if t is None else nu[self.epochs.index(t)]][xk, prev]
            prev = xk
        return A

    def archive_vector(self, a, nu=None):
        """Phi_a^nu of Eq. (readvector); nu = None gives the mark-only vector Phi_a (a may be a prefix)."""
        v = self.Phi0
        for i, (t, at) in enumerate(zip(self.epochs, a)):
            v = self.marks[t][at] @ v
            v = v if nu is None else self.reads[t][nu[i]] @ v
        return v

    def formula_i(self, steps, x_i):
        """(P from the double sum over (x,nu),(x',nu'), P from the terms nu = nu', largest |sum of the nu != nu' terms|)."""
        N, calN = self.N, len(steps)
        nus = list(prod(*[range(self.nout[t]) for t in self.epochs]))
        cache, P, Pd, cross = {}, np.zeros(N), np.zeros(N), 0.0
        for xf in range(N):
            vecs, lab = [], []
            for inner in prod(range(N), repeat=calN - 1):
                path = inner + (xf,)
                a = self.chron(path)
                for j, nu in enumerate(nus):
                    if (a, nu) not in cache:
                        cache[(a, nu)] = self.archive_vector(a, nu)
                    vecs.append(self.amp(steps, path, nu, x_i) * cache[(a, nu)])
                    lab.append(j)
            V = np.array(vecs)
            G = (V.conj() @ V.T).T      # G[i,j] = <Phi_j|Phi_i> phi_i conj(phi_j), i = (x,nu), j = (x',nu')
            same = np.equal.outer(lab, lab)
            P[xf], Pd[xf], cross = np.real(G.sum()), np.real(G[same].sum()), max(cross, abs(G[~same].sum()))
        return P, Pd, cross

    def kernel(self):
        seqs = list(prod(*[range(self.ncat[t]) for t in self.epochs]))
        V = np.array([self.archive_vector(a) for a in seqs])
        return seqs, V @ V.conj().T     # F[i,j] = <Phi_j|Phi_i>

    def definite_s(self):
        """s_t read off the mark-only archive states; returns (s, largest violation of definiteness)."""
        s, res = {}, 0.0
        for i, t in enumerate(self.epochs):
            s[t] = {}
            for pre in prod(*[range(self.ncat[u]) for u in self.epochs[:i + 1]]):
                v = self.archive_vector(pre)
                w = [Q @ v for Q in self.reads[t]]
                nu = int(np.argmax([np.linalg.norm(x) for x in w]))
                res = max(res, np.linalg.norm(w[nu] - v), *[np.linalg.norm(x) for j, x in enumerate(w) if j != nu])
                s[t][pre] = nu
        return s, res

    def formula_ii(self, steps, x_i, s, F=None):
        seqs, F0 = self.kernel()
        F = F0 if F is None else F
        idx = {a: i for i, a in enumerate(seqs)}
        Psi = np.zeros((self.N, len(seqs)), dtype=complex)
        for path in prod(range(self.N), repeat=len(steps)):
            a = self.chron(path)
            Psi[path[-1], idx[a]] += self.amp(steps, path, tuple(s[t][a[:i + 1]] for i, t in enumerate(self.epochs)), x_i)
        return np.real(np.einsum('xa,ab,xb->x', Psi, F, Psi.conj()))

# ----------------------------------------------------------------------------- archives

def make_cats(N, R, kind, nc=2):
    if kind == 'injective':
        return {t: rng.permutation(N) for t in R}
    return {t: rng.permutation(np.concatenate([np.arange(nc), rng.integers(nc, size=N - nc)])) for t in R}

def ncats(cats):
    return [int(cats[t].max()) + 1 for t in sorted(cats)]

def random_archive(N, cats, D):
    reads = {}
    for t in cats:
        nout = int(rng.integers(2, min(3, D) + 1))
        lab = rng.permutation(np.concatenate([np.arange(nout), rng.integers(nout, size=D - nout)]))
        U = haar_unitary(D, rng)
        reads[t] = [U[:, lab == j] @ U[:, lab == j].conj().T for j in range(nout)]
    marks = {t: [haar_unitary(D, rng) for _ in range(int(cats[t].max()) + 1)] for t in cats}
    return Consulted(N, cats, D, haar_unitary(D, rng)[:, 0], marks, reads)

def counter(N, cats, D, mu=None):
    mu = rng.integers(D, size=max(ncats(cats))) if mu is None else mu
    marks = {t: [shift(D, mu[a]) for a in range(int(cats[t].max()) + 1)] for t in cats}
    return Consulted(N, cats, D, e(D, 0), marks, {t: [proj(D, j) for j in range(D)] for t in cats}), mu

def perm_fuzzy(N, cats, K, cyclic=False):
    """A K-level register permuted by every mark (read in its basis) and a fuzzy probe (random marks) per epoch."""
    eps, nc = sorted(cats), ncats(cats)
    marks = {t: [np.kron(shift(K, rng.integers(K)) if cyclic else np.eye(K)[rng.permutation(K)],
                         kron_all([haar_unitary(nc[j], rng) if j == i else np.eye(nc[j]) for j in range(len(eps))]))
                 for a in range(nc[i])] for i, t in enumerate(eps)}
    L = int(np.prod(nc))
    reads = {t: [np.kron(proj(K, j), np.eye(L)) for j in range(K)] for t in eps}
    return Consulted(N, cats, K * L, np.kron(e(K, 0), e(L, 0)), marks, reads)

def bell_pair(N, cats, d):
    t1, t2 = sorted(cats)
    X, I = shift(d), np.eye(d)
    Phi = sum(np.kron(e(d, j), e(d, j)) for j in range(d)) / np.sqrt(d)
    Qs = [np.outer(b, b.conj()) for b in [np.kron(np.linalg.matrix_power(X, j), I) @ Phi for j in range(d)]]
    Qs.append(np.eye(d * d) - sum(Qs))          # completion of the reading (an outcome that never occurs)
    marks = {t1: [np.kron(np.linalg.matrix_power(X, a), I) for a in range(d)],
             t2: [np.kron(I, np.linalg.matrix_power(X, a)) for a in range(d)]}
    return Consulted(N, cats, d * d, Phi, marks, {t1: Qs, t2: Qs})

def copying(N, cats, d):
    """Fresh d-level probes; probe i receives a copy of probe i-1 (controlled shift), then the mark X^a."""
    eps = sorted(cats)
    m, I = len(eps), [np.eye(d)] * len(eps)
    marks, reads = {}, {}
    for i, t in enumerate(eps):
        C = np.eye(d ** m) if i == 0 else sum(kron_all([proj(d, u) if j == i - 1 else shift(d, u) if j == i else I[j]
                                                       for j in range(m)]) for u in range(d))
        marks[t] = [kron_all([shift(d, a) if j == i else I[j] for j in range(m)]) @ C for a in range(d)]
        reads[t] = [kron_all([proj(d, nu) if j == i else I[j] for j in range(m)]) for nu in range(d)]
    return Consulted(N, cats, d ** m, e(d ** m, 0), marks, reads)

def sharp_consulted(N, cats, s, nout):
    """Fresh sharp probes (pointer basis); the reading at t projects on the prefixes with s_t = nu."""
    eps, nc = sorted(cats), ncats(cats)
    m, D = len(eps), int(np.prod(nc))
    marks = {t: [kron_all([shift(nc[j], a) if j == i else np.eye(nc[j]) for j in range(m)]) for a in range(nc[i])]
             for i, t in enumerate(eps)}
    reads = {}
    for i, t in enumerate(eps):
        reads[t] = [np.zeros((D, D), dtype=complex) for _ in range(nout[t])]
        for pre in prod(*[range(c) for c in nc[:i + 1]]):
            reads[t][s[t][pre]] += kron_all([proj(nc[j], pre[j]) for j in range(i + 1)] + [np.eye(c) for c in nc[i + 1:]])
    return Consulted(N, cats, D, e(D, 0), marks, reads)

def classical_law(N, As, sel, chron, x_i):
    """sum over paths of prod_k |<x_k|A_k^(sel(k, chronicle))|x_{k-1}>|^2, the formula of (iv)."""
    P = np.zeros(N)
    for path in prod(range(N), repeat=len(As)):
        a, pr, prev = chron(path), 1.0, x_i
        for k, xk in enumerate(path, start=1):
            pr *= abs(As[k - 1][0 if k == 1 else sel(k, a)][xk, prev]) ** 2
            prev = xk
        P[path[-1]] += pr
    return P

CONFIGS = [(2, 3, (1, 2)), (3, 3, (1,)), (2, 4, (1, 3)), (3, 4, (2, 3)), (2, 4, (1, 2, 3)), (3, 4, (1, 2, 3)), (2, 4, (2,))]

# ----------------------------------------------------------------------------- 1. statement (i)
w_full, w_kept, max_cross = 0.0, 0.0, 0.0
for (N, calN, R), kind, D in prod(CONFIGS, ('coarse', 'injective'), (2, 3)):
    proc = random_archive(N, make_cats(N, R, kind), D)
    steps, x_i = proc.random_steps(calN), int(rng.integers(N))
    P_full, P_diag, cross = proc.formula_i(steps, x_i)
    w_full = max(w_full, np.max(np.abs(proc.simulate(steps, x_i) - P_full)))
    w_kept = max(w_kept, np.max(np.abs(proc.simulate(steps, x_i, copy=True) - P_diag)))
    max_cross = max(max_cross, cross)
check("1a (i) double sum over (x,nu),(x',nu') = joint simulation (%d instances)" % (len(CONFIGS) * 4), w_full < 1e-12, "max diff %.1e" % w_full)
check("1b (i) readings copied on fresh probes: the nu = nu' terms = joint simulation", w_kept < 1e-12, "max diff %.1e" % w_kept)
check("1c without copying the nu != nu' terms do not vanish", max_cross > 1e-3, "largest %.3f" % max_cross)

# ----------------------------------------------------------------------------- 2. statement (ii)

def definite_instances():
    for (N, calN, R), kind in prod(CONFIGS, ('coarse', 'injective')):
        cats = make_cats(N, R, kind)
        yield calN, counter(N, cats, int(rng.integers(2, 4)))[0]
        yield calN, perm_fuzzy(N, cats, int(rng.integers(2, 4)))
        if len(R) <= 2:
            yield calN, copying(N, cats, max(ncats(cats)))
        if len(R) == 2:
            yield calN, bell_pair(N, cats, max(ncats(cats)))

def backaction_check(proc, steps, s, x_i):
    """Record with back-action relative to random reference steps; returns (simulation diff, kernel diff)."""
    N, D, eps, calN = proc.N, proc.D, proc.epochs, len(steps)
    Wref = [Wk[0] if proc.t_of(k) is None else step(N) for k, Wk in enumerate(steps, start=1)]
    units, Bs = {}, {}
    for i, t in enumerate(eps):
        t2 = eps[i + 1] if i + 1 < len(eps) else calN
        Bs[t] = []
        for nu in range(proc.nout[t]):
            Mr, Mn = np.eye(N), np.eye(N)
            for k in range(t + 1, t2 + 1):
                Mr, Mn = Wref[k - 1] @ Mr, steps[k - 1][nu] @ Mn
            Bs[t].append(np.linalg.inv(Mr) @ Mn)
        units[t] = sum(np.kron(B, Q) for B, Q in zip(Bs[t], proc.reads[t])) @ proc.record_op(t)
    psi = np.kron(e(N, x_i), proc.Phi0)
    for k, W in enumerate(Wref, start=1):
        psi = np.kron(W, np.eye(D)) @ psi
        psi = units[k] @ psi if k in units else psi
    d_sim = np.max(np.abs(np.sum(np.abs(psi.reshape(N, D)) ** 2, axis=1) - proc.simulate(steps, x_i)))
    seqs, F = proc.kernel()
    idx = {a: j for j, a in enumerate(seqs)}
    Phis, pred = [], []
    for r in prod(prod(range(N), repeat=2), repeat=len(eps)):   # transcript r = ((x_i, y_i))_i
        v = proc.Phi0
        for t, (x, y) in zip(eps, r):
            v = units[t][y * D:(y + 1) * D, x * D:(x + 1) * D] @ v
        Phis.append(v)
        a = tuple(int(proc.cats[t][x]) for t, (x, y) in zip(eps, r))
        pred.append((a, np.prod([Bs[t][s[t][a[:i + 1]]][y, x] for i, (t, (x, y)) in enumerate(zip(eps, r))])))
    Phis = np.array(Phis)
    Fp = np.array([[F[idx[a], idx[a2]] * b * np.conj(b2) for (a2, b2) in pred] for (a, b) in pred])
    return d_sim, np.max(np.abs(Phis @ Phis.conj().T - Fp))

w_def, w_ii, w_sim, w_ker, n_ii = 0.0, 0.0, 0.0, 0.0, 0
for calN, proc in definite_instances():
    s, res = proc.definite_s()
    steps, x_i = proc.random_steps(calN), int(rng.integers(proc.N))
    ds, dk = backaction_check(proc, steps, s, x_i)
    w_ii = max(w_ii, np.max(np.abs(proc.simulate(steps, x_i) - proc.formula_ii(steps, x_i, s))))
    w_def, w_sim, w_ker, n_ii = max(w_def, res), max(w_sim, ds), max(w_ker, dk), n_ii + 1
check("2a readings of counters, pair, permutation register (x) fuzzy probes, copying probes are definite",
      w_def < 1e-12, "max violation %.1e" % w_def)
check("2b (ii) sum F[c(x),c(x')] phi_s[x] conj(phi_s[x']) = joint simulation (%d instances)" % n_ii, w_ii < 1e-12, "max diff %.1e" % w_ii)
check("2c (ii) the back-action form with B_t^(nu) reproduces the consulted process", w_sim < 1e-12, "max diff %.1e" % w_sim)
check("2d (ii) transcript kernel = F[a,a'] beta(r) conj(beta(r'))", w_ker < 1e-12, "max diff %.1e" % w_ker)

# ----------------------------------------------------------------------------- 3. statement (iii)
FAMS = ['counter2', 'counter3', 'pair', 'perm+fuzzy', 'counter+fuzzy']
ratios, viol, n_pos = [], 0, {f: 0 for f in FAMS}
for it in range(300):
    N, calN, fam = int(rng.integers(2, 4)), int(rng.integers(3, 5)), FAMS[it % 5]
    m = 2 if fam == 'pair' else int(rng.integers(1, min(3, calN - 1) + 1))
    R = tuple(sorted(int(t) for t in rng.choice(np.arange(1, calN), size=m, replace=False)))
    cats = make_cats(N, R, 'injective' if (rng.random() < 0.5 and not (fam == 'pair' and N == 3)) else 'coarse')
    if fam in ('counter2', 'counter3'):
        proc = counter(N, cats, int(fam[-1]))[0]
    elif fam == 'pair':
        proc = bell_pair(N, cats, max(ncats(cats)))
    else:
        proc = perm_fuzzy(N, cats, int(rng.integers(2, 4)), cyclic=(fam == 'counter+fuzzy'))
    s, res = proc.definite_s()
    assert res < 1e-12
    steps, x_i = proc.random_steps(calN), int(rng.integers(N))
    d = tv(proc.simulate(steps, x_i), sharp_consulted(N, cats, s, proc.nout).simulate(steps, x_i))
    lam = float(np.min(np.linalg.eigvalsh(proc.kernel()[1])))
    viol += int(d > 1 - lam + 1e-12)
    ratios += [d / (1 - lam)] if 1 - lam > 1e-9 else []
    n_pos[fam] += int(lam > 1e-6)
check("3 (iii) TV(P, P_delta,s) <= 1 - lambda_min(F) over 300 instances", viol == 0,
      "violations %d, largest ratio %.3f; instances with lambda_min > 0: %s" % (viol, max(ratios), n_pos))

# ----------------------------------------------------------------------------- 4. statement (iv)
w_prod, w_val = 0.0, 0.0
for (N, calN), _ in prod([(2, 3), (3, 3), (2, 4), (3, 4)], range(3)):
    R = tuple(range(1, calN))
    cats = make_cats(N, R, 'injective')
    K = {t: int(rng.integers(2, 4)) for t in R}
    s = {t: {pre: int(rng.integers(K[t])) for pre in prod(range(N), repeat=i + 1)} for i, t in enumerate(R)}
    proc = sharp_consulted(N, cats, s, K)
    As = [[haar_unitary(N, rng) for _ in range(1 if k == 1 else K[k - 1])] for k in range(1, calN + 1)]
    x_i = int(rng.integers(N))
    Ps = [proc.simulate([[step(N, A) for A in Ak] for Ak in As], x_i) for _ in range(4)]   # 4 sets of valuations
    w_prod = max(w_prod, np.max(np.abs(Ps[0] - classical_law(N, As, lambda k, a: s[k - 1][a[:k - 1]], proc.chron, x_i))))
    w_val = max(w_val, max(tv(Ps[0], q) for q in Ps[1:]))
check("4a (iv) P_delta,s = sum over paths of prod |<x_k|A_k^(s)|x_{k-1}>|^2 (12 instances)", w_prod < 1e-12, "max diff %.1e" % w_prod)
check("4b (iv) P_delta,s does not depend on the valuations V_k, V_k^(nu)", w_val < 1e-12, "max TV %.1e" % w_val)

def unitary_with_column(v, j):
    Q, Rr = np.linalg.qr(np.column_stack([v] + [rng.normal(size=len(v)) + 1j * rng.normal(size=len(v)) for _ in v[1:]]))
    Q[:, 0] *= Rr[0, 0] / abs(Rr[0, 0])
    order = list(range(1, len(v)))
    order.insert(j, 0)
    return Q[:, order]

w_real = 0.0
for it in range(24):
    N, calN = [(2, 3), (3, 3), (2, 4)][it % 3]
    R = tuple(range(1, calN))
    cats = make_cats(N, R, 'injective')
    inv = {t: np.argsort(cats[t]) for t in R}
    p = rng.dirichlet(np.ones(N ** calN)) * (rng.random(N ** calN) > (0.3 if it % 2 else -1))   # odd it: zeros
    p = (p / p.sum()).reshape((N,) * calN)
    pres = {t: list(prod(range(N), repeat=i + 1)) for i, t in enumerate(R)}
    s = {t: {pre: j for j, pre in enumerate(pres[t])} for t in R}          # the reading = the whole chronicle
    proc, x_i = sharp_consulted(N, cats, s, {t: len(pres[t]) for t in R}), int(rng.integers(N))

    def cond(h):   # p_k(. | h), h = (x_1, ..., x_{k-1})
        q = p.sum(axis=tuple(range(len(h) + 1, calN)))[h]
        return q / q.sum() if q.sum() > 0 else np.ones(N) / N
    steps = [[step(N, unitary_with_column(np.sqrt(cond(())).astype(complex), x_i))]]
    for k in range(2, calN + 1):
        hs = [tuple(int(inv[t][a]) for t, a in zip(R, pre)) for pre in pres[k - 1]]
        steps.append([step(N, unitary_with_column(np.sqrt(cond(h)).astype(complex), h[-1])) for h in hs])
    J = np.abs(proc.simulate(steps, x_i, joint=True)[1]) ** 2
    q = np.zeros_like(p)
    for xf, (j, pre) in prod(range(N), enumerate(prod(range(N), repeat=calN - 1))):
        q[tuple(int(inv[t][a]) for t, a in zip(R, pre)) + (xf,)] += J[xf, j]
    w_real = max(w_real, np.max(np.abs(q - p)))
check("4c (iv) 24 random laws on courses (12 with zeros) = joint law of (chronicle, x_f)", w_real < 1e-12, "max diff %.1e" % w_real)

# ----------------------------------------------------------------------------- 5. examples
# (a) consulted counter
w_a, w_tally, w_F = 0.0, 0, 0.0
for (N, calN, R) in CONFIGS[:5]:
    D, cats = int(rng.integers(2, 4)), make_cats(N, R, 'coarse')
    proc, mu = counter(N, cats, D)
    tally = lambda pre: int(sum(mu[a] for a in pre)) % D
    s, _ = proc.definite_s()
    w_tally = max(w_tally, max(abs(s[t][pre] - tally(pre)) for t in R for pre in s[t]))
    seqs, F = proc.kernel()
    w_F = max(w_F, np.max(np.abs(F - np.array([[float(tally(a) == tally(b)) for b in seqs] for a in seqs]))))
    steps, x_i = proc.random_steps(calN), int(rng.integers(N))
    Psi = np.zeros((N, D), dtype=complex)
    for path in prod(range(N), repeat=calN):
        a = proc.chron(path)
        Psi[path[-1], tally(a)] += proc.amp(steps, path, tuple(tally(a[:i + 1]) for i in range(len(R))), x_i)
    w_a = max(w_a, np.max(np.abs(np.sum(np.abs(Psi) ** 2, axis=1) - proc.simulate(steps, x_i))))
check("5a (a) counter: s_t = running tally, F = [tally = tally'], P = sum_j |Psi_j|^2",
      w_tally == 0 and w_F < 1e-12 and w_a < 1e-12, "max diffs %.1e / %.1e" % (w_F, w_a))

# (b) records lost together: per epoch a flag (0 kept, 1 lost) and a probe; the reading at t shows flags and
# pointers of all epochs <= t; the flags are prepared in |S> and P is averaged over S with weight mu(S)

def lost_together(N, cats, S):
    eps, nc = sorted(cats), ncats(cats)
    dims = [x for c in nc for x in (2, c)]
    I = [np.eye(dd) for dd in dims]
    marks = {t: [sum(kron_all([proj(2, f) if j == 2 * i else (shift(nc[i], a * (1 - f)) if j == 2 * i + 1 else I[j])
                               for j in range(len(dims))]) for f in (0, 1)) for a in range(nc[i])] for i, t in enumerate(eps)}
    reads = {t: [kron_all([proj(dims[j], v[j]) for j in range(len(v))] + I[len(v):])
                 for v in prod(*[range(dd) for dd in dims[:2 * i + 2]])] for i, t in enumerate(eps)}
    Phi0 = kron_all([e(dd, int(eps[j // 2] in S) if j % 2 == 0 else 0)[:, None] for j, dd in enumerate(dims)])[:, 0]
    return Consulted(N, cats, int(np.prod(dims)), Phi0, marks, reads)

def P_component(N, cats, S, steps, x_i):
    """P_S by its definition: sharp records outside S, consulted through nu(S, a) (lost pointers show 0)."""
    eps, nc, proc = sorted(cats), ncats(cats), lost_together(N, cats, S)
    Psi = {}
    for path in prod(range(N), repeat=len(steps)):
        a = [int(cats[t][path[t - 1]]) for t in eps]
        shown = [x for j, t in enumerate(eps) for x in (int(t in S), 0 if t in S else a[j])]
        nu = [int(np.ravel_multi_index(shown[:2 * i + 2], [x for c in nc[:i + 1] for x in (2, c)])) for i in range(len(eps))]
        key = (path[-1], tuple(a[i] for i, t in enumerate(eps) if t not in S))
        Psi[key] = Psi.get(key, 0) + proc.amp(steps, path, nu, x_i)
    P = np.zeros(N)
    for (xf, _), A in Psi.items():
        P[xf] += abs(A) ** 2
    return P

w_b, worst_bound = 0.0, -1.0
for (N, calN, R), kind in prod([(2, 3, (1, 2)), (3, 3, (1, 2)), (2, 4, (1, 3)), (2, 4, (1, 2, 3))], ('coarse', 'injective')):
    cats = make_cats(N, R, kind)
    subsets = [frozenset(c) for r in range(len(R) + 1) for c in itertools.combinations(R, r)]
    mu = dict(zip(subsets, rng.dirichlet(np.ones(len(subsets)))))
    steps, x_i = lost_together(N, cats, frozenset()).random_steps(calN), int(rng.integers(N))
    P = sum(q * lost_together(N, cats, S).simulate(steps, x_i) for S, q in mu.items())
    w_b = max(w_b, np.max(np.abs(P - sum(q * P_component(N, cats, S, steps, x_i) for S, q in mu.items()))))
    worst_bound = max(worst_bound, tv(P, P_component(N, cats, frozenset(), steps, x_i)) - (1 - mu[frozenset()]))
check("5b (b) lost together and consulted: P = sum_S mu(S) P_S", w_b < 1e-12, "max diff %.1e" % w_b)
check("5c (b) TV(P, P_empty) <= 1 - mu(empty)", worst_bound < 1e-12, "max excess %.2e" % worst_bound)
# components differ without interference: N = 2, R = {1,2}, W_1 = H, W_2 = 1, W_3 = X^(pointer of epoch 1)
cats, H, X = {1: np.array([0, 1]), 2: np.array([0, 1])}, hadamard_matrix(2), shift(2)
steps = [[H], [np.eye(2)] * 4, [X if v[1] else np.eye(2) for v in prod(range(2), repeat=4)]]
comps = [lost_together(2, cats, S) for S in (frozenset(), frozenset({1}))]
P0, P1 = [c.simulate(steps, 0) for c in comps]
no_int = all(np.allclose(c.simulate(steps, 0), c.formula_ii(steps, 0, c.definite_s()[0], F=np.eye(4))) for c in comps)
check("5d (b) components differ, no two courses interfere in either: P_empty = %s, P_{1} = %s"
      % (np.round(P0, 3), np.round(P1, 3)), no_int and tv(P0, P1) > 0.4, "TV %.3f" % tv(P0, P1))

# (c) copying probes, under the assumptions of (iv) with d = N
w_c1, w_c2, gap = 0.0, 0.0, 0.0
for (N, calN) in [(2, 3), (2, 4), (3, 3)]:
    R = tuple(range(1, calN))
    cats = make_cats(N, R, 'injective')
    proc = copying(N, cats, N)
    s, res = proc.definite_s()
    seqs, F = proc.kernel()
    tally_ok = all(s[t][pre] == sum(pre) % N for t in R for pre in s[t]) and res < 1e-12
    w_c1 = max(w_c1, np.max(np.abs(F - np.eye(len(seqs)))) + (0 if tally_ok else 1))
    As = [[haar_unitary(N, rng) for _ in range(1 if k == 1 else N)] for k in range(1, calN + 1)]
    steps, x_i = [[step(N, A) for A in Ak] for Ak in As], int(rng.integers(N))
    Pc = proc.simulate(steps, x_i)
    w_c2 = max(w_c2, np.max(np.abs(Pc - classical_law(N, As, lambda k, a: sum(a[:k - 1]) % N, proc.chron, x_i))))
    gap = max(gap, tv(Pc, counter(N, cats, N, mu=np.arange(N))[0].simulate(steps, x_i)))
check("5e (c) copying probes: newest probe = running tally (definite), F = delta", w_c1 < 1e-12, "max %.1e" % w_c1)
check("5f (c) course = classical process with tally-dependent transitions", w_c2 < 1e-12, "max diff %.1e" % w_c2)
check("5g (c) it differs from the consulted counter with the same steps", gap > 1e-2, "largest TV %.3f" % gap)

# (d) entangled pair
pair = bell_pair(2, cats := {1: np.array([0, 1]), 2: np.array([0, 1])}, 2)
s, res = pair.definite_s()
seqs, F = pair.kernel()
s_ok = res < 1e-12 and all(s[1][(a,)] == a and s[2][(a, b)] == (a - b) % 2 for a in range(2) for b in range(2))
F_ok = np.allclose(F, [[float((a[1] - a[0]) % 2 == (b[1] - b[0]) % 2) for b in seqs] for a in seqs])
w_d = 0.0
for theta in np.linspace(0, 2 * np.pi, 7):
    steps = [[H], [np.eye(2), np.exp(1j * theta) * np.eye(2), np.eye(2)], [H] * 3]
    P, Psh = pair.simulate(steps, 0), sharp_consulted(2, cats, s, pair.nout).simulate(steps, 0)
    w_d = max(w_d, abs(P[0] - (1 + np.cos(theta)) / 2), abs(Psh[0] - 0.5), abs(pair.formula_ii(steps, 0, s)[0] - P[0]))
check("5h (d) entangled pair: s_t1 = a1, s_t2 = a1 - a2, F = [change = change'], P(0) = (1+cos)/2 vs 1/2",
      s_ok and F_ok and w_d < 1e-12, "max diff %.1e" % w_d)

# ----------------------------------------------------------------------------- 6. commutant lemma

def commutant(mats):
    n = mats[0].shape[0]
    C = np.vstack([np.kron(np.eye(n), M.T) - np.kron(M, np.eye(n)) for M in mats])   # row-major vec(TM - MT)
    return n * n - int(np.sum(np.linalg.svd(C, compute_uv=False) > 1e-9)), C

ok6, info = True, []
for N in (2, 3, 4):
    Om = sum(np.kron(e(N, x), e(N, x)) for x in range(N)) / np.sqrt(N)
    rot = np.eye(N, dtype=complex)
    rot[:2, :2] = [[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]]
    diag = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, N)))
    perms = [np.eye(N)[list(p)] for p in itertools.permutations(range(N))]
    for label, Ws in [('Haar', [haar_unitary(N, rng) for _ in range(3)]), ('generators', [diag] + perms + [rot]),
                      ('no rotation', [diag] + perms)]:
        dim, C = commutant([np.kron(W, W.conj()) for W in Ws])
        inside = max(np.linalg.norm(C @ np.eye(N * N).reshape(-1)), np.linalg.norm(C @ np.outer(Om, Om).reshape(-1)))
        info.append("N=%d %s: %d" % (N, label, dim))
        ok6 &= label == 'no rotation' or (dim == 2 and inside < 1e-10)
check("6 commutant of {W (x) conj W} = span{1, P_Omega} for N = 2, 3, 4 (dimensions listed)", ok6, "; ".join(info))

# ----------------------------------------------------------------------------- 7. a reading that is not definite
# N = 2, calN = 2, R = {1}, a fresh sharp probe (F = delta), W_1 = Hadamard.  The valuation alone is consulted:
# W_2^(nu) = H diag(1, exp(i(theta + nu pi))).  Read in the pointer basis (definite reading) the course is blind,
# P(0) = 1/2; read in the Fourier basis of the probe (not definite) the interference that the mark had removed
# returns, P(0) = (1 + cos theta)/2.  This is the reason for the hypothesis of definite readings in (ii)-(iv).

def consulted_two_step(theta, basis):
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    psi = np.zeros((2, 2), dtype=complex)             # psi[x, probe]
    psi[:, 0] = H[:, 0]                               # step 1 from x_i = 0, probe in |0>
    psi = np.stack([psi[0], psi[1][::-1]])            # sharp record: probe -> |x>
    out = np.zeros((2, 2), dtype=complex)
    for nu in range(2):
        Q = np.outer(basis[:, nu], basis[:, nu].conj())
        W = H @ np.diag([1.0, np.exp(1j * (theta + nu * np.pi))])
        out += W @ psi @ Q.T                          # (W (x) Q) acting on psi[x, probe]
    return float(np.sum(np.abs(out[0]) ** 2))

worst_def, worst_indef = 0.0, 0.0
fourier = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
for theta in np.linspace(0.0, np.pi, 7):
    worst_def = max(worst_def, abs(consulted_two_step(theta, np.eye(2, dtype=complex)) - 0.5))
    worst_indef = max(worst_indef, abs(consulted_two_step(theta, fourier) - 0.5 * (1 + np.cos(theta))))
check("7 valuation-only consultation of a sharp probe: pointer reading gives 1/2, Fourier reading (1 + cos theta)/2",
      worst_def < 1e-12 and worst_indef < 1e-12, "max diffs %.1e / %.1e" % (worst_def, worst_indef))

print("\nrun time %.1f s" % (time.time() - T0))
print("ALL PASS" if all(ok for _, ok in report) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(ok for _, ok in report) else 1)
