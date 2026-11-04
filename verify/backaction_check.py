"""Records that act on what they record (records with back-action).

The recording unitary at a recorded epoch is now an arbitrary unitary U_t on C^N (x) C^D, written as
U_t = sum_{x,y} |y><x| (x) M_t^{(y,x)}.  A course is then a sequence of configurations before and after
each record (a transcript), the system amplitude is carried by the steps alone, and the archive vector of
a transcript r is  Phi_r = M_{t_m}^{(y_m, x_m)} ... M_{t_1}^{(y_1, x_1)} Phi_0.

Checks for the manuscript (theorem on records with back-action and its corollary):
  1. P(x_f) = sum_{r, r'} F[r, r'] Psi[x_f, r] conj(Psi[x_f, r'])  with F[r, r'] = <Phi_{r'}|Phi_r>,
     against the explicit joint simulation (random unitaries).
  2. F is positive semidefinite and satisfies the normalisation of quantum combs:
        sum_y F[(r,(x,y)), (r',(x',y))] = delta_{x x'} F_prev[r, r'].
  3. Converse: from such a kernel a process with archive dimension max_i rank F^{(i)} is rebuilt and
     reproduces the realised distributions.
  4. Latitude with back-action: TV(P_F, P_D) <= 1 - lambda_min(D^{-1/2} F D^{-1/2}), D = diag F,
     and P_D is the distribution of the process augmented by a sharp record of the transcript.
  5. The sharp transcript (F diagonal) is blind to the valuations; a memoryless record with back-action is a
     channel at the recorded epoch.
Only numpy is required.
"""
from __future__ import annotations

import itertools
import numpy as np

from pimh_memory import haar_unitary, tv, is_psd

rng = np.random.default_rng(1104)
TOL = 1e-10
report = []


def check(name, ok, detail=""):
    report.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))


class BackActionProcess:
    """units: dict epoch -> unitary on C^N (x) C^D (system index most significant)."""

    def __init__(self, N, D, Phi0, units):
        self.N, self.D = N, D
        self.Phi0 = np.asarray(Phi0, dtype=complex)
        self.units = {int(t): np.asarray(u, dtype=complex) for t, u in units.items()}
        self.epochs = sorted(self.units)

    def block(self, t, y, x):
        D = self.D
        return self.units[t][y * D:(y + 1) * D, x * D:(x + 1) * D]

    def simulate(self, Ws, x_i, archive_post=None, return_joint=False):
        N, D = self.N, self.D
        psi = np.zeros(N * D, dtype=complex)
        psi[x_i * D:(x_i + 1) * D] = self.Phi0
        for k, W in enumerate(Ws, start=1):
            psi = np.kron(W, np.eye(D)) @ psi
            if k in self.units:
                psi = self.units[k] @ psi
        if archive_post is not None:
            psi = np.kron(np.eye(N), archive_post) @ psi
        P = np.array([np.sum(np.abs(psi[x * D:(x + 1) * D]) ** 2) for x in range(N)])
        return (P, psi) if return_joint else P

    # transcripts and kernels
    def transcripts(self, upto=None):
        eps = self.epochs if upto is None else [t for t in self.epochs if t <= upto]
        return list(itertools.product(itertools.product(range(self.N), repeat=2), repeat=len(eps)))

    def archive_vector(self, r):
        v = self.Phi0.copy()
        for t, (x, y) in zip(self.epochs, r):
            v = self.block(t, y, x) @ v
        return v

    def kernel(self, upto=None):
        trs = self.transcripts(upto)
        eps = self.epochs if upto is None else [t for t in self.epochs if t <= upto]
        V = []
        for r in trs:
            v = self.Phi0.copy()
            for t, (x, y) in zip(eps, r):
                v = self.block(t, y, x) @ v
            V.append(v)
        V = np.array(V)
        return trs, (V.conj() @ V.T).conj()   # F[r, r'] = <Phi_{r'}|Phi_r>

    def transcript_amplitudes(self, Ws, x_i):
        """Psi[x_f, idx(r)] = sum over the paths with transcript r of prod_k <x_k^-|W_k|x_{k-1}^+>."""
        N = self.N
        calN = len(Ws)
        trs = self.transcripts()
        idx = {r: i for i, r in enumerate(trs)}
        Psi = np.zeros((N, len(trs)), dtype=complex)
        # doubled path: for every intermediate epoch k a pair (x_k^-, x_k^+); x_k^+ = x_k^- if k unrecorded
        choices = []
        for k in range(1, calN):
            if k in self.units:
                choices.append(list(itertools.product(range(N), repeat=2)))
            else:
                choices.append([(x, x) for x in range(N)])
        for dp in itertools.product(*choices):
            amp = 1 + 0j
            prev = x_i
            for k in range(1, calN):
                xm, xp = dp[k - 1]
                amp *= Ws[k - 1][xm, prev]
                prev = xp
            r = tuple(dp[k - 1] for k in self.epochs)
            for x_f in range(N):
                Psi[x_f, idx[r]] += amp * Ws[calN - 1][x_f, prev]
        return trs, Psi

    def pair_sum(self, Ws, x_i, F_override=None):
        trs, Psi = self.transcript_amplitudes(Ws, x_i)
        F = self.kernel()[1] if F_override is None else F_override
        return np.real(np.einsum('xa,ab,xb->x', Psi, F, Psi.conj()))


def random_process(N, D, epochs):
    Phi0 = np.zeros(D, dtype=complex)
    Phi0[0] = 1.0
    return BackActionProcess(N, D, Phi0, {t: haar_unitary(N * D, rng) for t in epochs})


# ----------------------------------------------------------------------------- 1. pair sum
worst = 0.0
for (N, D, calN, R) in [(2, 2, 3, (1, 2)), (3, 2, 3, (1, 2)), (2, 3, 4, (1, 3)), (2, 2, 4, (1, 2, 3))]:
    for _ in range(5):
        proc = random_process(N, D, R)
        Ws = [haar_unitary(N, rng) for _ in range(calN)]
        x_i = int(rng.integers(N))
        worst = max(worst, np.max(np.abs(proc.simulate(Ws, x_i) - proc.pair_sum(Ws, x_i))))
check("1 pair sum over transcripts = joint simulation (20 random processes)", worst < 1e-12, "max diff %.1e" % worst)

# ----------------------------------------------------------------------------- 2. kernel properties
proc = random_process(2, 3, (1, 2, 3))
trs, F = proc.kernel()
idx = {r: i for i, r in enumerate(trs)}
check("2a F is positive semidefinite", is_psd(F))
check("2b F[r,r] = ||Phi_r||^2 <= 1 and not all one (back-action)",
      np.all(np.real(np.diag(F)) <= 1 + TOL) and np.min(np.real(np.diag(F))) < 0.99)
trs2, F2 = proc.kernel(upto=2)
idx2 = {r: i for i, r in enumerate(trs2)}
N = 2
worst = 0.0
for r in trs2:
    for r2 in trs2:
        for x in range(N):
            for x2 in range(N):
                s = sum(F[idx[r + ((x, y),)], idx[r2 + ((x2, y),)]] for y in range(N))
                target = (F2[idx2[r], idx2[r2]] if x == x2 else 0.0)
                worst = max(worst, abs(s - target))
check("2c comb normalisation: sum_y F[(r,(x,y)),(r',(x',y))] = delta_{xx'} F_prev[r,r']", worst < 1e-12, "max %.1e" % worst)

# ----------------------------------------------------------------------------- 3. converse


def process_from_transcript_kernel(F_full, trs, N, epochs, tol=1e-9):
    """Rebuild a record process with back-action from a kernel on transcripts that is PSD and comb-normalised."""
    m = len(epochs)
    # level kernels by summing the common last output (any common last input, here x = 0)
    kernels = {m: (list(trs), F_full)}
    for i in range(m, 0, -1):
        trs_i, F_i = kernels[i]
        idx_i = {r: k for k, r in enumerate(trs_i)}
        trs_prev = sorted(set(r[:-1] for r in trs_i))
        F_prev = np.zeros((len(trs_prev), len(trs_prev)), dtype=complex)
        for a, r in enumerate(trs_prev):
            for b, r2 in enumerate(trs_prev):
                F_prev[a, b] = sum(F_i[idx_i[r + ((0, y),)], idx_i[r2 + ((0, y),)]] for y in range(N))
        kernels[i - 1] = (trs_prev, F_prev)
    # Gram vectors at every level (F = V V^H with F[r,r'] = <v_{r'}|v_r>, rows v_r)
    vecs = {}
    dims = []
    for i in range(0, m + 1):
        trs_i, F_i = kernels[i]
        w, Q = np.linalg.eigh((F_i + F_i.conj().T) / 2)
        keep = w > tol
        V = Q[:, keep] * np.sqrt(w[keep])
        vecs[i] = {r: V[k] for k, r in enumerate(trs_i)}
        dims.append(V.shape[1])
    Dp = max(dims)
    pad = lambda v: np.concatenate([v, np.zeros(Dp - len(v), dtype=complex)])
    units = {}
    for i in range(1, m + 1):
        trs_prev = kernels[i - 1][0]
        cols_A, cols_B = [], []
        for r in trs_prev:
            for x in range(N):
                ex = np.zeros(N); ex[x] = 1
                cols_A.append(np.kron(ex, pad(vecs[i - 1][r])))
                b = np.zeros(N * Dp, dtype=complex)
                for y in range(N):
                    ey = np.zeros(N); ey[y] = 1
                    b += np.kron(ey, pad(vecs[i][r + ((x, y),)]))
                cols_B.append(b)
        A = np.array(cols_A).T
        B = np.array(cols_B).T
        assert np.allclose(A.conj().T @ A, B.conj().T @ B, atol=1e-8), "Gram matrices differ: kernel not comb-normalised"
        Ua, s, Vh = np.linalg.svd(A, full_matrices=False)
        rank = int(np.sum(s > tol))
        Apinv = Vh[:rank].conj().T @ np.diag(1 / s[:rank]) @ Ua[:, :rank].conj().T
        Upart = B @ Apinv
        Qa = Ua[:, :rank]
        Ub, sb, _ = np.linalg.svd(B, full_matrices=False)
        Qb = Ub[:, :int(np.sum(sb > tol))]
        n = N * Dp
        wa, ea = np.linalg.eigh(np.eye(n) - Qa @ Qa.conj().T)
        wb, eb = np.linalg.eigh(np.eye(n) - Qb @ Qb.conj().T)
        Ea = ea[:, wa > 0.5]
        Eb = eb[:, wb > 0.5]
        U = Upart @ (Qa @ Qa.conj().T) + Eb @ Ea.conj().T
        assert np.allclose(U.conj().T @ U, np.eye(n), atol=1e-8), "not unitary"
        assert np.allclose(U @ A, B, atol=1e-8)
        units[epochs[i - 1]] = U
    Phi0 = pad(vecs[0][()])
    return BackActionProcess(N, Dp, Phi0, units)


proc = random_process(2, 3, (1, 2))
trs, F = proc.kernel()
rebuilt = process_from_transcript_kernel(F, trs, 2, [1, 2])
worst = 0.0
for _ in range(10):
    Ws = [haar_unitary(2, rng) for _ in range(3)]
    x_i = int(rng.integers(2))
    worst = max(worst, np.max(np.abs(proc.simulate(Ws, x_i) - rebuilt.simulate(Ws, x_i))))
trs_r, F_r = rebuilt.kernel()
check("3 converse: the rebuilt process (archive dimension %d) has the same kernel and distributions" % rebuilt.D,
      worst < 1e-10 and np.allclose(F_r, F, atol=1e-8), "max diff %.1e" % worst)

# ----------------------------------------------------------------------------- 4. latitude with back-action


def augmented_sharp_transcript(proc, calN):
    """The same process with, at every recorded epoch, a fresh probe that records (x^-, x^+) sharply."""
    N, D = proc.N, proc.D
    m = len(proc.epochs)
    Dtot = D * (N * N) ** m
    Phi0 = proc.Phi0.copy()
    for _ in range(m):
        e = np.zeros(N * N, dtype=complex); e[0] = 1
        Phi0 = np.kron(Phi0, e)
    units = {}
    for i, t in enumerate(proc.epochs):
        U = np.zeros((N * Dtot, N * Dtot), dtype=complex)
        for x in range(N):
            for y in range(N):
                Mxy = proc.block(t, y, x)
                # shift the probe i from |0> to |xy> : operator S with S|j> = |j + (x N + y) mod N^2>
                S = np.roll(np.eye(N * N), x * N + y, axis=0)
                op = Mxy
                for j in range(m):
                    op = np.kron(op, S if j == i else np.eye(N * N))
                U[y * Dtot:(y + 1) * Dtot, x * Dtot:(x + 1) * Dtot] = op
        assert np.allclose(U.conj().T @ U, np.eye(N * Dtot), atol=1e-10)
        units[t] = U
    return BackActionProcess(N, Dtot, Phi0, units)


proc = random_process(2, 2, (1, 2))
trs, F = proc.kernel()
d = np.real(np.diag(F))
Dm = np.diag(d)
keep = d > 1e-12
Corr = (F[np.ix_(keep, keep)] / np.sqrt(np.outer(d[keep], d[keep])))
lam = float(np.min(np.linalg.eigvalsh((Corr + Corr.conj().T) / 2)))
check("4a F - lambda D is positive semidefinite for lambda = lambda_min of the correlation matrix (%.3f)" % lam,
      is_psd(F - lam * Dm, tol=1e-8))
aug = augmented_sharp_transcript(proc, 3)
worst_bound, worst_aug, worst_sum = 0.0, 0.0, 0.0
for _ in range(30):
    Ws = [haar_unitary(2, rng) for _ in range(3)]
    x_i = int(rng.integers(2))
    PF = proc.simulate(Ws, x_i)
    PD = proc.pair_sum(Ws, x_i, F_override=Dm)
    worst_sum = max(worst_sum, abs(PD.sum() - 1))
    worst_bound = max(worst_bound, tv(PF, PD) - (1 - lam))
    worst_aug = max(worst_aug, np.max(np.abs(PD - aug.simulate(Ws, x_i))))
check("4b P_D is a probability distribution and equals the augmented sharp-transcript process", worst_sum < 1e-12 and worst_aug < 1e-12,
      "max %.1e / %.1e" % (worst_sum, worst_aug))
check("4c TV(P_F, P_D) <= 1 - lambda_min(correlation matrix) for 30 random step sequences", worst_bound < 1e-12,
      "max excess %.1e" % worst_bound)

# ----------------------------------------------------------------------------- 5. blindness of the sharp transcript, channels
worst = 0.0
for _ in range(10):
    As = [haar_unitary(2, rng) for _ in range(3)]
    x_i = int(rng.integers(2))
    Ps = []
    for _ in range(4):
        v = rng.uniform(0, 2 * np.pi, size=(3, 2))
        Ps.append(proc.pair_sum([A @ np.diag(np.exp(1j * v[k])) for k, A in enumerate(As)], x_i, F_override=Dm))
    worst = max(worst, max(tv(Ps[0], Pj) for Pj in Ps[1:]))
check("5a the sharp transcript is blind to the valuations", worst < 1e-12, "max TV %.1e" % worst)
# a memoryless record with back-action = a channel at the recorded epoch
proc1 = random_process(2, 2, (1,))
trs1, f = proc1.kernel()
i1 = {r: k for k, r in enumerate(trs1)}
W1 = haar_unitary(2, rng)
P, psi = proc1.simulate([W1], 0, return_joint=True)
rho_after = psi.reshape(2, 2) @ psi.reshape(2, 2).conj().T
rho_before = np.outer(W1[:, 0], W1[:, 0].conj())
rho_chan = np.zeros((2, 2), dtype=complex)
for (x, y) in itertools.product(range(2), repeat=2):
    for (x2, y2) in itertools.product(range(2), repeat=2):
        rho_chan[y, y2] += f[i1[((x, y),)], i1[((x2, y2),)]] * rho_before[x, x2]
check("5b one record with back-action acts on the system as the channel with kernel f[(x,y),(x',y')]",
      np.allclose(rho_chan, rho_after, atol=1e-12))

print()
print("ALL PASS" if all(ok for _, ok in report) else "SOME CHECKS FAILED")
