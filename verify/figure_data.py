"""Data for the figures of the manuscript (written as pgfplots tables into ../figures/).

fig_counter.dat : a reused D-level counter over m = 6 recorded epochs (N = 2, Hadamard steps,
                  independent random valuation phases at every epoch).  Columns: D, the mean and the
                  maximum over 150 random pairs of valuations of the total-variation distance between
                  the two realised distributions (the sensitivity to the valuation), and the mean
                  distance from the complete chronicle (uniform).  D = 1 is no record.
fig_gauss.dat   : Gaussian pointer field with the stationary kernel Gamma_tt' = g r^|t-t'| over
                  m = 4 recorded epochs, g chosen so that a single-epoch difference is damped to 1/2.
                  Columns: r, F(single), F(persistent), F(alternating), 1 - lambda_min(F).
"""
import itertools
import os
import numpy as np

from pimh_memory import ledger_process, fresh_probe_process, tv

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(out, exist_ok=True)

# ------------------------------------------------------------------ counter capacity
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
m = 6
calN = m + 1
cats = {t: np.array([0, 1]) for t in range(1, m + 1)}
rng = np.random.default_rng(1)
pairs = [(rng.uniform(0, 2 * np.pi, size=calN), rng.uniform(0, 2 * np.pi, size=calN)) for _ in range(150)]
Pc = np.array([0.5, 0.5])


def steps(th):
    return [H @ np.diag([1, np.exp(1j * t)]) for t in th]


rows = []
for D in range(1, 9):
    proc = ledger_process(2, cats, D)
    sens, dep = [], []
    for th1, th2 in pairs:
        P1 = proc.simulate(steps(th1), 0)
        P2 = proc.simulate(steps(th2), 0)
        sens.append(tv(P1, P2))
        dep.append(tv(P1, Pc))
    rows.append((D, float(np.mean(sens)), float(np.max(sens)), float(np.mean(dep))))
ref = {}
for label, kappa in [("sharp", 0.0), ("half", 0.5)]:
    proc = fresh_probe_process(2, cats, kappa)
    sens = [tv(proc.simulate(steps(th1), 0), proc.simulate(steps(th2), 0)) for th1, th2 in pairs]
    ref[label] = (float(np.mean(sens)), float(np.max(sens)))
with open(os.path.join(out, 'fig_counter.dat'), 'w') as f:
    f.write("D meanSens maxSens meanDep\n")
    for D, a, b, c in rows:
        f.write(f"{D} {a:.6f} {b:.6f} {c:.6f}\n")
print("counter capacity (m = 6 recorded epochs, random valuations):")
for D, a, b, c in rows:
    print(f"   D = {D}: sensitivity mean {a:.4f} max {b:.4f}; departure from chronicle mean {c:.4f}")
print(f"   fresh sharp probes: sensitivity {ref['sharp']}, fresh kappa = 0.5: {ref['half']}")

# ------------------------------------------------------------------ Gaussian kernel
m = 4
g = 2.0 * np.log(2.0)
seqs = list(itertools.product([0, 1], repeat=m))
pers = np.ones(m)
alt = np.array([1, -1, 1, -1.0])
single = np.array([1, 0, 0, 0.0])
with open(os.path.join(out, 'fig_gauss.dat'), 'w') as f:
    f.write("r Fsingle Fpersistent Falternating latitude\n")
    for r in np.linspace(0, 1, 101):
        Gam = np.array([[r ** abs(t - s) for s in range(m)] for t in range(m)])
        Fof = lambda d: np.exp(-0.5 * g * d @ Gam @ d)
        Fmat = np.array([[Fof(np.array(a) - np.array(b)) for b in seqs] for a in seqs])
        lam = max(np.min(np.linalg.eigvalsh(Fmat)), 0.0)
        f.write(f"{r:.3f} {Fof(single):.6f} {Fof(pers):.6f} {Fof(alt):.6f} {1 - lam:.6f}\n")
print("gaussian kernel table written")
