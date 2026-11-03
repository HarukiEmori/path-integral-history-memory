# path-integral-history-memory

Numerical verification scripts for the manuscript *Records with Memory: A Non-Markovian Extension of a Path Integral Model of History* (H. Emori and A. Iriki, 2026).
The manuscript extends *A Path Integral Model of History* (same authors) from records taken by fresh independent probes to records with memory, an archive that is marked at several epochs, prepared in a correlated state, or processed between the records.
This repository holds only the checks and the data of the two numerical figures.

## Requirements
Python 3.11 or later with NumPy (SciPy is not needed). On the authors' machine: `~/Develop/venv/quantum-env/bin/python` (NumPy 2.4).

## Layout
| Path | Content |
| :-- | :-- |
| `verify/pimh_memory.py` | Shared library. A `RecordProcess` (archive space, initial state, one unitary mark per category and recorded epoch) with an explicit joint simulation of system and archive, the influence kernel $F[\mathbf a,\mathbf a'] = \langle\Phi_{\mathbf a'}\vert\Phi_{\mathbf a}\rangle$, the pair sum of Theorem 1, a causality test for the kernel, constructors for fresh probes, counters (ledgers), rotated-qubit ledgers, entangled pairs and chains of pairs, chained qubit probes with partial swaps, and classically correlated intermittent records, and the converse construction of Theorem 2 (a record process from a causal correlation kernel) |
| `verify/influence_functional_check.py` | Theorem 1 (pair sum = explicit simulation for random record processes with reused archives), properties of $F$ (unit diagonal, positive semidefinite, causal), Theorem 2 (rebuild a process from its kernel; a mixture kernel), the latitude bound $1-\lambda_{\min}(F)$ (Corollary 1) and its value $(1-\kappa)^{\lvert\mathcal R\rvert}$ for fresh probes, the finite-archive bound (Proposition 3), invariance under post-processing of the archive (Proposition 4), the failure of blindness for a parity ledger (Corollary 2), and the identification of $F$ by two-path dynamics (Proposition 5) |
| `verify/ledger_check.py` | Ledgers (Proposition 6): $F$ depends on the tallies only, counters of capacity $D$, rotated qubits, the closed forms of the minimal example with memory (fresh probes, counters of capacity 2 and 3 or more, rotated qubit, archive lost as a whole), and the kernel of correlated intermittent records (Proposition 7) |
| `verify/correlated_records_check.py` | Entangled pairs (Proposition 8): the record of change $F = [a_2 - a_1 \equiv a_2' - a_1' \bmod d]$, the general two-epoch formula and the trace formula for maximally entangled preparations, recoherence and the trace-distance backflow $1 \to 0 \to 1$ (Proposition 11), partial revival $\lvert\cos\phi\rvert$ with a rotation between the records, chains of pairs, and the range of the latitude bound of correlated losses |
| `verify/gaussian_field_check.py` | Gaussian pointer fields (Proposition 9): the overlap of displaced Gaussians by quadrature, the momentum covariance of the pure Gaussian state, the exponential kernel (persistent versus alternating differences, white-noise and ledger limits), and the spectral (filter-function) form of the exponent |
| `verify/retrodiction_check.py` | Section 6: causality (states and forward readout statistics do not depend on later operations), the past-quantum-state formula against brute-force conditional probabilities, the dependence of the retrodicted distribution on later valuations at $\kappa > 0$ and its reduction to forward-backward smoothing at $\kappa = 0$, and the ABL rule |
| `verify/figure_data.py` | Writes `figures/fig_counter.dat` (sensitivity to the valuation of a counter of capacity $D$ over six recorded epochs) and `figures/fig_gauss.dat` (influence functional of the exponential kernel as a function of the correlation $r$), whose values are inlined as pgfplots coordinates in the manuscript |

## Run
```bash
cd verify
for f in influence_functional_check.py ledger_check.py correlated_records_check.py gaussian_field_check.py retrodiction_check.py; do python "$f"; done
python figure_data.py
```
Every check script prints `ALL PASS` when all checks pass; the printed discrepancies are of the order of machine precision (the quadrature and finite-difference checks of `gaussian_field_check.py` are accurate to about $10^{-4}$ or better).

## Conventions
The numbering of theorems and propositions in the table follows the manuscript. Epochs are numbered from 1 and a record at epoch $t$ is applied after the step $W_t$. The kernel is stored as a matrix indexed by record sequences in the order of `itertools.product`, with `F[i, j] = <Phi_j | Phi_i>` in the convention $F[\mathbf a,\mathbf a'] = \langle\Phi_{\mathbf a'}\vert\Phi_{\mathbf a}\rangle$ of the manuscript.
