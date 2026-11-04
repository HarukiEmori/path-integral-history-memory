# path-integral-history-memory

Numerical verification scripts for the manuscript *A Non-Markovian Extension of a Path Integral Model of History* (H. Emori and A. Iriki, 2026).
The manuscript extends *A Path Integral Model of History* (same authors) from records taken by fresh independent probes to records with memory, an archive that is marked at several epochs, prepared in a correlated state, or processed between the records.
This repository holds only the checks and the data of the two numerical figures; the manuscript itself is not kept here.

## Requirements
Python 3.11 or later with NumPy (SciPy is not needed). On the authors' machine: `~/Develop/venv/quantum-env/bin/python` (NumPy 2.4).

## Layout
| Path | Content |
| :-- | :-- |
| `verify/pimh_memory.py` | Shared library. A `RecordProcess` (archive space, initial state, one unitary mark per category and recorded epoch) with an explicit joint simulation of system and archive, the influence kernel $F[\mathbf a,\mathbf a'] = \langle\Phi_{\mathbf a'}\vert\Phi_{\mathbf a}\rangle$, the pair sum of Theorem 1, a causality test for the kernel, constructors for fresh probes, counters (ledgers), rotated-qubit ledgers, entangled pairs and chains of pairs, chained qubit probes with partial swaps, and classically correlated intermittent records, and the converse construction of Theorem 2 (a record process from a causal correlation kernel) |
| `verify/influence_functional_check.py` | Theorem 1 (pair sum = explicit simulation for random record processes with reused archives), properties of $F$ (unit diagonal, positive semidefinite, causal), Theorem 2 (rebuild a process from its kernel; a mixture kernel), the latitude bound $1-\lambda_{\min}(F)$ (Corollary 1) and its value $(1-\kappa)^{\lvert\mathcal R\rvert}$ for fresh probes, the finite-archive bound (Proposition 4), invariance under post-processing of the archive (Proposition 5), the failure of blindness for a parity ledger (Corollary 2), and the identification of $F$ by two-path dynamics (Theorem 3(i), one block) |
| `verify/identification_check.py` | Theorem 3 and Proposition 3: the two-configuration archive with hidden memory (its kernel, its blindness for random steps and valuations), the rank of the real-linear map from Hermitian kernels to realised distributions ($N = 3$ with injective and with coarse resolutions: injective; $N = 2$: a null space spanned exactly by the hidden tensors of the parity rule, and the visible and invisible Walsh sums), and the multi-block interferometer with sinks for $N = 3$ and two blocks, with the discrete Fourier transform that recovers $F[\mathbf a,\mathbf a']$ |
| `verify/backaction_check.py` | Theorem 4 and Corollary 3 (records with back-action): the pair sum over transcripts against the joint simulation for random unitaries on system and archive, positivity and the comb normalisation of the kernel, the converse construction (a process rebuilt from its kernel), the latitude bound with the correlation matrix, the augmented sharp record of the transcript, its blindness to the valuations, and the channel of a memoryless record |
| `verify/account_memory_check.py` | Proposition 13: the account of an epoch under an archive with memory (joint past quantum state) against the brute-force conditional probability with an inserted sharp probe, before and after the record, the forward marginal, the sharp archive read in its pointer basis (record repeated, Markov smoothing at an unrecorded epoch, no valuation dependence) and in a scrambled basis (dependence), the counter read in two bases, and the analytic example $\Pr(a_2 = 0 \mid \pm, x_{\mathrm f}) = (1 \pm \cos\theta)/2$ |
| `verify/ledger_check.py` | Ledgers (Proposition 6): $F$ depends on the tallies only, counters of capacity $D$, rotated qubits, the closed forms of the minimal example with memory (fresh probes, counters of capacity 2 and 3 or more, rotated qubit, archive lost as a whole), and the kernel of correlated intermittent records (Proposition 7) |
| `verify/correlated_records_check.py` | Entangled pairs (Proposition 8): the record of change $F = [a_2 - a_1 \equiv a_2' - a_1' \bmod d]$, the general two-epoch formula and the trace formula for maximally entangled preparations, recoherence and the trace-distance backflow $1 \to 0 \to 1$ (Proposition 10), partial revival $\lvert\cos\phi\rvert$ with a rotation between the records, chains of pairs, and the range of the latitude bound of correlated losses |
| `verify/gaussian_field_check.py` | Gaussian pointer fields (Proposition 9): the overlap of displaced Gaussians by quadrature, the momentum covariance of the pure Gaussian state, the exponential kernel (persistent versus alternating differences, white-noise and ledger limits), and the spectral (filter-function) form of the exponent |
| `verify/retrodiction_check.py` | Section 6 (Propositions 11 and 12): causality (states and forward readout statistics do not depend on later operations), the past-quantum-state formula against brute-force conditional probabilities, the dependence of the retrodicted distribution on later valuations at $\kappa > 0$ and its reduction to forward-backward smoothing at $\kappa = 0$, and the ABL rule |
| `verify/figure_data.py` | Writes `figures/fig_counter.dat` (sensitivity to the valuation of a counter of capacity $D$ over six recorded epochs) and `figures/fig_gauss.dat` (influence functional of the exponential kernel as a function of the correlation $r$), whose values are inlined as pgfplots coordinates in the manuscript |

## Run
```bash
cd verify
for f in influence_functional_check.py identification_check.py backaction_check.py account_memory_check.py ledger_check.py correlated_records_check.py gaussian_field_check.py retrodiction_check.py; do python "$f"; done
python figure_data.py
```
Every check script prints `ALL PASS` when all checks pass; the printed discrepancies are of the order of machine precision (the quadrature and finite-difference checks of `gaussian_field_check.py` are accurate to about $10^{-4}$ or better).

## Conventions
The numbering of theorems and propositions in the table follows the manuscript of 2026-11-04 (Theorems 1--4, Corollaries 1--3, Propositions 1--13). Epochs are numbered from 1 and a record at epoch $t$ is applied after the step $W_t$. The kernel is stored as a matrix indexed by record sequences in the order of `itertools.product`, with `F[i, j] = <Phi_j | Phi_i>` in the convention $F[\mathbf a,\mathbf a'] = \langle\Phi_{\mathbf a'}\vert\Phi_{\mathbf a}\rangle$ of the manuscript.

## Licence
MIT (see `LICENSE`).
