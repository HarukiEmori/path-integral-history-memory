"""Memory witness for records with memory: circuits, ideal values, analysis, and execution through Quantinuum Nexus.

Manuscript: "A Non-Markovian Extension of a Path Integral Model of History" (H. Emori and A. Iriki), Sec. 8 and
Proposition 8(ii) (record of change).  The course is one qubit s (two configurations); the archive is a pair of
qubits a1, a2.  A record of the configuration at an epoch is a CNOT from s to the archive qubit of that epoch.

Two-path interferometer.  s is prepared in (|0> + e^{i theta}|1>)/sqrt(2) (two courses with a relative phase),
the records are taken with no dynamics in between, and s is recombined by a Hadamard gate, so that

    P(s = 0) = (1 + V cos(theta)) / 2,      V = F[(0,..),(1,..)]  (overlap of the archive states of the two courses).

For a memoryless record process the overlap is a product of single-record overlaps and cannot rise when a record
is added.  The witness of memory is a visibility that rises:

    pair_one   archive prepared in the Bell state, one record    ->  V = 0
    pair_two   archive prepared in the Bell state, two records   ->  V = 1   (the second record restores the interference)
    fresh_two  archive prepared in |00> (fresh probes), two records ->  V = 0   (memoryless control)

Each configuration is run at theta = 0 and theta = pi, and V is estimated by P0(theta = 0) - P0(theta = pi).

Consulted archive (Proposition 11, example of the entangled pair; emulator only by default).  Between the two
records the phase vartheta is applied when the pair shows the first record a_1 = 1, which is the gate
ZZPhase on the two archive qubits up to a global phase.  Then P(s = 0) = (1 + cos(vartheta)) / 2, although after the second
record the archive holds only the change a_1 - a_2 = 0 of both courses.

Usage (Python of ~/Develop/venv/quantum-2026; pytket and, for remote runs, qnexus with a stored login):
    python memory_witness.py --local                         # ideal values from the state vectors of the circuits
    python memory_witness.py --submit reimei-E --shots 1000  # emulator (set: all)
    python memory_witness.py --submit reimei --shots 200 --set witness --confirm-hardware --max-total-cost 60
    python memory_witness.py --status results/reimei/submission_<tag>.json
    python memory_witness.py --fetch  results/reimei/submission_<tag>.json
Nothing is submitted without --submit; a processor (a device whose name does not end in E, SC or simulator)
also needs --confirm-hardware and an estimated cost not above --max-total-cost.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pytket import Circuit, OpType

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = "path-integral-history-memory"
S, A1, A2 = 0, 1, 2  # qubit indices; classical bit i records qubit i


# ----------------------------------------------------------------------------------------- circuits
def witness_circuit(prep: str, n_records: int, theta: float, consult: Optional[float] = None) -> Circuit:
    """prep: 'pair' (Bell state of a1, a2) or 'fresh' (|00>).  theta: phase between the two courses (radians).
    consult: if not None, the phase applied between the two records when the pair shows a_1 = 1."""
    c = Circuit(3, 3)
    if prep == "pair":
        c.H(A1)
        c.CX(A1, A2)
    elif prep != "fresh":
        raise ValueError(prep)
    c.H(S)
    c.Rz(theta / math.pi, S)                       # pytket angles are in half-turns
    c.CX(S, A1)                                    # record of epoch 1
    if consult is not None:
        # e^{i consult} on the odd-parity subspace of (a1, a2), up to a global phase: ZZPhase(-consult/pi)
        c.ZZPhase(-consult / math.pi, A1, A2)
    if n_records == 2:
        c.CX(S, A2)                                # record of epoch 2
    c.H(S)
    for q in (S, A1, A2):
        c.Measure(q, q)
    return c


def experiment_set(which: str = "all") -> List[Dict]:
    """List of dicts {name, group, circuit, p0_ideal}.  which: witness | control | consulted | all."""
    items = []

    def add(name, group, circ):
        items.append({"name": name, "group": group, "circuit": circ, "p0_ideal": ideal_p0(circ)})

    if which in ("witness", "all"):
        for nrec, label in ((1, "pair_one"), (2, "pair_two")):
            for th, tl in ((0.0, "theta0"), (math.pi, "thetapi")):
                add(f"{label}_{tl}", label, witness_circuit("pair", nrec, th))
    if which in ("control", "all"):
        for th, tl in ((0.0, "theta0"), (math.pi, "thetapi")):
            add(f"fresh_two_{tl}", "fresh_two", witness_circuit("fresh", 2, th))
    if which in ("consulted", "all"):
        for vt, vl in ((0.0, "v0"), (math.pi / 2, "vhalfpi"), (math.pi, "vpi")):
            add(f"consulted_{vl}", "consulted", witness_circuit("pair", 2, 0.0, consult=vt))
    if not items:
        raise ValueError(f"unknown set {which!r}")
    return items


# ----------------------------------------------------------------------------------------- ideal values
def ideal_p0(circ: Circuit) -> float:
    """P(s = 0) from the state vector of the circuit without its measurements (pytket convention: qubit 0 is
    the most significant bit of the basis index)."""
    c = Circuit(3)
    for cmd in circ:
        if cmd.op.type == OpType.Measure:
            continue
        c.add_gate(cmd.op, [q.index[0] for q in cmd.qubits])
    psi = c.get_statevector().reshape(2, 2, 2)
    return float(np.sum(np.abs(psi[0]) ** 2))


def ideal_table() -> Dict[str, float]:
    return {it["name"]: it["p0_ideal"] for it in experiment_set("all")}


# ----------------------------------------------------------------------------------------- analysis
def p0_from_counts(counts: Dict[str, int]) -> Tuple[float, float, int]:
    """counts: dict from bit strings 'c0c1c2' (c0 = system) to integers.  Returns (P(s=0), standard error, shots)."""
    n = sum(counts.values())
    k = sum(v for b, v in counts.items() if b[0] == "0")
    p = k / n
    return p, math.sqrt(max(p * (1 - p), 1e-12) / n), n


def analyse(results: Dict[str, Dict[str, int]]) -> Dict:
    """results: name -> counts.  Visibility of a group = P0(theta0) - P0(thetapi) with its standard error."""
    out = {"p0": {}, "visibility": {}}
    for name, counts in results.items():
        p, se, n = p0_from_counts(counts)
        out["p0"][name] = {"p0": p, "stderr": se, "shots": n, "ideal": ideal_table().get(name)}
    for group in ("pair_one", "pair_two", "fresh_two"):
        a, b = f"{group}_theta0", f"{group}_thetapi"
        if a in out["p0"] and b in out["p0"]:
            v = out["p0"][a]["p0"] - out["p0"][b]["p0"]
            se = math.hypot(out["p0"][a]["stderr"], out["p0"][b]["stderr"])
            out["visibility"][group] = {"V": v, "stderr": se, "ideal": {"pair_one": 0.0, "pair_two": 1.0, "fresh_two": 0.0}[group]}
    vis = out["visibility"]
    if "pair_one" in vis and "pair_two" in vis:
        rise = vis["pair_two"]["V"] - vis["pair_one"]["V"]
        se = math.hypot(vis["pair_two"]["stderr"], vis["pair_one"]["stderr"])
        out["witness"] = {"rise": rise, "stderr": se, "sigmas": rise / se if se > 0 else float("inf"),
                          "statement": "a memoryless record process cannot raise the visibility when a record is added"}
    return out


# ----------------------------------------------------------------------------------------- cost
def hqc_formula_estimate(circ: Circuit, shots: int) -> float:
    """Rough estimate 5 + shots (N1q + 10 N2q + 5 Nm) / 5000 for a compiled circuit (a preview only; on reimei the
    cost reported by Nexus for comparable circuits on 2026-11-04 was about 1.18 times this value)."""
    n1 = n2 = nm = 0
    for cmd in circ:
        t = cmd.op.type
        if t == OpType.Measure:
            nm += 1
        elif t in (OpType.Barrier, OpType.Reset):
            continue
        elif len(cmd.qubits) == 1:
            n1 += 1
        elif len(cmd.qubits) == 2:
            n2 += 1
    return 5.0 + shots * (n1 + 10 * n2 + 5 * nm) / 5000.0


def is_processor(device: str) -> bool:
    d = device.lower()
    return not (d.endswith("e") or d.endswith("sc") or d.endswith("le") or "simulator" in d or "emulator" in d)


# ----------------------------------------------------------------------------------------- Nexus
def _nexus():
    try:
        import qnexus as qnx
    except ImportError as e:  # pragma: no cover
        raise SystemExit("qnexus is required for remote execution") from e
    return qnx


def compile_on_nexus(items: Sequence[Dict], device: str, tag: str, optimisation_level: int = 2):
    qnx = _nexus()
    project = qnx.projects.get_or_create(name=PROJECT)
    qnx.context.set_active_project(project)
    cfg = qnx.QuantinuumConfig(device_name=device)
    refs = [qnx.circuits.upload(circuit=it["circuit"], name=f"{tag}-{it['name']}", project=project) for it in items]
    job = qnx.start_compile_job(programs=refs, backend_config=cfg, name=f"{tag}-compile-{device}", project=project,
                                optimisation_level=optimisation_level)
    qnx.jobs.wait_for(job)
    by_input = {str(r.get_input().id): r.get_output() for r in qnx.jobs.results(job)}
    compiled = [by_input[str(ref.id)] for ref in refs]
    return project, cfg, refs, compiled, job


def submit(device: str, shots: int, which: str, confirm_hardware: bool, max_total_cost: float,
           per_circuit_margin: float = 1.5) -> Dict:
    qnx = _nexus()
    items = experiment_set(which)
    tag = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    project, cfg, refs, compiled, compile_job = compile_on_nexus(items, device, tag)
    compiled_circuits = [c.download_circuit() for c in compiled]
    est = [hqc_formula_estimate(c, shots) for c in compiled_circuits]
    total = sum(est)
    print(f"device {device}, {len(items)} circuits, {shots} shots each; rough cost {total:.1f} HQC "
          f"({', '.join(f'{e:.1f}' for e in est)})")
    hardware = is_processor(device)
    if hardware and not confirm_hardware:
        raise SystemExit(f"not submitted: {device} is a processor; pass --confirm-hardware")
    if hardware and total > max_total_cost:
        raise SystemExit(f"not submitted: rough cost {total:.1f} HQC exceeds --max-total-cost {max_total_cost}")
    max_cost = [math.ceil(per_circuit_margin * e) for e in est] if hardware else []
    job = qnx.start_execute_job(programs=compiled, n_shots=[shots] * len(compiled), backend_config=cfg,
                                name=f"{tag}-execute-{device}", project=project, max_cost=max_cost)
    record = {
        "tag": tag, "device": device, "shots": shots, "set": which, "project": PROJECT,
        "compile_job_id": str(compile_job.id), "execute_job_id": str(job.id),
        "rough_cost_hqc": est, "max_cost_hqc": max_cost,
        "items": [{"index": i, "name": it["name"], "group": it["group"], "p0_ideal": it["p0_ideal"],
                   "circuit_id": str(r.id), "compiled_id": str(c.id),
                   "two_qubit_gates": int(sum(1 for cmd in cc if len(cmd.qubits) == 2 and cmd.op.type != OpType.Barrier)),
                   "gates": int(cc.n_gates)}
                  for i, (it, r, c, cc) in enumerate(zip(items, refs, compiled, compiled_circuits))],
        "submitted_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    outdir = os.path.join(ROOT, "results", device)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"submission_{tag}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    print("submitted; record:", os.path.relpath(path, ROOT))
    return record


def status(record: Dict) -> Dict:
    qnx = _nexus()
    project = qnx.projects.get(name=record["project"])
    qnx.context.set_active_project(project)
    st = qnx.jobs.status(qnx.jobs.get(id=record["execute_job_id"]))
    out = {"status": str(getattr(st, "status", st)).split(".")[-1]}
    for attr in ("message", "error_detail", "cost", "queue_position", "submitted_time", "running_time", "completed_time"):
        if hasattr(st, attr) and getattr(st, attr) is not None:
            out[attr] = str(getattr(st, attr))
    return out


def fetch(record: Dict, wait: bool = False) -> Dict:
    qnx = _nexus()
    project = qnx.projects.get(name=record["project"])
    qnx.context.set_active_project(project)
    job = qnx.jobs.get(id=record["execute_job_id"])
    if wait:
        qnx.jobs.wait_for(job)
    by_input = {}
    for r in qnx.jobs.results(job, allow_incomplete=not wait):
        if hasattr(r, "download_result"):
            try:
                by_input[str(r.get_input().id)] = r
            except Exception:
                continue
    results = {}
    for it in record["items"]:
        r = by_input.get(it["compiled_id"])
        if r is None:
            continue
        br = r.download_result()
        bits = sorted(br.c_bits.keys(), key=lambda b: b.index[0])
        results[it["name"]] = {"".join(str(int(x)) for x in k): int(v) for k, v in br.get_counts(cbits=bits).items()}
    return results


# ----------------------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--local", action="store_true", help="print the ideal values and gate counts")
    ap.add_argument("--submit", metavar="DEVICE")
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--set", default="all", choices=("witness", "control", "consulted", "all"))
    ap.add_argument("--confirm-hardware", action="store_true")
    ap.add_argument("--max-total-cost", type=float, default=60.0)
    ap.add_argument("--status", metavar="RECORD")
    ap.add_argument("--fetch", metavar="RECORD")
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args(argv)
    if a.local:
        ok = True
        expected = {"pair_one_theta0": 0.5, "pair_one_thetapi": 0.5, "pair_two_theta0": 1.0, "pair_two_thetapi": 0.0,
                    "fresh_two_theta0": 0.5, "fresh_two_thetapi": 0.5, "consulted_v0": 1.0, "consulted_vhalfpi": 0.5,
                    "consulted_vpi": 0.0}
        for it in experiment_set("all"):
            good = abs(it["p0_ideal"] - expected[it["name"]]) < 1e-12
            ok &= good
            n2 = sum(1 for cmd in it["circuit"] if len(cmd.qubits) == 2)
            print(f"{'ok  ' if good else 'FAIL'} {it['name']:<20} P(s=0) = {it['p0_ideal']:.6f}  (expected {expected[it['name']]:.1f}; {n2} two-qubit gates)")
        print("ALL PASS" if ok else "SOME FAILED")
        return 0 if ok else 1
    if a.submit:
        submit(a.submit, a.shots, a.set, a.confirm_hardware, a.max_total_cost)
        return 0
    if a.status:
        print(json.dumps(status(json.load(open(a.status))), indent=1))
        return 0
    if a.fetch:
        record = json.load(open(a.fetch))
        results = fetch(record, wait=a.wait)
        if len(results) < len(record["items"]):
            print(f"{len(results)} of {len(record['items'])} results available; status: {status(record)}")
            if not results:
                return 0
        outdir = os.path.dirname(os.path.abspath(a.fetch))
        with open(os.path.join(outdir, f"counts_{record['tag']}.json"), "w") as f:
            json.dump({"record": os.path.basename(a.fetch), "device": record["device"], "shots": record["shots"],
                       "counts": results}, f, indent=1)
        summary = analyse(results)
        with open(os.path.join(outdir, f"analysis_{record['tag']}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps(summary, indent=1))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
