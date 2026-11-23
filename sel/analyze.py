"""Aggregate results: per-condition means with paired-bootstrap 95% CIs, exact paired sign
tests vs. baselines, breakdowns by K and subtask kind, and figures.

  python3 analyze.py ../results/main.jsonl --tag main
Writes ../results/summary_<tag>.json and ../paper/figs/<tag>_*.png
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--tag", default="main")
ap.add_argument("--no_figs", action="store_true")
args = ap.parse_args()

ORDER = ["isolated", "summary", "full_history", "ledger", "check", "check_generic", "sel"]
LABEL = {"isolated": "Isolated", "summary": "Summary notes", "full_history": "Full history",
         "ledger": "Ledger only", "check": "Check only", "check_generic": "Generic re-check",
         "sel": "SEL (ledger+check)"}
METRICS = ["goal_success", "clean_success", "goal_frac", "clobbers", "lingering", "steps",
           "llm_calls", "ktok", "check_rejects", "invalid", "parse_fail", "step_cap_hits"]

rows = [json.loads(l) for l in open(args.path)]
for r in rows:
    r["ktok"] = r["prompt_chars"] / 4 / 1000  # approx. prompt kilo-tokens per episode
conds = [c for c in ORDER if any(r["cond"] == c for r in rows)]
by = defaultdict(dict)  # (seed,K) -> cond -> row
for r in rows:
    by[(r["seed"], r["K"])][r["cond"]] = r
eps = sorted(k for k, v in by.items() if all(c in v for c in conds))  # complete, paired episodes
rng = random.Random(0)
B = 4000


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(vals):
    n = len(vals)
    ms = sorted(mean([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def sign_test(a, b):
    """Exact two-sided sign test on paired outcomes (ties dropped). Returns (wins_a, wins_b, p)."""
    wa = sum(1 for x, y in zip(a, b) if x > y)
    wb = sum(1 for x, y in zip(a, b) if y > x)
    n = wa + wb
    if n == 0:
        return wa, wb, 1.0
    k = min(wa, wb)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return wa, wb, min(1.0, p)


summary = {"n_episodes": len(eps), "n_rows": len(rows), "conditions": {}, "by_K": {}, "tests": {},
           "by_kind": {}, "errors": {}}
for c in conds:
    rs = [by[e][c] for e in eps]
    d = {}
    for m in METRICS:
        v = [float(r.get(m, 0)) for r in rs]
        lo, hi = boot_ci(v)
        d[m] = {"mean": mean(v), "lo": lo, "hi": hi}
    summary["conditions"][c] = d
    summary["errors"][c] = sum(r.get("llm_errors", 0) for r in rs)
    for K in sorted({e[1] for e in eps}):
        rk = [by[e][c] for e in eps if e[1] == K]
        summary["by_K"].setdefault(str(K), {})[c] = {m: mean([float(r.get(m, 0)) for r in rk])
                                                     for m in ("goal_success", "clean_success", "clobbers", "ktok")}
        summary["by_K"][str(K)]["n"] = len(rk)
    kinds = defaultdict(list)
    for r in rs:
        for s in r.get("subtask_end", []):
            kinds[s["kind"]].append(int(s["held_at_end"]))
    summary["by_kind"][c] = {k: {"rate": mean(v), "n": len(v)} for k, v in sorted(kinds.items())}

for base in ("isolated", "summary", "full_history", "check_generic", "ledger", "check"):
    if base in conds and "sel" in conds:
        for m in ("goal_success", "clean_success"):
            a = [by[e]["sel"][m] for e in eps]
            b = [by[e][base][m] for e in eps]
            wa, wb, p = sign_test(a, b)
            diffs = [x - y for x, y in zip(a, b)]
            lo, hi = boot_ci(diffs)
            summary["tests"][f"sel_vs_{base}:{m}"] = {"sel_wins": wa, "base_wins": wb, "p": p,
                                                       "diff": mean(diffs), "lo": lo, "hi": hi}

out = Path(args.path).parent / f"summary_{args.tag}.json"
out.write_text(json.dumps(summary, indent=1))

print(f"paired episodes: {len(eps)}  rows: {len(rows)}")
hdr = f"{'condition':20s} {'goal':>14s} {'clean':>14s} {'clob':>6s} {'ling':>6s} {'steps':>6s} {'calls':>6s} {'ktok':>7s} {'rej':>5s} {'err':>4s}"
print(hdr)
for c in conds:
    d = summary["conditions"][c]
    f = lambda m: f"{d[m]['mean']:.2f}[{d[m]['lo']:.2f},{d[m]['hi']:.2f}]"
    print(f"{LABEL[c]:20s} {f('goal_success'):>14s} {f('clean_success'):>14s} {d['clobbers']['mean']:6.2f} "
          f"{d['lingering']['mean']:6.2f} {d['steps']['mean']:6.1f} {d['llm_calls']['mean']:6.1f} "
          f"{d['ktok']['mean']:7.1f} {d['check_rejects']['mean']:5.2f} {summary['errors'][c]:4d}")
print("\nby K (clean_success):")
for K, v in summary["by_K"].items():
    print(f"  K={K} (n={v['n']}): " + "  ".join(f"{c}={v[c]['clean_success']:.2f}" for c in conds))
print("\nby subtask kind (goal held at end):")
for c in conds:
    print(f"  {c:14s} " + "  ".join(f"{k}={v['rate']:.2f}(n={v['n']})" for k, v in summary["by_kind"][c].items()))
print("\npaired tests (SEL minus baseline):")
for k, v in summary["tests"].items():
    print(f"  {k:36s} diff={v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] wins {v['sel_wins']}:{v['base_wins']} p={v['p']:.4f}")

if not args.no_figs:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = Path(__file__).parent.parent / "paper" / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    # Fig 1: goal / clean success per condition with CIs
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = range(len(conds))
    for j, (m, col) in enumerate((("goal_success", "#4C72B0"), ("clean_success", "#DD8452"))):
        means = [summary["conditions"][c][m]["mean"] for c in conds]
        err = [[means[i] - summary["conditions"][c][m]["lo"] for i, c in enumerate(conds)],
               [summary["conditions"][c][m]["hi"] - means[i] for i, c in enumerate(conds)]]
        ax.bar([i + (j - 0.5) * 0.38 for i in x], means, 0.38, yerr=err, capsize=2, color=col,
               label="all goals hold" if m == "goal_success" else "goals + clean state")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[c].replace(" (ledger+check)", "") for c in conds], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("episode success", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / f"{args.tag}_success.png", dpi=220)
    # Fig 2: clean success vs prompt tokens, by K
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for c, mk in zip(conds, "os^vDPX"):
        xs = [summary["by_K"][K][c]["ktok"] for K in summary["by_K"]]
        ys = [summary["by_K"][K][c]["clean_success"] for K in summary["by_K"]]
        ax.plot(xs, ys, marker=mk, ms=4, lw=1, label=LABEL[c].replace(" (ledger+check)", ""))
    ax.set_xlabel("prompt tokens per episode (k, approx.)", fontsize=8)
    ax.set_ylabel("clean success", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / f"{args.tag}_tokens.png", dpi=220)
    print("figures written to", figdir)
