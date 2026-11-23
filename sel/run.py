"""Run episodes x conditions in parallel; append one JSON line per (episode, condition).
Resumable: rows already present in the output file are skipped.

  python3 run.py --out ../results/main.jsonl --Ks 4 6 8 --n 15 --conds isolated summary full_history ledger check sel
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("LLM_PERSIST", "1")
from llm import FreeLLM
from tasks import make_episode
from agent import run_episode, CONDITIONS

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--Ks", type=int, nargs="+", default=[4, 6, 8])
ap.add_argument("--n", type=int, default=15, help="episodes per K")
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--conds", nargs="+", default=CONDITIONS)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--exo_p", type=float, default=0.0)
args = ap.parse_args()

done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        r = json.loads(line)
        done.add((r["seed"], r["K"], r["cond"]))

lock = threading.Lock()
jobs = [(args.seed0 + K * 100 + i, K, c) for K in args.Ks for i in range(args.n) for c in args.conds]
jobs = [j for j in jobs if j not in done]
print(f"{len(jobs)} jobs to run", flush=True)


def work(job):
    seed, K, cond = job
    llm = FreeLLM(log_name="sel_calls", temperature=0.0)
    ep = make_episode(seed, K)
    log = []
    r = run_episode(ep, cond, llm, log, exo_p=args.exo_p)
    r.update({"seed": seed, "K": K, "cond": cond, "model": llm.primary_model, "model_counts": llm.model_counts, "prompt_tokens": llm.prompt_tokens, "completion_tokens": llm.completion_tokens, "exo_p": args.exo_p,
              "interference": ep.interference, "trace": log})
    with lock:
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
    return job, r


with ThreadPoolExecutor(args.workers) as ex:
    futs = [ex.submit(work, j) for j in jobs]
    for k, fu in enumerate(as_completed(futs), 1):
        try:
            (seed, K, cond), r = fu.result()
            print(f"[{k}/{len(jobs)}] K={K} seed={seed} {cond:12s} goal={r['goal_success']} "
                  f"clean={r['clean_success']} clob={r['clobbers']} ling={r['lingering']} "
                  f"steps={r['steps']} err={r.get('llm_errors', 0)}", flush=True)
        except Exception as e:
            print("job failed:", repr(e), flush=True)
