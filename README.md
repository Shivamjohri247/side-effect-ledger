# Side-Effect Ledger (SEL)

Code, benchmark, and raw results for the paper *Side-Effect Ledgers for Context-Isolated LLM Subtask Workers in Long-Horizon Embodied Tasks*.

A long household job is split into K subtasks. Each subtask is run by a fresh LLM worker that has no memory of the earlier ones. SEL adds two things:

- **Ledger.** It records the effects of executed actions and protects the results of finished subtasks.
- **Commit check.** It rejects a worker's `done` while an earlier result is broken, a container is open, an appliance is on, or the worker's hand is not empty.

## Results (Qwen3-14B via Ollama, 24 paired episodes, K = 4/6/8)

| Condition | Goal success | Clean success | Clobbers / episode | Prompt k-tokens |
|---|---|---|---|---|
| Isolated | .04 | .00 | 1.50 | 53 |
| Summary notes | .04 | .00 | 1.58 | 67 |
| Full history | .04 | .00 | 1.33 | 160 |
| Ledger only | .04 | .00 | 1.71 | 49 |
| Check only | .29 | .29 | 0.38 | 114 |
| Generic re-check | .04 | .00 | 1.71 | 59 |
| Rules-only check (added after review) | .04 | .04 | 1.62 | 70 |
| LLM critic (added after review) | .08 | .00 | 0.42 | 208 |
| **SEL** | **.83** | **.83** | **0.17** | 112 |

- Clobbers are earlier goals undone by a later subtask.
- On clean success SEL beats every handoff baseline, ledger only, the generic re-check and the LLM critic 20:0 (exact sign test p = 1.9e-6, Holm-adjusted 3.1e-5), the rules-only check 19:0 (Holm 3.8e-5), and check only 15:2 (p = 0.0023, Holm 0.0047).
- The check enforces exactly the conditions that are scored, so these results measure whether workers can repair what an exact verifier reports.
- Disturbance stress test (`results/exo.jsonl`, 12 episodes): SEL .42 clean, baselines .00; not significant after Holm adjustment.

## Layout

- `sel/sim.py`: symbolic household simulator with partial observability and grasp slips. Each action returns its effects.
- `sel/tasks.py`: procedural episode generator.
- `sel/oracle.py`: scripted full-state solver. It certifies that every episode can be solved within the step budget.
- `sel/agent.py`: the nine conditions, the ledger, and the commit check.
- `sel/llm.py`: LLM client for local Ollama, NVIDIA NIM, or OpenRouter. API keys are read from the environment or a local auth file and are never stored here.
- `sel/run.py`: resumable parallel runner.
- `sel/analyze.py`: Clopper-Pearson and paired-bootstrap intervals, Holm-adjusted sign tests, and figures.
- `sel/test_sel.py`: deterministic tests.
- `sel/fetch_refs.py`: helper used to fetch reference metadata.
- `verify/recompute.py`: independent recomputation of the original-run numbers; `verify/stats_exact.py`: exact intervals, sign test, Holm.
- `verify/recompute_v2.py`: independent recomputation of every number in the revised paper (main + extension + disturbance runs); `verify/critic_rejections.py`: classification of the LLM critic's rejections.
- `results/main.jsonl`, `results/exo.jsonl`, `results/ext.jsonl`: raw per-episode rows, each including the full trace.
- `results/summary_*.json`: the aggregated results.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install httpx numpy matplotlib
ollama pull qwen3:14b
cd sel
python3 test_sel.py && python3 oracle.py && python3 oracle.py 5000   # main and disturbance seeds
SEL_BACKEND=ollama python3 run.py --out ../results/main.jsonl --Ks 4 6 8 --n 8 --workers 1 \
    --conds isolated summary full_history ledger check check_generic sel
SEL_BACKEND=ollama python3 run.py --out ../results/exo.jsonl --Ks 4 6 8 --n 4 \
    --conds isolated full_history sel --exo_p 0.5 --seed0 5000 --workers 1
# added after review: rules-only check and LLM critic on the same episodes
SEL_BACKEND=ollama python3 run.py --out ../results/ext.jsonl --Ks 4 6 8 --n 8 --workers 1 \
    --conds check_rules critic
../.venv/bin/python analyze.py ../results/main.jsonl ../results/ext.jsonl --tag main
../.venv/bin/python analyze.py ../results/exo.jsonl --tag exo
```

`run_all.sh` runs the first three study steps and `run_ext.sh` the review extension. The main study takes
about 30 hours on a laptop GPU and the extension about 15 hours (the critic is slow at K=8).

## Known issues (found in review; kept so the code matches the reported runs)

- `Ledger.apply` clears `holding` when a wipe sweeps objects, so the commit check can miss "put down the cloth" right after a wipe. Replaying the runs shows this never let a wrong `done` through under `check` or `sel` in the main study, and twice under `sel` in the disturbance study (episodes that ended with no rule violations). Under `check_rules` it let a `done` through with an object still held 34 times in 21 of 24 episodes (almost always the cloth right after a wipe), so the rules-only lingering count is partly a product of this bug.
- `llm.extract_json` can fail on a reply whose `thought` contains a colon followed by a comma. This never happened in the logged runs: `parse_fail` is 0 everywhere.
- `sel/tasks.py` used to iterate a `set`, so the episodes depended on `PYTHONHASHSEED`. It now uses the fixed order the main run used, which reproduces every committed episode (main, extension and disturbance runs) under any hash seed. A first fix used a different order, which changed seeds 1404 and 1600; the extension rows run on those altered episodes are kept in `results/ext_mispaired.jsonl`, and both jobs were re-run on the correct episodes.

## License

MIT
