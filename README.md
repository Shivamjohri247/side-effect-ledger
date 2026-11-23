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
| **SEL** | **.83** | **.83** | **0.17** | 112 |

- Clobbers are earlier goals undone by a later subtask.
- SEL beats every baseline 20:0 on clean success (exact sign test, p = 1.9e-6) and beats check only 15:2 (p = 0.0023).

## Layout

- `sel/sim.py`: symbolic household simulator with partial observability and grasp slips. Each action returns its effects.
- `sel/tasks.py`: procedural episode generator.
- `sel/oracle.py`: scripted full-state solver. It certifies that every episode can be solved within the step budget.
- `sel/agent.py`: the seven conditions, the ledger, and the commit check.
- `sel/llm.py`: LLM client for local Ollama, NVIDIA NIM, or OpenRouter. API keys are read from the environment or a local auth file and are never stored here.
- `sel/run.py`: resumable parallel runner.
- `sel/analyze.py`: paired-bootstrap confidence intervals, sign tests, and figures.
- `sel/test_sel.py`: deterministic tests.
- `results/main.jsonl`, `results/exo.jsonl`: raw per-episode rows, each including the full trace.
- `results/summary_*.json`: the aggregated results.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install httpx numpy matplotlib
ollama pull qwen3:14b
cd sel
python3 test_sel.py && python3 oracle.py
SEL_BACKEND=ollama python3 run.py --out ../results/main.jsonl --Ks 4 6 8 --n 8 --workers 1
SEL_BACKEND=ollama python3 run.py --out ../results/exo.jsonl --Ks 4 6 8 --n 4 \
    --conds isolated full_history sel --exo_p 0.5 --seed0 5000
../.venv/bin/python analyze.py ../results/main.jsonl --tag main
```

`run_all.sh` runs all of these steps in order. The full study takes about 30 hours on a laptop GPU.

## License

MIT
