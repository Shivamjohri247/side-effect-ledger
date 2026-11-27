# Pre-registered replication of the Side-Effect Ledger study

Written and committed before any replication episode was run. Code is unchanged from this commit.

## Why
The reported run used episode seeds 1400-1807 (K = 4/6/8, 8 per K) and a shared response cache;
two baselines (rules-only check, LLM critic) were added after review. This replication uses fresh
seeds and an empty response cache.

## Design
- Episodes: `run.py --seed0 9000 --Ks 4 6 8 --n 8` -> seeds 9400-9407, 9600-9607, 9800-9807
  (24 episodes, same generator). `oracle.py 9000` certified all of them feasible before this commit.
- Conditions: isolated, check, check_rules, sel (96 runs) -> `results/rep_main.jsonl`.
  The other five conditions are not re-run, for time (about 8 s per LLM call; a full replication
  would take about 45 h).
- Same model and settings (Qwen3-14B Q4_K_M via Ollama, temperature 0, seed 0, thinking off).
  Empty cache: `LLM_CACHE_DIR=results/_llm_cache_rep`.
- No episode is excluded. Runs with LLM errors or parse failures are reported, not dropped.

## Primary endpoints (clean success, exact two-sided sign tests on the 24 paired episodes, Holm over 3, alpha = .05)
1. sel higher than isolated (reported: 20:0).
2. sel higher than check (reported: 15:2).
3. sel higher than check_rules (reported: 19:0).

## Secondary (reported, exploratory)
- Goal success, clobbers per episode, lingering violations, prompt tokens, step-cap hits per condition,
  with Clopper-Pearson intervals; the same three comparisons on goal success.
- Per-K clean success (trend not tested).
- How often the known held-object bug lets a `done` through (replay as in the paper), per condition.

## Replication criterion
A primary result replicates if its effect has the same direction as in the reported run. Every
p-value is reported regardless of outcome, and the paper will state the outcome whatever it is.
