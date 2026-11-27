#!/bin/bash
# Pre-registered fresh-seed replication (replication/PREREG.md). Empty response cache. Resumable.
set -e
cd "$(dirname "$0")/sel"
export SEL_BACKEND=ollama OLLAMA_MODEL=qwen3:14b
export LLM_CACHE_DIR=../results/_llm_cache_rep LLM_LOG_DIR=../results/_llm_logs_rep
python3 test_sel.py
python3 oracle.py 9000 | tail -1
python3 run.py --out ../results/rep_main.jsonl --Ks 4 6 8 --n 8 --seed0 9000 --workers 1 \
    --conds isolated check check_rules sel
echo REP DONE
