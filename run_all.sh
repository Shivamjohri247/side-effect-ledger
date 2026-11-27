#!/bin/bash
# Full Study-1 run (local Qwen3-14B via Ollama). Resumable: rerun to continue.
set -e
cd "$(dirname "$0")/sel"
export SEL_BACKEND=ollama OLLAMA_MODEL=qwen3:14b
python3 test_sel.py
python3 oracle.py | tail -1
python3 oracle.py 5000 | tail -1
python3 run.py --out ../results/main.jsonl --Ks 4 6 8 --n 8 --workers 1 \
    --conds isolated summary full_history ledger check check_generic sel
python3 run.py --out ../results/exo.jsonl --Ks 4 6 8 --n 4 --conds isolated full_history sel --exo_p 0.5 --seed0 5000 --workers 1
echo ALL DONE
