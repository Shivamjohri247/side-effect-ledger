#!/bin/bash
# Post-review extension: two new conditions on the same 24 main-study episodes (paired). Resumable.
set -e
cd "$(dirname "$0")/sel"
export SEL_BACKEND=ollama OLLAMA_MODEL=qwen3:14b
python3 test_sel.py
python3 run.py --out ../results/ext.jsonl --Ks 4 6 8 --n 8 --conds check_rules critic --workers 1
echo EXT DONE
