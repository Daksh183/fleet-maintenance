#!/usr/bin/env bash
# Execute every per-version notebook in place (3 at a time). Usage: cd notebooks && ./run_all.sh
set -u
run() { ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=fleet \
        --ExecutePreprocessor.timeout=3600 "$1" > "logs/$1.log" 2>&1 && echo "OK   $1" || echo "FAIL $1"; }
export -f run; mkdir -p logs
ls v[0-9]*_*.ipynb | xargs -P 3 -I{} bash -c 'run {}'
