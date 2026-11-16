#!/usr/bin/env bash
# m6-12c lane: run every probe through the requested tools on the Apple M6
# and validate the rows. Used by the self-hosted job in workflows/matrix.yml;
# runs the same way from a local checkout.
#
#   .github/scripts/m6-lane.sh [all|mvcc|cumetal]
#
# Tools come from $CUPORT_TOOLS (default ~/.local/cuport-tools), built by
# the pinned build script; matrix/tools.yaml lists the pins.
set -euo pipefail

tool=${1:-all}
case $tool in
  all) tools="mvcc cumetal" ;;
  mvcc|cumetal) tools=$tool ;;
  *) echo "unknown tool: $tool (want all, mvcc or cumetal)" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"
export PATH="$HOME/.local/bin:$PATH"

# Rows are labelled m6-12c, so refuse to write them from any other machine.
gpu=$(system_profiler SPDisplaysDataType)
if ! grep -q "Chipset Model: Apple M6" <<<"$gpu" || ! grep -q "Total Number of Cores: 12" <<<"$gpu"; then
  echo "this is not the m6-12c machine:" >&2
  echo "$gpu" >&2
  exit 1
fi
sw_vers

uv venv -q --clear --python 3.13 .venv
uv pip install -q --python .venv/bin/python -r runner/requirements.txt
export PYTHONPATH=runner

.venv/bin/python -m pytest runner -q
for t in $tools; do
  .venv/bin/python -m cuport run --tool "$t" --hardware m6-12c \
    --jobs "$(sysctl -n hw.ncpu)" --gpu-jobs 4 --force
done
.venv/bin/python -m cuport validate results/*/m6-12c/*.json
