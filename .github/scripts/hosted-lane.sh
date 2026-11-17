#!/usr/bin/env bash
# paravirtual lane: run every probe through the requested tools on a
# GitHub-hosted macOS runner and validate the rows. Used by the
# hosted-paravirtual job in workflows/matrix.yml.
#
#   .github/scripts/hosted-lane.sh [all|mvcc|cumetal]
#
# The hosted runner exposes Metal through an "Apple Paravirtual device", a
# feature subset of real Apple silicon. Rows are labelled `paravirtual` and
# never stand in for an M6 result. Tools come from $CUPORT_TOOLS, built by
# .github/scripts/build-tools.sh.
set -euo pipefail

tool=${1:-all}
case $tool in
  all) tools="mvcc cumetal" ;;
  mvcc|cumetal) tools=$tool ;;
  *) echo "unknown tool: $tool (want all, mvcc or cumetal)" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"

# Rows are labelled paravirtual, so refuse to write them from real hardware.
gpu=$(system_profiler SPDisplaysDataType)
echo "$gpu"
sw_vers
if ! grep -q "Apple Paravirtual device" <<<"$gpu"; then
  echo "this runner's GPU is not the Apple Paravirtual device" >&2
  exit 1
fi

python3 -m venv --clear .venv
.venv/bin/python -m pip install -q -r runner/requirements.txt
export PYTHONPATH=runner

.venv/bin/python -m pytest runner -q
.venv/bin/python -m cuport check

status=0
for t in $tools; do
  # A tool that fails to build or run still gets its other tool's row.
  .venv/bin/python -m cuport run --tool "$t" --hardware paravirtual \
    --jobs "$(sysctl -n hw.ncpu)" --gpu-jobs 2 --force || status=1
done
shopt -s nullglob
rows=(results/*/paravirtual/*.json)
if [ ${#rows[@]} -eq 0 ]; then
  echo "no paravirtual rows written" >&2
  exit 1
fi
.venv/bin/python -m cuport validate "${rows[@]}"
exit $status
