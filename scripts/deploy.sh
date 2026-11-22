#!/usr/bin/env bash
# One command after you change the app (or src/): check it, publish it, and prove Kaggle runs it.
#
#   bash scripts/deploy.sh            # full check + quick Kaggle test run (~5 min)
#   bash scripts/deploy.sh --no-kaggle-test   # skip the Kaggle test run
#
# How the demo gets your changes:
#   * Kaggle notebook 04 downloads the app from GitHub (branch main) at the START OF EVERY RUN.
#     So a change reaches Kaggle as soon as it is merged into main - there is no separate upload.
#   * Your laptop runs whatever is in your local folder (python app/app.py).
#
# What this script does:
#   1. runs the tests (all 56 if SPIDER_DIR is set, otherwise the 52 that need no Spider data);
#   2. checks your work is committed, pushed and merged into GitHub main (that is what Kaggle runs);
#   3. refreshes the Kaggle dataset sqlbuddy-src (used by the evaluation notebooks 01-03);
#   4. rebuilds the notebooks and pushes notebook 04 as a short test run on a Kaggle GPU
#      (a CLI run lasts 1 minute; your own "Run all" in the Kaggle editor lasts 90 minutes),
#      then prints which GitHub commit Kaggle ran and the self-test answers of both models.
#
# Needs: git, python, the kaggle CLI with KAGGLE_USERNAME + KAGGLE_API_TOKEN. Tokens are never printed.
set -euo pipefail
cd "$(dirname "$0")/.."
OWNER="${KAGGLE_OWNER:-${KAGGLE_USERNAME//./}}"
KERNEL="${OWNER}/sqlbuddy-04-demo"
KAGGLE_TEST=1
[[ "${1:-}" == "--no-kaggle-test" ]] && KAGGLE_TEST=0

step() { printf '\n=== %s\n' "$*"; }
fail() { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

step "1/4 tests"
python -m pytest -q tests || fail "tests failed - fix them before deploying"

step "2/4 git: committed, pushed and merged into main?"
[[ -z "$(git status --porcelain)" ]] || fail "uncommitted changes - commit (and push) them first"
git fetch -q origin
HEAD_SHA="$(git rev-parse HEAD)"
MAIN_SHA="$(git rev-parse origin/main)"
if ! git merge-base --is-ancestor "$HEAD_SHA" "$MAIN_SHA"; then
  fail "your commit ${HEAD_SHA:0:7} is not in GitHub main yet (main = ${MAIN_SHA:0:7}).
Kaggle runs GitHub main, so push your branch and merge it (pull request) first."
fi
echo "OK: GitHub main = ${MAIN_SHA:0:7} contains your work"

step "3/4 refresh Kaggle dataset sqlbuddy-src (for notebooks 01-03)"
bash scripts/upload_src.sh "deploy ${MAIN_SHA:0:7}"

step "4/4 rebuild notebooks + Kaggle test run of notebook 04"
python notebooks/build_notebooks.py >/dev/null
if [[ $KAGGLE_TEST -eq 0 ]]; then
  echo "skipped (--no-kaggle-test). Kaggle will still run main ${MAIN_SHA:0:7} on its next run."
  exit 0
fi
if kaggle kernels status "$KERNEL" 2>/dev/null | grep -q RUNNING; then
  fail "notebook 04 is running right now (maybe someone is presenting). Try again later, or use --no-kaggle-test."
fi
PUSH="$(kaggle kernels push -p notebooks/04_demo --accelerator NvidiaTeslaT4 | tail -1)"
echo "$PUSH"
VERSION="$(echo "$PUSH" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' || true)"
[[ -n "$VERSION" ]] || fail "push failed: $PUSH"
RUN="${KERNEL}/${VERSION}"     # always ask about THIS version (the plain name can report the previous run)
echo "waiting for the Kaggle test run v${VERSION} (about 4-6 minutes) ..."
sleep 30
while kaggle kernels status "$RUN" 2>/dev/null | grep -qE "RUNNING|QUEUED"; do sleep 30; done
STATUS="$(kaggle kernels status "$RUN" 2>&1)"
echo "$STATUS"
if echo "$STATUS" | grep -q CANCEL; then
  fail "Kaggle cancelled the test run v${VERSION} before it produced output.
Usual cause: another Kaggle GPU session is still open on this account (e.g. notebook 04 started from the
Kaggle editor). Stop it under kaggle.com -> 'View Active Events', then run this script again."
fi
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
kaggle kernels output "$RUN" -p "$OUT" -q >/dev/null 2>&1 || true
LOG="$(ls "$OUT"/*.log 2>/dev/null | head -1 || true)"
[[ -n "$LOG" ]] || fail "no Kaggle log found - open https://www.kaggle.com/code/${KERNEL}"
# print the lines that prove which app version ran and what the two models answered
python - "$LOG" "${MAIN_SHA:0:7}" <<'PY'
import json, re, sys
log, want = sys.argv[1], sys.argv[2]
text = "".join(e.get("data", "") for e in json.load(open(log)))
keep = [l for l in text.splitlines()
        if re.search(r"app version|device:|run type|Done ·|SQL error|Error|Traceback|PUBLIC LINK|SELECT", l)]
print("\n".join(l[:200] for l in keep))
ok = f"commit {want}" in text and "adapter loaded: True" in text and "Done ·" in text
print("\nRESULT:", "OK - Kaggle ran GitHub main %s with the fine-tuned adapter" % want if ok
      else "CHECK THE LOG ABOVE - something is not as expected")
sys.exit(0 if ok else 1)
PY
echo
echo "Done. For the presentation: open the notebook on Kaggle, GPU T4 + Internet on, Run all (90 min link)."
