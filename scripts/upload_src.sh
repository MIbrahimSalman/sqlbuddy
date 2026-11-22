#!/usr/bin/env bash
# Upload the shared src/ package (and the demo app/) to the private Kaggle dataset <user>/sqlbuddy-src,
# so that every Kaggle notebook imports exactly the code that was tested here.
#   bash scripts/upload_src.sh "message describing the change"
set -euo pipefail
cd "$(dirname "$0")/.."
MSG="${1:-update src}"
# Kaggle owner slug = username without dots (e.g. "m.name" -> "mname")
OWNER="${KAGGLE_OWNER:-${KAGGLE_USERNAME//./}}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# copy the package (+ demo app and sample DB for notebook 04) without caches
tar --exclude='__pycache__' -cf - src app sample_db/university.db sample_db/make_university_db.py | tar -xf - -C "$STAGE"
git rev-parse --short HEAD > "$STAGE/src/GIT_COMMIT" 2>/dev/null || true
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "sqlbuddy-src", "id": "${OWNER}/sqlbuddy-src", "licenses": [{"name": "Apache 2.0"}]}
JSON
if kaggle datasets status "${OWNER}/sqlbuddy-src" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "$MSG" --dir-mode zip -q
else
  kaggle datasets create -p "$STAGE" --dir-mode zip -q   # private by default
fi
