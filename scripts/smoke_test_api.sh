#!/usr/bin/env bash
# Exercise the production API contract after the service is running.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="${1:-${API_BASE_URL:-http://localhost:8000}}"
TIMEOUT_SECONDS="${API_SMOKE_TIMEOUT_SECONDS:-90}"
UNKNOWN_CUSTOMER_ID="${UNKNOWN_CUSTOMER_ID:-ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff}"
SHORT_CUSTOMER_ID="abc"

find_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
  elif [[ -x ".venv/bin/python" ]]; then
    printf '%s\n' ".venv/bin/python"
  elif [[ -x "../.venv/bin/python" ]]; then
    printf '%s\n' "../.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    printf 'ERROR: no Python interpreter found. Set PYTHON_BIN=/path/to/python.\n' >&2
    return 1
  fi
}

if ! command -v curl >/dev/null 2>&1; then
  printf 'ERROR: curl is required for API smoke tests.\n' >&2
  exit 1
fi

SMOKE_FILE="smoke_test_customers.txt"
if [[ ! -f "$SMOKE_FILE" && -f "tests/fixtures/smoke_test_customers.txt" ]]; then
  SMOKE_FILE="tests/fixtures/smoke_test_customers.txt"
fi

if [[ ! -f "$SMOKE_FILE" ]]; then
  printf 'ERROR: %s is missing. Run bash scripts/bootstrap_production_eval.sh or bash scripts/build_production_artifacts.sh first.\n' "$SMOKE_FILE" >&2
  exit 1
fi

REAL_CUSTOMER_ID="$(sed -n '/^[[:space:]]*$/!{s/[[:space:]]//g;p;q;}' "$SMOKE_FILE")"
if [[ -z "$REAL_CUSTOMER_ID" ]]; then
  printf 'ERROR: %s does not contain a usable customer_id.\n' "$SMOKE_FILE" >&2
  exit 1
fi

PYTHON="$(find_python)"
HEALTH_JSON="$(mktemp /tmp/hm-recsys-health.XXXXXX.json)"
REAL_JSON="$(mktemp /tmp/hm-recsys-real.XXXXXX.json)"
COLD_JSON="$(mktemp /tmp/hm-recsys-cold.XXXXXX.json)"
trap 'rm -f "$HEALTH_JSON" "$REAL_JSON" "$COLD_JSON"' EXIT

printf '==> Waiting for %s/health (timeout: %ss)...\n' "$BASE_URL" "$TIMEOUT_SECONDS"
deadline=$((SECONDS + TIMEOUT_SECONDS))
until curl -fsS "$BASE_URL/health" -o "$HEALTH_JSON" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    printf 'ERROR: API did not become healthy within %s seconds.\n' "$TIMEOUT_SECONDS" >&2
    exit 1
  fi
  sleep 2
done

HEALTH_JSON="$HEALTH_JSON" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path(os.environ["HEALTH_JSON"]).read_text(encoding="utf-8"))
checks = {
    "model_loaded": data.get("model_loaded") is True,
    "indexed_customers": data.get("indexed_customers", 0) > 0,
    "catalog_customers": data.get("catalog_customers", 0) > 0,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"ERROR: /health failed readiness checks: {failed}; payload={data}")
print(
    "[OK] /health: "
    f"model_loaded={data['model_loaded']}, "
    f"indexed_customers={data['indexed_customers']}, "
    f"catalog_customers={data['catalog_customers']}"
)
PY

printf '==> Checking docs and OpenAPI schema...\n'
curl -fsS "$BASE_URL/docs" -o /dev/null
curl -fsS "$BASE_URL/openapi.json" -o /dev/null

printf '==> Checking personalized recommendation for indexed customer...\n'
curl -fsS -X POST "$BASE_URL/recommend/$REAL_CUSTOMER_ID" -o "$REAL_JSON"
EXPECTED_COLD_START=false JSON_FILE="$REAL_JSON" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path(os.environ["JSON_FILE"]).read_text(encoding="utf-8"))
expected = os.environ["EXPECTED_COLD_START"].lower() == "true"
recs = data.get("recommendations")
if data.get("is_cold_start") is not expected:
    raise SystemExit(f"ERROR: unexpected is_cold_start in response: {data}")
if not isinstance(recs, list) or len(recs) != 12:
    raise SystemExit(f"ERROR: expected exactly 12 recommendations: {data}")
if not all(isinstance(item, str) and len(item) == 10 and item.isdigit() for item in recs):
    raise SystemExit(f"ERROR: invalid article_id format in recommendations: {data}")
print(f"[OK] recommendation: is_cold_start={data['is_cold_start']}, n={len(recs)}")
PY

printf '==> Checking cold-start fallback for unknown customer...\n'
curl -fsS -X POST "$BASE_URL/recommend/$UNKNOWN_CUSTOMER_ID" -o "$COLD_JSON"
EXPECTED_COLD_START=true JSON_FILE="$COLD_JSON" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

data = json.loads(Path(os.environ["JSON_FILE"]).read_text(encoding="utf-8"))
expected = os.environ["EXPECTED_COLD_START"].lower() == "true"
recs = data.get("recommendations")
if data.get("is_cold_start") is not expected:
    raise SystemExit(f"ERROR: unexpected is_cold_start in response: {data}")
if not isinstance(recs, list) or len(recs) != 12:
    raise SystemExit(f"ERROR: expected exactly 12 recommendations: {data}")
if not all(isinstance(item, str) and len(item) == 10 and item.isdigit() for item in recs):
    raise SystemExit(f"ERROR: invalid article_id format in recommendations: {data}")
print(f"[OK] recommendation: is_cold_start={data['is_cold_start']}, n={len(recs)}")
PY

printf '==> Checking validation error for short customer_id...\n'
status_code="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$BASE_URL/recommend/$SHORT_CUSTOMER_ID")"
if [[ "$status_code" != "422" ]]; then
  printf 'ERROR: expected HTTP 422 for short customer_id, got %s\n' "$status_code" >&2
  exit 1
fi

printf '\n[OK] API smoke test completed successfully against %s\n' "$BASE_URL"
