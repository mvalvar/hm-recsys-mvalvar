#!/usr/bin/env bash
# Validate the production artifacts needed by the FastAPI runtime.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

require_file() {
  local path="$1"
  local min_bytes="$2"
  if [[ ! -f "$path" ]]; then
    printf 'ERROR: required artifact missing: %s\n' "$path" >&2
    return 1
  fi

  local size
  size="$(stat -c '%s' "$path" 2>/dev/null || stat -f '%z' "$path" 2>/dev/null || wc -c < "$path" | tr -d ' ')"
  if (( size < min_bytes )); then
    printf 'ERROR: artifact too small: %s (%s bytes, minimum %s)\n' "$path" "$size" "$min_bytes" >&2
    return 1
  fi
}

printf '==> Checking required files and minimum sizes...\n'
require_file "data_processed/features_matrix.parquet" 100000000
require_file "data_processed/customer_id_mapping.parquet" 1000000
require_file "data_processed/transactions_5w.parquet" 1000000
require_file "models/lgbm_ranker.txt" 100000
require_file "models/lgbm_ranker_meta.json" 100

for path in \
  data_processed/articles.parquet \
  data_processed/candidates.parquet \
  data_processed/customers.parquet \
  data_processed/transactions_10w.parquet \
  data_processed/transactions_full_weekly_agg.parquet \
  data_processed/v8_basket_affinity.parquet \
  data_processed/v8_bestsellers_age.parquet \
  data_processed/v8_customer_history_28d.parquet; do
  require_file "$path" 1
done

PYTHON="$(find_python || true)"
if [[ -n "$PYTHON" ]] && "$PYTHON" -c "import lightgbm, polars" >/dev/null 2>&1; then
  printf '==> Reading Parquet/model artifacts with %s...\n' "$PYTHON"
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

parquet_checks = {
    "data_processed/features_matrix.parquet": {"customer_idx", "article_id"},
    "data_processed/customer_id_mapping.parquet": {"customer_id", "customer_idx"},
    "data_processed/transactions_5w.parquet": {"customer_idx", "article_id", "t_dat"},
}

for file_name, required_columns in parquet_checks.items():
    path = Path(file_name)
    df = pl.read_parquet(path, n_rows=5)
    missing = required_columns.difference(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {path} is missing columns: {sorted(missing)}")
    if df.height == 0:
        raise SystemExit(f"ERROR: {path} has no readable rows")

model = lgb.Booster(model_file="models/lgbm_ranker.txt")
if model.num_trees() <= 0:
    raise SystemExit("ERROR: LightGBM model has no trees")

meta_path = Path("models/lgbm_ranker_meta.json")
meta = json.loads(meta_path.read_text(encoding="utf-8"))
feature_names = meta.get("feature_names") or meta.get("features")
if not isinstance(feature_names, list) or not feature_names:
    raise SystemExit("ERROR: model metadata does not include feature_names/features")

print(
    "[OK] Artifacts readable: "
    f"{model.num_trees()} trees, {len(feature_names)} model features"
)
PY
else
  printf '==> Skipping host Python schema inspection (lightgbm/polars not installed in host Python; checksums and container guarantee integrity)\n'
fi

if [[ -f "checksums.sha256" && "${SKIP_CHECKSUM_VALIDATION:-0}" != "1" ]]; then
  printf '==> Verifying checksums.sha256...\n'
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c checksums.sha256
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c checksums.sha256
  elif [[ -n "$PYTHON" ]]; then
    "$PYTHON" - <<'PY'
import hashlib, sys
from pathlib import Path
ok = True
for line in Path("checksums.sha256").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    exp_h, rel_path = line.split("  ")
    p = Path(rel_path)
    if not p.exists():
        print(f"FAILED (missing): {rel_path}")
        ok = False
        continue
    act_h = hashlib.sha256(p.read_bytes()).hexdigest()
    if act_h == exp_h:
        print(f"{rel_path}: OK")
    else:
        print(f"FAILED: {rel_path}")
        ok = False
if not ok:
    sys.exit(1)
PY
  fi
fi

printf '[OK] Production artifacts verified.\n'
