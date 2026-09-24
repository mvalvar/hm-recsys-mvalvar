#!/usr/bin/env bash
# Build the release bundle consumed by the production evaluation bootstrap.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_NAME="${ARTIFACT_NAME:-production_artifacts.zip}"
SMOKE_FILE="smoke_test_customers.txt"
CHECKSUM_FILE="checksums.sha256"

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

required_paths=(
  "data_processed/articles.parquet"
  "data_processed/candidates.parquet"
  "data_processed/customer_id_mapping.parquet"
  "data_processed/customers.parquet"
  "data_processed/features_matrix.parquet"
  "data_processed/transactions_10w.parquet"
  "data_processed/transactions_5w.parquet"
  "data_processed/transactions_full_weekly_agg.parquet"
  "data_processed/v8_basket_affinity.parquet"
  "data_processed/v8_bestsellers_age.parquet"
  "data_processed/v8_customer_history_28d.parquet"
  "models/lgbm_ranker.txt"
  "models/lgbm_ranker_meta.json"
)

printf '==> Verifying local artifacts before packaging...\n'
SKIP_CHECKSUM_VALIDATION=1 bash scripts/verify_artifacts.sh

PYTHON="$(find_python)"
printf '==> Selecting smoke-test customers from indexed feature rows...\n'
"$PYTHON" - <<'PY'
import os
from pathlib import Path

import polars as pl

mapping_path = Path("data_processed/customer_id_mapping.parquet")
features_path = Path("data_processed/features_matrix.parquet")
out_path = Path("smoke_test_customers.txt")
max_rows_env = os.getenv("API_MAX_INDEXED_ROWS", "50000")
max_rows = int(max_rows_env) if max_rows_env.isdigit() and int(max_rows_env) > 0 else 50000

features_df = pl.read_parquet(features_path, columns=["customer_idx"], n_rows=max_rows)
feature_ids = features_df.select("customer_idx").unique(maintain_order=True).head(50)["customer_idx"].to_list()

mapping_df = (
    pl.scan_parquet(mapping_path)
    .filter(pl.col("customer_idx").is_in(feature_ids))
    .select(["customer_idx", "customer_id"])
    .collect()
)
customer_by_idx = dict(zip(mapping_df["customer_idx"].to_list(), mapping_df["customer_id"].to_list()))
ordered_customers = [customer_by_idx[idx] for idx in feature_ids if idx in customer_by_idx][:10]

if len(ordered_customers) < 5:
    raise SystemExit(
        f"ERROR: expected at least 5 indexed smoke-test customers, found {len(ordered_customers)}."
    )

out_path.write_text("\n".join(ordered_customers) + "\n", encoding="utf-8")
print(f"Wrote {len(ordered_customers)} customers to {out_path} from the first {max_rows:,} feature rows")
PY

printf '==> Writing SHA-256 manifest...\n'
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${required_paths[@]}" "$SMOKE_FILE" > "$CHECKSUM_FILE"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${required_paths[@]}" "$SMOKE_FILE" > "$CHECKSUM_FILE"
else
  "$PYTHON" - "${required_paths[@]}" "$SMOKE_FILE" <<'PY'
import hashlib, sys
from pathlib import Path

files = sys.argv[1:]
lines = [f"{hashlib.sha256(Path(f).read_bytes()).hexdigest()}  {f}" for f in files]
Path("checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
fi

if [[ -f "$ARTIFACT_NAME" ]]; then
  rm -f "$ARTIFACT_NAME"
fi

printf '==> Creating %s...\n' "$ARTIFACT_NAME"
if command -v zip >/dev/null 2>&1; then
  zip -q -9 "$ARTIFACT_NAME" "${required_paths[@]}" "$SMOKE_FILE" "$CHECKSUM_FILE"
else
  "$PYTHON" - "$ARTIFACT_NAME" "${required_paths[@]}" "$SMOKE_FILE" "$CHECKSUM_FILE" <<'PY'
import sys, zipfile
from pathlib import Path

zip_out = sys.argv[1]
files = sys.argv[2:]
with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for f in files:
        zf.write(f, arcname=f)
PY
fi

printf '==> Validating checksum manifest...\n'
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c "$CHECKSUM_FILE"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c "$CHECKSUM_FILE"
else
  "$PYTHON" - <<'PY'
import hashlib, sys
from pathlib import Path

ok = True
for line in Path("checksums.sha256").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    exp_h, rel_path = line.split("  ")
    p = Path(rel_path)
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != exp_h:
        print(f"FAILED: {rel_path}")
        ok = False
    else:
        print(f"{rel_path}: OK")
if not ok:
    sys.exit(1)
PY
fi

printf '\n[OK] Release artifact ready: %s\n' "$ARTIFACT_NAME"
du -h "$ARTIFACT_NAME" 2>/dev/null || "$PYTHON" -c "from pathlib import Path; print(f'{Path(\"$ARTIFACT_NAME\").stat().st_size / (1024*1024):.2f} MB')"

