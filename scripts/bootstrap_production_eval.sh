#!/usr/bin/env bash
# Download and validate the GitHub Release artifact bundle for evaluation.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_RELEASE_TAG="${DEFAULT_RELEASE_TAG:-v1.0-production-artifacts}"
DEFAULT_ASSET_NAME="${DEFAULT_ASSET_NAME:-production_artifacts.zip}"
DEFAULT_ARTIFACTS_URL="${DEFAULT_ARTIFACTS_URL:-https://github.com/mvalvar/hm-recsys-mvalvar/releases/download/v1.0-production-artifacts/production_artifacts.zip}"
ARTIFACTS_URL="${ARTIFACTS_URL:-$DEFAULT_ARTIFACTS_URL}"
REPO_SLUG="${REPO_SLUG:-${GITHUB_REPOSITORY:-}}"

find_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
  elif [[ -x ".venv/bin/python" ]]; then
    printf '%s\n' ".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    printf 'ERROR: no Python interpreter found. Set PYTHON_BIN=/path/to/python.\n' >&2
    return 1
  fi
}

derive_repo_slug_from_git() {
  if ! command -v git >/dev/null 2>&1; then
    return 1
  fi
  local remote
  remote="$(git config --get remote.origin.url 2>/dev/null || true)"
  [[ -n "$remote" ]] || return 1
  case "$remote" in
    https://github.com/*)
      remote="${remote#https://github.com/}"
      ;;
    git@github.com:*)
      remote="${remote#git@github.com:}"
      ;;
    *)
      return 1
      ;;
  esac
  remote="${remote%.git}"
  [[ "$remote" == */* ]] || return 1
  printf '%s\n' "$remote"
}

if [[ -z "$REPO_SLUG" ]]; then
  REPO_SLUG="$(derive_repo_slug_from_git || true)"
fi
if [[ -z "$REPO_SLUG" ]]; then
  REPO_SLUG="mvalvar/hm-recsys-mvalvar"
fi

if [[ -z "$ARTIFACTS_URL" ]]; then
  ARTIFACTS_URL="https://github.com/${REPO_SLUG}/releases/download/${DEFAULT_RELEASE_TAG}/${DEFAULT_ASSET_NAME}"
fi

# Allow local artifact file to bypass download entirely (offline or manual testing)
LOCAL_ARTIFACTS="${LOCAL_ARTIFACTS:-${1:-}}"
if [[ -z "$LOCAL_ARTIFACTS" && -f "$ROOT_DIR/production_artifacts.zip" && "${USE_LOCAL_ARTIFACTS:-0}" == "1" ]]; then
  LOCAL_ARTIFACTS="$ROOT_DIR/production_artifacts.zip"
fi

CLEANUP_ZIP=0
download_ok=0

if [[ -n "$LOCAL_ARTIFACTS" && -f "$LOCAL_ARTIFACTS" ]]; then
  printf '==> Using local artifacts bundle: %s\n' "$LOCAL_ARTIFACTS"
  TMP_ZIP="$LOCAL_ARTIFACTS"
  download_ok=1
else
  if ! command -v curl >/dev/null 2>&1; then
    printf 'ERROR: curl is required to download %s\n' "$ARTIFACTS_URL" >&2
    exit 1
  fi

  curl_args=(-fL --retry 3 --retry-delay 2)
  if [[ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" ]]; then
    curl_args+=(-H "Authorization: Bearer ${GITHUB_TOKEN:-${GH_TOKEN:-}}")
    curl_args+=(-H "Accept: application/octet-stream")
  fi

  TMP_ZIP="$(mktemp "/tmp/${DEFAULT_ASSET_NAME}.XXXXXX")"
  CLEANUP_ZIP=1
  trap '[[ "$CLEANUP_ZIP" == "1" ]] && rm -f "$TMP_ZIP"' EXIT

  printf '==> Downloading production artifacts...\n'
  printf '    %s\n' "$ARTIFACTS_URL"
  if curl "${curl_args[@]}" -o "$TMP_ZIP" "$ARTIFACTS_URL"; then
    download_ok=1
  elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    printf '==> Direct URL download returned non-200; attempting via GitHub CLI (gh release download)...\n'
    if gh release download "$DEFAULT_RELEASE_TAG" --repo "$REPO_SLUG" --pattern "$DEFAULT_ASSET_NAME" --output "$TMP_ZIP" --clobber; then
      download_ok=1
    fi
  elif [[ -n "${GITHUB_TOKEN:-${GH_TOKEN:-}}" && -n "$REPO_SLUG" ]]; then
    printf '==> Direct release URL failed; trying GitHub Releases API asset download...\n'
    PYTHON="$(find_python || true)"
    if [[ -n "$PYTHON" ]]; then
      ASSET_ID="$(
        curl -fsS \
          -H "Authorization: Bearer ${GITHUB_TOKEN:-${GH_TOKEN:-}}" \
          -H "Accept: application/vnd.github+json" \
          "https://api.github.com/repos/${REPO_SLUG}/releases/tags/${DEFAULT_RELEASE_TAG}" \
          | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print(next((a["id"] for a in d.get("assets", []) if a.get("name") == "production_artifacts.zip"), ""))'
      )"
      if [[ -n "$ASSET_ID" ]] && curl -fL \
        -H "Authorization: Bearer ${GITHUB_TOKEN:-${GH_TOKEN:-}}" \
        -H "Accept: application/octet-stream" \
        -o "$TMP_ZIP" \
        "https://api.github.com/repos/${REPO_SLUG}/releases/assets/${ASSET_ID}"; then
        download_ok=1
      fi
    fi
  fi
fi

if [[ "$download_ok" != "1" ]]; then
  cat >&2 <<EOF
ERROR: could not download the production artifacts.

- If the repository is PUBLIC: the direct curl download will work automatically.
- If the repository is currently PRIVATE:
    gh auth login
    bash scripts/bootstrap_production_eval.sh
  OR provide an access token:
    GITHUB_TOKEN=<token> bash scripts/bootstrap_production_eval.sh
  OR use a local zip if already downloaded:
    USE_LOCAL_ARTIFACTS=1 bash scripts/bootstrap_production_eval.sh
EOF
  exit 1
fi

printf '==> Extracting artifacts into %s...\n' "$ROOT_DIR"
PYTHON="$(find_python)"
"$PYTHON" - "$TMP_ZIP" "$ROOT_DIR" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()

with zipfile.ZipFile(zip_path) as zf:
    for member in zf.infolist():
        target = (root / member.filename).resolve()
        if root != target and root not in target.parents:
            raise SystemExit(f"ERROR: unsafe zip path: {member.filename}")
    zf.extractall(root)
PY

printf '==> Validating extracted structure and checksums...\n'
bash scripts/verify_artifacts.sh

if [[ ! -f checksums.sha256 ]]; then
  printf 'ERROR: checksums.sha256 was not found after extraction.\n' >&2
  exit 1
fi
sha256sum -c checksums.sha256

printf '\n[OK] Production evaluation artifacts are ready.\n'
printf 'Next: docker compose up --build -d && bash scripts/smoke_test_api.sh\n'
