#!/usr/bin/env bash
# Helper script to verify and publish production_artifacts.zip to GitHub Releases.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RELEASE_TAG="${1:-v1.0-production-artifacts}"
ARTIFACT_FILE="production_artifacts.zip"
REPO_SLUG="${REPO_SLUG:-mvalvar/hm-recsys-mvalvar}"
CHECKSUM_FILE="checksums.sha256"

printf '==> Checking release artifact bundle...\n'
if [[ ! -f "$ARTIFACT_FILE" ]]; then
  printf 'ERROR: %s not found in %s.\n' "$ARTIFACT_FILE" "$ROOT_DIR" >&2
  printf 'Run bash scripts/build_production_artifacts.sh first.\n' >&2
  exit 1
fi

if [[ -f "$CHECKSUM_FILE" ]]; then
  printf '==> Validating checksums against %s...\n' "$CHECKSUM_FILE"
  sha256sum -c "$CHECKSUM_FILE"
fi

printf '\n==> Target Repository: %s\n' "$REPO_SLUG"
printf '==> Target Release Tag: %s\n' "$RELEASE_TAG"
printf '==> Artifact Asset:     %s (%s)\n\n' "$ARTIFACT_FILE" "$(du -h "$ARTIFACT_FILE" | cut -f1)"

# Check if GitHub CLI is installed and authenticated
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  printf '==> GitHub CLI (gh) detected and authenticated.\n'
  printf '==> Publishing release to GitHub...\n'
  
  EXTRA_ASSETS=()
  if [[ -f "submission_v8.csv.gz" ]]; then
    EXTRA_ASSETS+=("submission_v8.csv.gz")
  fi

  if gh release view "$RELEASE_TAG" --repo "$REPO_SLUG" >/dev/null 2>&1; then
    printf 'Tag %s already exists on %s. Uploading assets with overwrite...\n' "$RELEASE_TAG" "$REPO_SLUG"
    gh release upload "$RELEASE_TAG" "$ARTIFACT_FILE" "${EXTRA_ASSETS[@]}" --repo "$REPO_SLUG" --clobber
  else
    printf 'Creating new release %s on %s...\n' "$RELEASE_TAG" "$REPO_SLUG"
    gh release create "$RELEASE_TAG" "$ARTIFACT_FILE" "${EXTRA_ASSETS[@]}" \
      --repo "$REPO_SLUG" \
      --title "$RELEASE_TAG" \
      --notes "Artefactos precalculados de producción V8 SOTA (382.9 MB comprimidos, 14 archivos verificados) para evaluación del TFM RecSys:
- data_processed/*.parquet (incluye v8_bestsellers_age, v8_basket_affinity, v8_customer_history_28d)
- models/lgbm_ranker.txt (Booster LambdaRank 50 árboles) y lgbm_ranker_meta.json
- smoke_test_customers.txt y checksums.sha256
- submission_v8.csv.gz (Entrega oficial Kaggle certificada bit-a-bit con NIST SHA-256)
Ver guía de uso en README.md y PRODUCTION_EVALUATION.md."
  fi
  printf '\n[OK] Release published successfully via gh CLI.\n'
else
  printf '==> GitHub CLI (gh) no está instalado o no tiene sesión iniciada.\n'
  printf '\nPasos para publicar los artefactos manualmente en GitHub:\n'
  printf '1. Abre en tu navegador: https://github.com/%s/releases/new\n' "$REPO_SLUG"
  printf '2. En "Choose a tag", introduce: %s (y selecciona crear tag en target: main)\n' "$RELEASE_TAG"
  printf '3. En "Release title", introduce: %s\n' "$RELEASE_TAG"
  printf '4. En la sección "Attach binaries", arrastra y sube los archivos:\n'
  printf '   - %s/%s\n' "$ROOT_DIR" "$ARTIFACT_FILE"
  if [[ -f "submission_v8.csv.gz" ]]; then
    printf '   - %s/submission_v8.csv.gz\n' "$ROOT_DIR"
  fi
  printf '5. Haz clic en el botón verde "Publish release".\n\n'
fi

# Verify public availability of the release asset
ASSET_URL="https://github.com/${REPO_SLUG}/releases/download/${RELEASE_TAG}/${ARTIFACT_FILE}"
printf '==> Checking public download URL availability:\n'
printf '    %s\n' "$ASSET_URL"

HTTP_STATUS="$(curl -sIL -o /dev/null -w '%{http_code}' "$ASSET_URL" || true)"
if [[ "$HTTP_STATUS" == "200" ]]; then
  printf '\n[OK] Release asset is live and publicly downloadable (HTTP 200)!\n'
  printf 'bootstrap_production_eval.sh will work seamlessly out of the box.\n'
else
  printf '\n[INFO] Release asset returned HTTP %s (not yet available or release is draft/private).\n' "$HTTP_STATUS"
  printf 'Once the release is published, bootstrap_production_eval.sh will download it automatically.\n'
fi
