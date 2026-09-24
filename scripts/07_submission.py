"""Script 07: Inferencia Masiva y Generación del Archivo de Submission para Kaggle.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Directivas de ejecución:
1. Cobertura exhaustiva: Genera recomendaciones para exactamente 1.371.980 clientes únicos
   según las especificaciones del Kaggle Submission Format.
2. Arquitectura de Producción (V8 SOTA): DuckDB Multidimensional Waterfall con ventana óptima de 28 días,
   afinidad global en cesta P(B|A) y suavizado bayesiano multi-semana (MAP@12 Kaggle Private: 0.02386, Public: 0.02347).
3. Soporte Multiversión (V1 a V8): Reproduce pipelines históricos y experimentos certificados (V7: 0.02332, V5: 0.02212).
4. Certificación Criptográfica: Cálculo automático de hashes SHA-256 y MD5 en streaming e
   integración con results/SUBMISSIONS_MANIFEST.json para verificación 100% bit-for-bit.
5. Formato Kaggle: 2 columnas ('customer_id', 'prediction'), exactamente 12 artículos de 10 dígitos
   con ceros a la izquierda (str.zfill(10)) concatenados con espacio simple (longitud exacta: 131 chars).
6. Out-of-Core & Memory Safety: Operaciones vectorizadas en Polars/DuckDB con memoria controlada.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.submission import (
    build_submission,
    compute_file_hashes,
    load_all_customer_ids,
    register_submission_manifest,
    run_submission_pipeline,
    run_v5_waterfall_pipeline,
    run_v6_hybrid_waterfall_pipeline,
    run_v7_non_destructive_waterfall_pipeline,
    run_v8_affinity_waterfall_pipeline,
    run_version_pipeline,
    score_candidates,
    validate_submission,
    verify_submission_reproducibility,
    write_submission,
)

__all__ = [
    "score_candidates",
    "load_all_customer_ids",
    "build_submission",
    "validate_submission",
    "write_submission",
    "compute_file_hashes",
    "verify_submission_reproducibility",
    "register_submission_manifest",
    "run_submission_pipeline",
    "run_v5_waterfall_pipeline",
    "run_v6_hybrid_waterfall_pipeline",
    "run_v7_non_destructive_waterfall_pipeline",
    "run_v8_affinity_waterfall_pipeline",
    "run_version_pipeline",
    "main",
]


def main() -> None:
    """CLI para ejecución configurable de la inferencia masiva y certificación criptográfica."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Generador masivo de submission.csv y Certificador Criptográfico para Kaggle H&M RecSys"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v8",
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"],
        help="Versión del pipeline de recomendación a ejecutar (v1 a v8). Por defecto 'v8' (afinidad global ponderada y suavizado multi-semana)",
    )
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        help="Verifica la identidad bit-for-bit del archivo contra los hashes SHA-256 y MD5 en SUBMISSIONS_MANIFEST.json",
    )
    parser.add_argument(
        "--check-file",
        type=str,
        default=None,
        help="Audita criptográficamente un archivo existente contra el manifiesto sin re-ejecutar inferencia",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Ejecuta únicamente sobre los clientes evaluados en la muestra activa (desarrollo rápido con modelo)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Genera la submission completa para los 1.371.980 clientes oficiales de Kaggle (por defecto)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de destino del archivo submission.csv (por defecto raíz del proyecto)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Alias para ejecutar pipeline supervisado LGBMRanker sobre features_matrix (equivalente a --version v4)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Desactiva la generación del archivo comprimido submission.csv.gz",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Desactiva la validación de formato de la submission",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Registra o actualiza la submission en results/SUBMISSIONS_MANIFEST.json",
    )
    args = parser.parse_args()

    # Auditoría directa de un archivo existente
    if args.check_file:
        check_path = Path(args.check_file)
        if not check_path.is_absolute():
            check_path = BASE_DIR / check_path
        target_version = args.version or "v8"
        verify_submission_reproducibility(file_path=check_path, version=target_version)
        return

    # Selección de versión
    selected_version = "v4" if args.legacy else args.version.lower()
    sample_mode = args.sample and not args.full

    # Resolución de ruta de salida
    if args.output:
        target_output = Path(args.output)
        if not target_output.is_absolute():
            target_output = BASE_DIR / target_output
    else:
        if selected_version == "v5":
            target_output = BASE_DIR / "submission_v5.csv"
        else:
            target_output = BASE_DIR / f"submission_{selected_version}.csv"

    # Ejecución del pipeline multiversión
    result = run_version_pipeline(
        version=selected_version,
        output_path=target_output,
        compress=not args.no_compress,
        verify=not args.no_verify,
        sample_mode=sample_mode,
        verify_reproducibility=args.verify_reproducibility,
    )

    # Registro opcional en manifiesto
    if args.register:
        register_submission_manifest(
            version=selected_version,
            file_path=target_output,
            gz_path=result.get("gz_path"),
            architecture=f"Pipeline {selected_version.upper()}",
            diagnostic="Generación manual desde CLI con --register",
        )


if __name__ == "__main__":
    main()
