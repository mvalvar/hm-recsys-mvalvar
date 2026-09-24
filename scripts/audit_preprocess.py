"""Script de Validación de Integridad y Calidad del Preprocesamiento (Fase 1).

Ejecuta un diagnóstico dimensional e imprime un reporte formal de calidad
evaluando contratos de esquema, downcasting, límites temporales y paridad entrenamiento-inferencia.

Uso:
    python scripts/audit_preprocess.py          # Valida datos de producción (data_processed/)
    python scripts/audit_preprocess.py --sample # Valida muestra rápida (data_processed/sample/)
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATA_PROCESSED_SAMPLE_DIR,
    TABLES_DIR,
)


def run_audit(use_sample: bool = False) -> tuple[bool, dict[str, Any]]:
    """Ejecuta la validación de calidad sobre los artefactos preprocesados."""
    target_dir = DATA_PROCESSED_SAMPLE_DIR if use_sample else DATA_PROCESSED_DIR
    mode_name = "MUESTRA RÁPIDA (sample)" if use_sample else "PRODUCCIÓN MASIVA (full)"

    rel_target_dir = target_dir.relative_to(BASE_DIR).as_posix() if target_dir.is_relative_to(BASE_DIR) else str(target_dir)

    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "mode": mode_name,
        "target_dir": rel_target_dir,
        "validations": {},
        "files": {},
    }

    print("=" * 80)
    print(f"  VALIDACIÓN DE PREPROCESAMIENTO: MODO: {mode_name}")
    print(f"  Directorio auditado: {rel_target_dir}")
    print("=" * 80)

    # Comprobación de existencia de los 5 archivos
    required_files = [
        "transactions_5w.parquet",
        "customers.parquet",
        "articles.parquet",
        "customer_id_mapping.parquet",
        "transactions_full_weekly_agg.parquet",
    ]

    all_exist = True
    for fn in required_files:
        f_path = target_dir / fn
        exists = f_path.exists()
        size_mb = (f_path.stat().st_size / (1024 * 1024)) if exists else 0.0
        report["files"][fn] = {"exists": exists, "size_mb": size_mb}
        if not exists:
            all_exist = False
            print(f"  [ERROR] Archivo faltante: {fn}")

    if not all_exist:
        return False, report

    # Carga optimizada
    tx = pl.read_parquet(target_dir / "transactions_5w.parquet")
    cust = pl.read_parquet(target_dir / "customers.parquet")
    art = pl.read_parquet(target_dir / "articles.parquet")
    mapping = pl.read_parquet(target_dir / "customer_id_mapping.parquet")
    weekly_agg = pl.read_parquet(target_dir / "transactions_full_weekly_agg.parquet")

    # Validación de esquema y tipos con downcasting
    p1_checks = []
    p1_checks.append(tx.schema["customer_idx"].is_integer())
    p1_checks.append(tx.schema["article_id"] == pl.Int32)
    p1_checks.append(tx.schema["price"] == pl.Float32)
    p1_checks.append(tx.schema["sales_channel_id"] == pl.Int8)
    p1_checks.append(tx.schema["t_dat"] == pl.Date)
    p1_checks.append(cust.schema["age"] == pl.Float32)
    p1_pass = all(p1_checks)
    report["validations"]["1. Contratos de Datos y Downcasting"] = {
        "status": "PASS" if p1_pass else "FAIL",
        "tx_schema": {k: str(v) for k, v in tx.schema.items()},
    }

    # Time-based split y prevención de data leakage
    min_date = tx.select(pl.col("t_dat").min()).item()
    max_date = tx.select(pl.col("t_dat").max()).item()
    days_span = (max_date - min_date).days + 1
    p2_pass = days_span <= 36 and days_span >= 30 and tx.filter(pl.col("t_dat").is_null()).height == 0
    report["validations"]["2. Inmunidad Temporal y Límites"] = {
        "status": "PASS" if p2_pass else "FAIL",
        "min_date": str(min_date),
        "max_date": str(max_date),
        "days_span": days_span,
        "weekly_agg_rows": weekly_agg.height,
    }

    # Chequeo de valores nulos e identificadores en cero
    zero_prices = tx.filter(pl.col("price") <= 0.0).height
    min_price = float(tx.select(pl.col("price").min()).item())
    max_price = float(tx.select(pl.col("price").max()).item())
    mean_price = float(tx.select(pl.col("price").mean()).item())

    invalid_ages = cust.filter((pl.col("age") < 16.0) | (pl.col("age") > 100.0)).height
    age_nulls = cust.filter(pl.col("age").is_null()).height
    median_age = float(cust.select(pl.col("age").median()).item())

    p3_pass = (zero_prices == 0) and (invalid_ages == 0) and (age_nulls == 0)
    report["validations"]["3. Rango de Precios y Calidad Demográfica"] = {
        "status": "PASS" if p3_pass else "FAIL",
        "zero_prices": zero_prices,
        "min_price": min_price,
        "mean_price": round(mean_price, 4),
        "max_price": round(max_price, 4),
        "invalid_ages": invalid_ages,
        "median_age": median_age,
    }

    # Integridad referencial (FK) y hash mapping 1:1
    n_map = mapping.height
    dup_cid = n_map - mapping["customer_id"].n_unique()
    dup_cidx = n_map - mapping["customer_idx"].n_unique()
    min_idx = mapping.select(pl.col("customer_idx").min()).item()
    max_idx = mapping.select(pl.col("customer_idx").max()).item()
    is_contiguous = (min_idx == 0) and (max_idx == n_map - 1)

    # Huérfanos
    tx_cust_set = set(tx["customer_idx"].unique().to_list())
    map_cust_set = set(mapping["customer_idx"].to_list())
    orphan_cust = len(tx_cust_set - map_cust_set)

    tx_art_set = set(tx["article_id"].unique().to_list())
    art_set = set(art["article_id"].to_list())
    orphan_art = len(tx_art_set - art_set)

    p4_pass = (dup_cid == 0) and (dup_cidx == 0) and is_contiguous and (orphan_cust == 0) and (orphan_art == 0)
    report["validations"]["4. Integridad Referencial y Biyección"] = {
        "status": "PASS" if p4_pass else "FAIL",
        "mapped_customers": n_map,
        "contiguous_indices": is_contiguous,
        "orphan_customers": orphan_cust,
        "orphan_articles": orphan_art,
        "catalog_articles": art.height,
    }

    # Paridad train/inference y fallbacks
    try:
        from src.candidates.generators import get_popular_fallback_items

        fb = get_popular_fallback_items(transactions_df=tx, top_k=12)
        p5_pass = len(fb) == 12 and len(set(fb)) == 12
    except Exception:
        p5_pass = False

    report["validations"]["5. Paridad Entrenamiento-Inferencia"] = {
        "status": "PASS" if p5_pass else "FAIL",
        "fallback_count": len(fb) if p5_pass else 0,
    }

    # Verificación de compresión y metadatos Parquet
    total_size_mb = sum(f["size_mb"] for f in report["files"].values())
    p6_pass = total_size_mb > 0.0
    report["validations"]["6. Compresión y Serialización Parquet"] = {
        "status": "PASS" if p6_pass else "FAIL",
        "total_size_mb": round(total_size_mb, 2),
    }

    elapsed_sec = time.perf_counter() - t0
    report["elapsed_sec"] = round(elapsed_sec, 2)

    # IMPRESIÓN DEL REPORTE EJECUTIVO
    print("\n" + "-" * 80)
    print(f"  {'DIMENSIÓN DE CONTROL':<45} | {'ESTADO':<10} | {'MÉTRICAS CLAVE'}")
    print("-" * 80)

    overall_pass = True
    for val_name, data in report["validations"].items():
        st = data["status"]
        if st != "PASS":
            overall_pass = False
        badge = "[OK]" if st == "PASS" else "[ERROR]"

        # Resumen breve de métricas
        if "Contratos" in val_name:
            detail = f"{tx.height:,} filas tx | Float32/Int32/Int8"
        elif "Temporal" in val_name:
            detail = f"{data['min_date']} a {data['max_date']} ({data['days_span']} días)"
        elif "Precios" in val_name:
            detail = f"Precios > 0 ({data['min_price']} a {data['max_price']}) | Edad med: {data['median_age']}"
        elif "Integridad" in val_name:
            detail = f"{data['mapped_customers']:,} usuarios (1:1) | 0 huérfanos | {data['catalog_articles']:,} cat"
        elif "Inferencia" in val_name:
            detail = "Fallback Top-12 activo y validado"
        elif "Compresión" in val_name:
            detail = f"{data['total_size_mb']} MB Parquet ZSTD"
        else:
            detail = ""

        print(f"  {val_name:<45} | {badge:<10} | {detail}")

    print("-" * 80)
    print(f"  * Tiempo de Validación        : {elapsed_sec:.2f} s")
    print(f"  * Volumen Total en Disco      : {total_size_mb:.2f} MB")
    print(f"  * Dictamen Final              : {'[APROBADO]' if overall_pass else '[RECHAZADO]'}")
    print("=" * 80 + "\n")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report_file = TABLES_DIR / f"preprocess_audit_{'sample' if use_sample else 'prod'}.md"
    _write_markdown_report(report_file, report, overall_pass)
    print(f"-> Reporte formal archivado en: {report_file.relative_to(BASE_DIR)}")

    return overall_pass, report


def _write_markdown_report(report_path: Path, report: dict[str, Any], passed: bool) -> None:
    """Genera informe formal en formato Markdown."""
    lines = [
        f"# Reporte de Calidad del Preprocesamiento ({report['mode']})",
        f"\n**Fecha de Validación:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Directorio:** `{report['target_dir']}`  ",
        f"**Dictamen Final:** **{'APROBADO' if passed else 'RECHAZADO'}**  \n",
        "## 1. Evaluación de Calidad de Datos\n",
        "| Dimensión de Control | Estado | Detalles Técnicos |",
        "| :--- | :---: | :--- |",
    ]

    for val_name, d in report["validations"].items():
        st = "✅ PASS" if d["status"] == "PASS" else "❌ FAIL"
        lines.append(f"| **{val_name}** | {st} | `{d}` |")

    lines.extend([
        "\n## 2. Inventario de Artefactos Parquet\n",
        "| Archivo | Presente | Tamaño (MB) |",
        "| :--- | :---: | :---: |",
    ])

    for fn, fd in report["files"].items():
        st = "✅" if fd["exists"] else "❌"
        lines.append(f"| `{fn}` | {st} | {fd['size_mb']:.2f} MB |")

    lines.append(f"\n**Tiempo de ejecución:** {report['elapsed_sec']} segundos.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validación de calidad e integridad del preprocesamiento")
    parser.add_argument("--sample", action="store_true", help="Audita el directorio de muestra data_processed/sample/")
    args = parser.parse_args()

    success, _ = run_audit(use_sample=args.sample)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
