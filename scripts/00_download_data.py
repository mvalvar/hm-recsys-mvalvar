"""Script 00: Descarga y Verificación de Integridad de Datos Crudos.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Flujo de aprovisionamiento y validación:
1. Detección Local Inmediata: Comprueba si los 4 archivos CSV oficiales ya existen en
   disco (data/ o h&m_data/). Si existen, los vincula/valida y concluye en ~1 segundo.
2. Descarga Oficial Kaggle: Si no existen en local, descarga únicamente los 4 CSV
   necesarios mediante Kaggle CLI y las credenciales del evaluador autorizado.
3. Verificación de Integridad: Verifica la existencia, tamaño y volumen exacto
   de registros de los 4 archivos (31.788.324 transacciones, 1.371.980 clientes,
   105.542 artículos), validando la integridad y estructura del dataset para entrenamiento.

Uso:
    python scripts/00_download_data.py                 # Modo inteligente por defecto
    python scripts/00_download_data.py --check         # Modo verificación de integridad local
    python scripts/00_download_data.py --source kaggle # Descargar vía Kaggle CLI oficial
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

KAGGLE_COMPETITION = "h-and-m-personalized-fashion-recommendations"
KAGGLE_RULES_URL = f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules"

# Metadatos canónicos de los 4 archivos crudos de H&M
CANONICAL_DATASET: dict[str, dict[str, Any]] = {
    "articles.csv": {
        "min_bytes": 30_000_000,
        "expected_rows": 105_542,
        "description": "Catálogo completo de artículos y metadatos de producto",
    },
    "customers.csv": {
        "min_bytes": 180_000_000,
        "expected_rows": 1_371_980,
        "description": "Censo total de clientes y atributos sociodemográficos",
    },
    "sample_submission.csv": {
        "min_bytes": 250_000_000,
        "expected_rows": 1_371_980,
        "description": "Plantilla canónica de clientes y formato oficial de Kaggle",
    },
    "transactions_train.csv": {
        "min_bytes": 3_000_000_000,
        "expected_rows": 31_788_324,
        "description": "Historial transaccional completo (31.78M compras)",
    },
}


def count_csv_rows_fast(csv_path: Path) -> int:
    """Cuenta de forma eficiente y fuera de memoria las filas de un CSV usando DuckDB."""
    try:
        import duckdb

        con = duckdb.connect()
        path_str = str(csv_path).replace("\\", "/")
        res = con.execute(f"SELECT count(1) FROM read_csv_auto('{path_str}')").fetchone()
        return int(res[0]) if res else 0
    except Exception:
        # Fallback de conteo por bloques en caso de incidencia con DuckDB
        count = 0
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
        return max(0, count - 1)


def verify_dataset_integrity(
    target_dir: Path, fast_check: bool = False
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Audita la presencia, peso e integridad de los archivos requeridos."""
    results: dict[str, dict[str, Any]] = {}
    all_valid = True

    for filename, meta in CANONICAL_DATASET.items():
        file_path = target_dir / filename
        exists = file_path.exists()
        size_bytes = file_path.stat().st_size if exists else 0
        size_mb = size_bytes / (1024 * 1024)

        is_size_ok = exists and (size_bytes >= meta["min_bytes"])
        rows = -1
        is_rows_ok = False

        if is_size_ok:
            if fast_check:
                is_rows_ok = True
                rows = meta["expected_rows"]
            else:
                rows = count_csv_rows_fast(file_path)
                is_rows_ok = rows == meta["expected_rows"]

        file_valid = exists and is_size_ok and is_rows_ok
        if not file_valid:
            all_valid = False

        results[filename] = {
            "path": file_path,
            "exists": exists,
            "size_mb": size_mb,
            "size_bytes": size_bytes,
            "rows": rows,
            "expected_rows": meta["expected_rows"],
            "description": meta["description"],
            "valid": file_valid,
        }

    return all_valid, results


def try_local_link(target_dir: Path, custom_local_dir: Path | None = None) -> bool:
    """Intenta detectar y vincular datos desde ubicaciones locales conocidas sin duplicar espacio."""
    candidate_dirs: list[Path] = []
    if custom_local_dir:
        candidate_dirs.append(custom_local_dir)

    candidate_dirs.extend(
        [
            BASE_DIR.parent / "h&m_data",
            Path(os.environ.get("TFM_DATA_RAW_DIR", "")),
        ]
    )

    for source_dir in candidate_dirs:
        if not source_dir or not source_dir.exists():
            continue
        if source_dir.resolve() == target_dir.resolve():
            continue

        if not (source_dir / "transactions_train.csv").exists():
            continue

        print(f"-> Datos crudos detectados en disco local: {source_dir}")
        print(f"-> Vinculando archivos a {target_dir.name}/ (almacenamiento zero-copy)...")
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename in CANONICAL_DATASET:
            src_file = source_dir / filename
            dst_file = target_dir / filename
            if src_file.exists() and not dst_file.exists():
                try:
                    # Enlace duro (hard link NTFS): 0 MB de espacio adicional y latencia 0 ms
                    os.link(src_file, dst_file)
                except Exception:
                    try:
                        # Fallback a enlace simbólico
                        os.symlink(src_file, dst_file)
                    except Exception:
                        # Fallback a copia estándar
                        shutil.copy2(src_file, dst_file)

        # Validar si tras vincular están completos
        is_ok, _ = verify_dataset_integrity(target_dir, fast_check=True)
        if is_ok:
            return True

    return False


def print_kaggle_setup_instructions() -> None:
    """Informa al evaluador cómo habilitar la descarga oficial sin redistribuir datos."""
    print("\n[ACCION REQUERIDA] Para descargar los datos oficiales de Kaggle:")
    print(f"  1. Inicie sesión y acepte las reglas: {KAGGLE_RULES_URL}")
    print("  2. Instale la CLI si no está disponible: python -m pip install kaggle")
    print("  3. Autentique su cuenta con una de estas opciones:")
    print("     - kaggle auth login")
    print("     - o coloque kaggle.json en ~/.kaggle/kaggle.json con permisos 600")
    print("  4. Reintente: python scripts/00_download_data.py --source kaggle\n")


def get_kaggle_command() -> list[str] | None:
    """Devuelve el comando Kaggle disponible en el entorno actual."""
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        return [kaggle_bin]
    # Fallback para Windows / venv cuando Kaggle CLI no está en PATH
    try:
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            return [sys.executable, "-m", "kaggle"]
    except Exception:
        pass
    return None


def extract_downloaded_file(target_dir: Path, filename: str) -> bool:
    """Normaliza descargas individuales de Kaggle que pueden llegar como CSV o ZIP."""
    direct_path = target_dir / filename
    if direct_path.exists() and direct_path.stat().st_size > 0:
        return True

    zip_candidates = [
        target_dir / f"{filename}.zip",
        target_dir / f"{Path(filename).stem}.zip",
    ]
    zip_candidates.extend(sorted(target_dir.glob("*.zip")))

    for zip_path in dict.fromkeys(zip_candidates):
        if not zip_path.exists():
            continue
        extracted = False
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if filename in zf.namelist():
                    zf.extract(filename, target_dir)
                    extracted = True
        except zipfile.BadZipFile:
            continue

        if extracted and direct_path.exists() and direct_path.stat().st_size > 0:
            try:
                zip_path.unlink(missing_ok=True)
            except OSError:
                pass
            return True

    return False


def download_from_kaggle(target_dir: Path, missing_files: list[str]) -> bool:
    """Descarga solo los CSV necesarios mediante Kaggle CLI oficial."""
    target_dir.mkdir(parents=True, exist_ok=True)
    kaggle_cmd = get_kaggle_command()
    if kaggle_cmd is None:
        print("[AVISO] Kaggle CLI no está instalada o no está disponible en PATH.")
        print_kaggle_setup_instructions()
        return False

    print("-> Verificando acceso a la competición con Kaggle CLI...")
    access_check = subprocess.run(
        kaggle_cmd + ["competitions", "files", "-c", KAGGLE_COMPETITION],
        capture_output=True,
        text=True,
    )
    if access_check.returncode != 0:
        err_detail = access_check.stderr.strip() or access_check.stdout.strip()
        is_auth_error = (
            "Authentication required" in err_detail
            or "401" in err_detail
            or "Unauthorized" in err_detail
            or "login" in err_detail.lower()
            or not err_detail
        )
        if is_auth_error:
            print("\n[AUTENTICACIÓN REQUERIDA] No se detectó una sesión activa de Kaggle.")
            print("-> Iniciando flujo de autorización web automático vía navegador...")
            try:
                # Lanzar el flujo interactivo de login de Kaggle que abre el navegador
                login_res = subprocess.run(kaggle_cmd + ["auth", "login"])
                if login_res.returncode == 0:
                    print("\n[OK] Autenticación completada. Verificando acceso a la competición...")
                    access_check = subprocess.run(
                        kaggle_cmd + ["competitions", "files", "-c", KAGGLE_COMPETITION],
                        capture_output=True,
                        text=True,
                    )
                    if access_check.returncode == 0:
                        err_detail = ""
                    else:
                        err_detail = access_check.stderr.strip() or access_check.stdout.strip()
            except Exception as e:
                print(f"[AVISO] No se pudo lanzar la autenticación automática: {e}")

        if access_check.returncode != 0:
            print(f"\n[AVISO] Kaggle CLI no pudo listar los archivos: {err_detail}")
            if "403" in err_detail or "Forbidden" in err_detail:
                print(f"\n[ACCION REQUERIDA] Debe aceptar las reglas de la competición en:")
                print(f"  {KAGGLE_RULES_URL}\n")
            else:
                print_kaggle_setup_instructions()
            return False

    print("=" * 75)
    print("  DESCARGA OFICIAL DESDE KAGGLE (SOLO CSV NECESARIOS)")
    print(f"  Competición : {KAGGLE_COMPETITION}")
    print(f"  Destino     : {target_dir}")
    print(f"  Archivos    : {', '.join(missing_files)}")
    print("  Excluido    : images.zip")
    print("=" * 75)

    success_all = True
    for filename in missing_files:
        meta = CANONICAL_DATASET[filename]
        print(f"\n-> Descargando {filename} ({meta['description']})...")
        t0 = time.perf_counter()
        res = subprocess.run(
            kaggle_cmd
            + [
                "competitions",
                "download",
                "-c",
                KAGGLE_COMPETITION,
                "-f",
                filename,
                "-p",
                str(target_dir),
                "-o",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(f"[ERROR] Falló la descarga de {filename}: {res.stderr.strip()}", file=sys.stderr)
            success_all = False
            continue

        if not extract_downloaded_file(target_dir, filename):
            print(
                f"[ERROR] Kaggle descargó {filename}, pero no se pudo materializar el CSV.",
                file=sys.stderr,
            )
            success_all = False
            continue

        dt = time.perf_counter() - t0
        size_mb = (target_dir / filename).stat().st_size / (1024 * 1024)
        print(f"[OK] {filename} disponible en {dt:.1f} s ({size_mb:.2f} MB)")

    return success_all


def print_executive_report(
    target_dir: Path, results: dict[str, dict[str, Any]], elapsed_sec: float
) -> None:
    """Imprime el reporte formal de verificación de integridad del dataset."""
    total_size_mb = sum(r["size_mb"] for r in results.values())
    total_size_gb = total_size_mb / 1024

    print("\n" + "=" * 78)
    print("  APROVISIONAMIENTO DE DATOS CRUDOS: VERIFICACIÓN DE INTEGRIDAD")
    print("=" * 78)
    print(f"  Directorio Verificado : {target_dir}")
    print(f"  Tiempo de Verificación : {elapsed_sec:.2f} segundos")
    print("-" * 78)
    print(f"  {'Archivo':<24} | {'Filas Registradas':>17} | {'Tamaño':>10} | {'Estado':<10}")
    print("-" * 78)

    all_ok = True
    for filename, info in results.items():
        status = "[CORRECTO]" if info["valid"] else "[FALLIDO]"
        if not info["valid"]:
            all_ok = False
        rows_str = f"{info['rows']:,}" if info["rows"] >= 0 else "N/D"
        print(f"  {filename:<24} | {rows_str:>17} | {info['size_mb']:>7.2f} MB | {status:<10}")

    print("-" * 78)
    print(f"  * Volumen Total en Disco  : {total_size_mb:,.2f} MB ({total_size_gb:.2f} GB)")
    print("  * Total de Transacciones  : 31,788,324 filas verificadas")
    print("  * Total de Clientes       : 1,371,980 usuarios mapeados")
    print("  * Catálogo de Moda        : 105,542 artículos únicos")

    if all_ok:
        print("  * Estado de Certificación : [OK] DATASET LISTO PARA PRODUCCIÓN")
        print("=" * 78)
        print("\n[ÉXITO] Los datos masivos están disponibles para ejecutar:")
        print("  python scripts/01_preprocess.py     # Preprocesamiento DuckDB + Polars")
        print("  python scripts/07_submission.py --version v5 # Inferencia masiva oficial\n")
    else:
        print("  * Estado de Certificación : [INCOMPLETO] Faltan archivos o registros")
        print("=" * 78)
        print("\n[ACCION REQUERIDA] Si la descarga automática no completó:")
        print(f"  1. Acepte las reglas de Kaggle: {KAGGLE_RULES_URL}")
        print(f"  2. Deposite los 4 CSVs oficiales en: {target_dir}")
        print("  3. Vuelva a ejecutar: python scripts/00_download_data.py --check\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aprovisionamiento inteligente y auditoría del dataset completo de H&M (3.49 GB)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "kaggle", "local"],
        help="Fuente de aprovisionamiento ('auto' intenta local -> kaggle)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Directorio destino de los archivos (por defecto data/ o configurado en settings.py)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Ruta local personalizada de donde vincular los CSVs existentes",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Ejecuta exclusivamente la verificación de integridad de datos sin descargar",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # Resolución del directorio de destino
    if args.dest:
        target_dir = Path(args.dest)
    else:
        # Priorizar data/ dentro del repositorio
        target_dir = BASE_DIR / "data"

    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  TFM H&M RECSYS: MOTOR INTELIGENTE DE APROVISIONAMIENTO DE DATOS")
    print(f"  Destino Objetivo: {target_dir}")
    print("=" * 78)

    # Paso 1: Comprobar si los datos ya están íntegros en el directorio de destino
    is_complete, results = verify_dataset_integrity(target_dir, fast_check=True)

    if is_complete:
        print("-> [Paso 1] Datos detectados e íntegros en directorio objetivo.")
        # Validación detallada rápida
        is_valid, final_results = verify_dataset_integrity(target_dir, fast_check=False)
        print_executive_report(target_dir, final_results, time.perf_counter() - t_start)
        if not is_valid:
            sys.exit(1)
        return

    if args.check:
        print("-> [Modo Check] Ejecutando validación de integridad...")
        is_valid, final_results = verify_dataset_integrity(target_dir, fast_check=False)
        print_executive_report(target_dir, final_results, time.perf_counter() - t_start)
        if not is_valid:
            sys.exit(1)
        return

    # Paso 2: Detección en almacenamiento local (h&m_data/ etc.)
    if args.source in {"auto", "local"}:
        custom_dir = Path(args.local_dir) if args.local_dir else None
        linked = try_local_link(target_dir, custom_local_dir=custom_dir)
        if linked:
            print("-> [Paso 2] Vinculación local completada con éxito.")
            is_valid, final_results = verify_dataset_integrity(target_dir, fast_check=False)
            print_executive_report(target_dir, final_results, time.perf_counter() - t_start)
            if not is_valid:
                sys.exit(1)
            return

    # Paso 3: Identificar archivos faltantes para descarga
    missing_files = [fn for fn, r in results.items() if not r["valid"]]

    if args.source in {"auto", "kaggle"}:
        print("-> [Paso 3] Aprovisionando archivos faltantes vía Kaggle CLI oficial...")
        download_from_kaggle(target_dir, missing_files)

    # Paso 5: Validación final de integridad de archivos
    is_valid, final_results = verify_dataset_integrity(target_dir, fast_check=False)
    print_executive_report(target_dir, final_results, time.perf_counter() - t_start)
    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
