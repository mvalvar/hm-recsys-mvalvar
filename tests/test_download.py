"""Suite de Pruebas Unitarias para el Aprovisionamiento y Auditoría de Datos (Script 00).

Valida:
1. Retorno de código de salida no-cero cuando la auditoría (--check) falla en un directorio vacío o corrupto.
2. Comprobación defensiva local sin uso de red ni descargas remotas de Kaggle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = BASE_DIR / "scripts" / "00_download_data.py"


def test_audit_fails_on_empty_directory(tmp_path: Path):
    """Valida que --check retorne código de salida no-cero cuando el directorio objetivo está vacío."""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT_PATH), "--check", "--dest", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode != 0, (
        f"Se esperaba código de salida no-cero ante dataset vacío, obtenido {result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
    assert "[FALLIDO]" in result.stdout
    assert "[INCOMPLETO]" in result.stdout


def test_audit_fails_on_corrupt_files(tmp_path: Path):
    """Valida que --check retorne código de salida no-cero si los archivos existen pero están vacíos/truncados."""
    # Crear archivos vacíos simulando descarga interrumpida
    for filename in [
        "articles.csv",
        "customers.csv",
        "transactions_train.csv",
        "sample_submission.csv",
    ]:
        (tmp_path / filename).write_text("dummy", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT_PATH), "--check", "--dest", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode != 0, (
        f"Se esperaba código de salida no-cero ante archivos truncados, obtenido {result.returncode}."
    )
