"""Script Maestro: Regeneración de la totalidad de Figuras y Diagramas del TFM.

Máster en Data Science, Big Data & Business Analytics: Universidad Complutense de Madrid
TFM: Motor de Recomendación Escalable para Retail de Moda (H&M RecSys Challenge)
Autor: Manuel Valdivia

Ejecuta secuencialmente los generadores oficiales de gráficos:
1. scripts/generate_architecture_diagrams.py -> Diagramas 1 a 8 (diag_01 a diag_08)
2. scripts/generate_eda_figures.py           -> Cap. 4 (Figuras 4.1 a 4.12)
3. scripts/generate_candidate_figures.py     -> Cap. 5 (Figuras 5.1 a 5.3)
4. scripts/generate_feature_importance_figure.py -> Cap. 7 (Figura 7.1)
5. scripts/generate_ablation_figures.py      -> Cap. 9 (Figuras 9.1 a 9.3)
6. scripts/06_xai_shap.py                   -> Cap. 10 (Figuras 10.1 a 10.2)

Uso:
    python scripts/generate_all_figures.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PIPELINE = [
    ("Diagramas Arquitectónicos (01 a 08)", "scripts/generate_architecture_diagrams.py"),
    ("Capítulo 4 (EDA: Figuras 4.01 a 4.12)", "scripts/generate_eda_figures.py"),
    ("Capítulo 5 (Candidatos: Figuras 5.01 a 5.03)", "scripts/generate_candidate_figures.py"),
    ("Capítulo 7 (Feature Importance: Figura 7.01)", "scripts/generate_feature_importance_figure.py"),
    ("Capítulo 9 (Ablación y Pareto: Figuras 9.01 a 9.03)", "scripts/generate_ablation_figures.py"),
    ("Capítulo 10 (XAI y SHAP: Figuras 10.01 a 10.02)", "scripts/06_xai_shap.py"),
]


def main() -> None:
    print("=" * 80)
    print("  SUITE MAESTRA: REGENERACIÓN COMPLETA DE FIGURAS Y DIAGRAMAS DEL TFM")
    print("=" * 80)
    t0_global = time.time()

    for idx, (label, script_rel) in enumerate(PIPELINE, 1):
        script_path = BASE_DIR / script_rel
        print(f"\n[{idx}/{len(PIPELINE)}] Ejecutando {label} ({script_rel})...")
        t0 = time.time()
        res = subprocess.run([sys.executable, "-X", "utf8", str(script_path)], cwd=str(BASE_DIR))
        if res.returncode != 0:
            print(
                f"[ERROR] Falló la ejecución de {script_rel} con código {res.returncode}",
                file=sys.stderr,
            )
            sys.exit(res.returncode)
        dt = time.time() - t0
        print(f"  -> Completado en {dt:.1f} s")

    dt_global = time.time() - t0_global
    print("\n" + "=" * 80)
    print(f"  [EXITO] TODAS LAS FIGURAS Y DIAGRAMAS REGENERADOS EN {dt_global:.1f} SEGUNDOS")
    print("=" * 80)


if __name__ == "__main__":
    main()
