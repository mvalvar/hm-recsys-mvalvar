"""Configuración compartida y fixtures globales de pytest."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Inicializar numpy una única vez antes de cualquier importación de test o tracing de coverage
import numpy as np  # noqa: F401

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

