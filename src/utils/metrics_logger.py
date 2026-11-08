"""Serialización y carga de métricas en JSON y Joblib para la interfaz Streamlit."""
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Codificador JSON compatible con tipos NumPy y Pandas."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_metrics_json(data: Dict[str, Any], path: Path) -> None:
    """Guarda un diccionario de métricas en formato JSON legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)


def load_metrics_json(path: Path) -> Dict[str, Any]:
    """Carga un archivo JSON de métricas previamente serializado."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
