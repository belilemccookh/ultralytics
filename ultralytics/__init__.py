# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.0.0"

from ultralytics.models import YOLO
from ultralytics.utils import SETTINGS

# Personal fork - exposing NAS and RTDETR models at top level for easier access
from ultralytics.models import NAS, RTDETR

__all__ = [
    "__version__",
    "YOLO",
    "NAS",
    "RTDETR",
    "SETTINGS",
]
