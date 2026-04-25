# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.0.0"

from ultralytics.models import YOLO
from ultralytics.utils import SETTINGS

# Personal fork - exposing NAS and RTDETR models at top level for easier access
from ultralytics.models import NAS, RTDETR

# Print a friendly message when the package is imported (useful for debugging env issues)
import logging
logging.getLogger(__name__).debug("ultralytics %s loaded (personal fork)", __version__)

# Disable analytics/telemetry by default in this personal fork
# Also set a consistent default imgsz and disable verbose output for cleaner logs
# Note: increased workers from default 8 -> 4 to avoid OOM on my 16GB machine
SETTINGS.update({
    "sync": False,
    "verbose": False,
    "imgsz": 640,  # explicitly pin default image size to avoid surprises across runs
    "workers": 4,  # reduced from 8; my machine struggles with the default during multi-GPU runs
})

__all__ = [
    "__version__",
    "YOLO",
    "NAS",
    "RTDETR",
    "SETTINGS",
]
