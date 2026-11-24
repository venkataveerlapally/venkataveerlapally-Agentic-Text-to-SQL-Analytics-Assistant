import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from warehouse.seed import build_sqlite  # noqa: E402

if not Path(settings.sqlite_path).exists():
    build_sqlite(Path(settings.sqlite_path))
