"""Persistência simples de estado em arquivo JSON."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StatePersistence:
    def __init__(self, base_dir: str | Path = "state"):
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ---------------------- API usada pelo bot ---------------------- #
    async def start_sync(self):
        # Placeholder para sincronização assíncrona (ex.: S3, DB)
        pass

    async def stop_sync(self):
        pass

    async def save_state(self, name: str, data: Dict[str, Any]):
        path = self.base_path / f"{name}.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, default=str)

    async def load_state(self, name: str) -> Dict[str, Any] | None:
        path = self.base_path / f"{name}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
