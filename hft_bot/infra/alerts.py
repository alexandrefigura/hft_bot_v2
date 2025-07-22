"""Gestão de alertas (placeholder simples)."""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.thresholds = cfg.get("thresholds", {})

    # ---------------------- API usada pelo bot ---------------------- #
    async def send_alert(self, message: str, severity: str = "info"):
        """Envia alerta (neste stub, apenas loga)."""
        logger.warning("[ALERT][%s] %s", severity.upper(), message)

    def check_thresholds(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Retorna lista de violações de limite."""
        breaches: List[Dict[str, Any]] = []
        for metric, threshold in self.thresholds.items():
            value = metrics.get(metric)
            if value is None:
                continue
            if (metric == "drawdown" and value >= threshold) or (
                metric != "drawdown" and value > threshold
            ):
                breaches.append(
                    {
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": "warning" if metric == "drawdown" else "info",
                    }
                )
        return breaches
