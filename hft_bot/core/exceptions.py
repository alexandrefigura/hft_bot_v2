"""Exceções específicas do HFT Bot (facilita tratamento granular)."""


class HFTBotException(Exception):
    """Base para todas as exceções do projeto."""


# Configuração / validação --------------------------------------------------


class ConfigurationError(HFTBotException):
    """Erro de configuração inválida ou ausente."""


# Exchange / ordens ---------------------------------------------------------


class ExchangeError(HFTBotException):
    """Falha genérica relacionada à exchange."""


class InsufficientBalanceError(ExchangeError):
    """Saldo insuficiente.

    Args:
        required: Valor necessário.
        available: Saldo disponível.
        asset: Ativo correspondente.
    """

    def __init__(self, required: float, available: float, asset: str):
        super().__init__(
            f"Saldo insuficiente de {asset}: necessário {required:.8f}, disponível {available:.8f}"
        )
        self.required = required
        self.available = available
        self.asset = asset


class OrderError(ExchangeError):
    """Falha ao criar/cancelar ordem."""


# Risco / estratégia --------------------------------------------------------


class RiskLimitError(HFTBotException):
    """Alguma regra de risco foi violada."""


class DataError(HFTBotException):
    """Problema ao carregar ou processar dados de mercado."""
