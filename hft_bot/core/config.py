"""Configuration management with validation and environment support"""

import os
from typing import Dict, Optional, Any
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import yaml
import json

# Load environment variables
load_dotenv()

class TradingParams(BaseModel):
    """Trading parameters with validation"""
    decision_interval: float = Field(0.5, gt=0, le=10)
    min_signal_confluences: int = Field(2, ge=1, le=7)
    min_signal_strength: float = Field(0.5, ge=0, le=1)
    gross_take_profit: float = Field(0.0008, gt=0, le=0.01)
    gross_stop_loss: float = Field(0.0004, gt=0, le=0.01)
    max_position_size: float = Field(500, gt=0)
    max_latency_ms: float = Field(100, gt=0, le=1000)
    kelly_fraction: float = Field(0.25, gt=0, le=1)
    ml_lookback: int = Field(20, ge=5, le=100)
    orderbook_levels: int = Field(10, ge=1, le=20)
    commission_rate: float = Field(0.0001, ge=0, le=0.001)
    expected_slippage: float = Field(0.00005, ge=0, le=0.001)
    order_timeout: int = Field(30, gt=0, le=300)

class RiskParams(BaseModel):
    """Risk management parameters"""
    max_drawdown: float = Field(0.15, gt=0, le=0.5)
    max_positions: int = Field(3, ge=1, le=10)
    max_exposure: float = Field(0.5, gt=0, le=1)
    daily_loss_limit: float = Field(0.05, gt=0, le=0.2)
    position_timeout: int = Field(600, gt=0)

class EmailConfig(BaseModel):
    """Email alert configuration"""
    enabled: bool = False
    smtp_server: str = Field(default="")
    smtp_port: int = Field(default=587)
    from_email: str = Field(default="")
    to_email: str = Field(default="")
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None

class AlertConfig(BaseModel):
    """Alert configuration"""
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        'latency_ms': 150,
        'drop_rate': 0.05,
        'drawdown': 0.10,
        'consecutive_losses': 5
    })
    email: EmailConfig = Field(default_factory=EmailConfig)
    slack_webhook: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

class BotConfig(BaseSettings):
    """Main bot configuration with environment variable support"""
    symbol: str = Field(default="BTCUSDT")
    paper_trading: bool = Field(default=True)
    initial_capital: float = Field(default=1000, gt=0)
    strategy: str = Field(default="momentum")
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    trading_params: TradingParams = Field(default_factory=TradingParams)
    risk: RiskParams = Field(default_factory=RiskParams)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    api_key: Optional[str] = Field(default=None)
    api_secret: Optional[str] = Field(default=None)
    testnet: bool = Field(default=False)
    
    class Config:
        env_prefix = "BOT_"
        env_nested_delimiter = "__"
        
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if not v.endswith('USDT'):
            raise ValueError('Only USDT pairs supported')
        return v
    
    @field_validator('api_key', 'api_secret', mode='before')
    @classmethod
    def load_from_env(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            field_name = info.field_name
            env_name = f"BINANCE_{field_name.upper()}"
            v = os.getenv(env_name)
        return v

class ConfigProvider:
    """Manages configuration from multiple sources with caching"""
    
    def __init__(self, config_file: Optional[str] = None, override_env: bool = True):
        self.config_file = Path(config_file or os.getenv('BOT_CONFIG_FILE', 'config/bot_config.yaml'))
        self.override_env = override_env
        self._config: Optional[BotConfig] = None
        
    def load(self) -> BotConfig:
        """Load configuration from file and environment with caching"""
        if self._config is None:
            # Load from file
            config_dict = self._load_file() if self.config_file.exists() else {}
            
            # Override with environment variables
            if self.override_env:
                # BotConfig will automatically load from env due to BaseSettings
                pass
                
            # Create and validate config
            self._config = BotConfig(**config_dict)
            
        return self._config
    
    def reload(self) -> BotConfig:
        """Force reload configuration"""
        self._config = None
        return self.load()
    
    def _load_file(self) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        with open(self.config_file, 'r') as f:
            if self.config_file.suffix in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            elif self.config_file.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_file.suffix}")
    
    def save(self, config: BotConfig, path: Optional[Path] = None) -> None:
        """Save configuration to file"""
        save_path = path or self.config_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.model_dump()
        
        with open(save_path, 'w') as f:
            if save_path.suffix in ('.yaml', '.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
