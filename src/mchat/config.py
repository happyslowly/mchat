import os
import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Config(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str
    summary_model: str | None = None
    max_history_turns: int = -1
    timeout: int = -1
    save_interval: int = 300


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _config: Optional[Config] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def reload(self) -> Config:
        self._config = self._load_config()
        return self._config

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    @staticmethod
    def _load_config() -> Config:
        if "XDG_CONFIG_HOME" in os.environ:
            config_dir = Path(os.environ["XDG_CONFIG_HOME"])
        else:
            config_dir = Path.home() / ".config"
        config_file = config_dir / "mchat" / "config.toml"
        try:
            with open(config_file, "rb") as f:
                config = tomllib.load(f)
            return Config(
                base_url=config["base_url"],
                api_key=config.get("api_key"),
                model=config["model"],
                summary_model=config.get("summary_model"),
                max_history_turns=config.get("max_history_turns", -1),
                timeout=config.get("timeout", -1),
                save_interval=config.get("save_interval", 300),
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")


config_manager = ConfigManager()
