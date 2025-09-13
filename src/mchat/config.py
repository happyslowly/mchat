import os
import tomllib
from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str
    summary_model: str | None = None
    max_history_turns: int = -1
    timeout: int = -1
    save_interval: int = 300
    continue_last_session: bool = True

    google_api_key: str
    google_search_engine_id: str


def _load_config() -> Config:
    if "XDG_CONFIG_HOME" in os.environ:
        config_dir = Path(os.environ["XDG_CONFIG_HOME"])
    else:
        config_dir = Path.home() / ".config"
    config_file = config_dir / "mchat" / "config.toml"
    try:
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
        return Config(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_file}")


_config = _load_config()


def get_config() -> Config:
    return _config
