import os
import tempfile
import tomllib
from pathlib import Path

from pydantic import BaseModel, field_validator


class Config(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str
    summary_model: str | None = None
    summary_interval_in_turns: int = 5
    timeout: int = -1
    save_interval: int = 300
    continue_last_session: bool = True
    workspace: str = tempfile.gettempdir()
    google_api_key: str
    google_search_engine_id: str

    @field_validator("summary_interval_in_turns")
    @classmethod
    def ensure_interval_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("summary_interval_in_turns must be >= 1")
        return value


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
