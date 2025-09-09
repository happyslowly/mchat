import os
import tomllib
from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str
    summary_model: str | None
    max_history_turns: int = -1
    timeout: int = -1
    save_interval: int = 300
    continue_last_session: bool = True


def load_config() -> Config:
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
            summary_model=config.get("summary_model", None),
            max_history_turns=config.get("max_history_turns", -1),
            timeout=config.get("timeout", -1),
            save_interval=config.get("save_interval", 300),
            continue_last_session=config.get("continue_last_session", True),
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_file}")
