from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class AppConfig:
    random_state: int
    test_size: float
    target_column: str
    tracking_uri: str
    experiment_name: str
    artifacts_dir: str
    models: dict

def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return AppConfig(**cfg)
