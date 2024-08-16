import yaml
from typing import Dict, Any


def save_yaml(config: Dict[str, Any]) -> None:
    with open("config.yaml", "w") as file:
        yaml.dump(config, file)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config
