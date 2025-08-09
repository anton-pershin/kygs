from pathlib import Path

import hydra
from hydra.utils import instantiate 
from omegaconf import DictConfig

from kygs.utils.common import get_config_path


CONFIG_NAME = "config_collect_recent_posts"


def collect_recent_posts(cfg: DictConfig) -> None:
    ...
    

if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(collect_recent_posts)()
