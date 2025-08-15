import hydra
from omegaconf import DictConfig

from kygs.utils.common import get_config_path


CONFIG_NAME = "config_print_messages"


def print_messages(cfg: DictConfig) -> None:
    mp = hydra.utils.call(cfg.message_provider)
    mp.display_messages()


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(print_messages)()
