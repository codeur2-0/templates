import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .runner import ExperimentRunner


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    result = ExperimentRunner(to_absolute_path(cfg.tracking_uri), cfg.experiment, cfg.seed).run(
        to_absolute_path(cfg.data.path), cfg.data.target, cfg.training.test_size, cfg.training.c
    )
    print(result)


if __name__ == "__main__":
    main()
