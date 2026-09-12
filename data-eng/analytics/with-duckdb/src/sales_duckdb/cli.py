import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import AnalyticsPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(
        AnalyticsPipeline(
            to_absolute_path(cfg.data.path), to_absolute_path(cfg.query.output_path)
        ).run()
    )


if __name__ == "__main__":
    main()
