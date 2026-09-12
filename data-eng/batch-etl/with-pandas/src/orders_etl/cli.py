import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import OrderBatchPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    result = OrderBatchPipeline(
        to_absolute_path(cfg.data.path), to_absolute_path(cfg.data.output_path)
    ).run()
    print(result)


if __name__ == "__main__":
    main()
