import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import ChurnTrainingPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pipeline = ChurnTrainingPipeline(
        data_path=to_absolute_path(cfg.data.path),
        output_dir=to_absolute_path(cfg.training.output_dir),
        seed=cfg.seed,
    )
    metrics = pipeline.run(
        target=cfg.data.target,
        test_size=cfg.training.test_size,
        c=cfg.model.c,
        max_iter=cfg.model.max_iter,
    )
    print({key: round(value, 4) for key, value in metrics.items()})


if __name__ == "__main__":
    main()
