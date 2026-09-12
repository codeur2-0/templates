import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import LoanTrainingPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = LoanTrainingPipeline(
        to_absolute_path(cfg.data.path), to_absolute_path(cfg.training.output_dir), cfg.seed
    ).run(
        cfg.training.test_size,
        cfg.training.epochs,
        cfg.training.batch_size,
        cfg.model.hidden_units,
        cfg.model.learning_rate,
    )
    print(metrics)


if __name__ == "__main__":
    main()
