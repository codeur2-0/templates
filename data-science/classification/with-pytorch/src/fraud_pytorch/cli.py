import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import FraudTrainingPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = FraudTrainingPipeline(
        to_absolute_path(cfg.data.path), to_absolute_path(cfg.training.output_dir), cfg.seed
    ).run(
        test_size=cfg.training.test_size,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        hidden_dim=cfg.model.hidden_dim,
    )
    print(metrics)


if __name__ == "__main__":
    main()
