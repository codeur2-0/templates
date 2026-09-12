import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import SentimentTrainingPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = SentimentTrainingPipeline(
        to_absolute_path(cfg.data.path), to_absolute_path(cfg.training.output_dir), cfg.seed
    ).run(
        cfg.training.epochs,
        cfg.training.batch_size,
        cfg.model.vocabulary_size,
        cfg.model.sequence_length,
        cfg.model.embedding_dim,
    )
    print(metrics)


if __name__ == "__main__":
    main()
