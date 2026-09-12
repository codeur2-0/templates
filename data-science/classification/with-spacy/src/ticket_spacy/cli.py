import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import TicketTrainingPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = TicketTrainingPipeline(
        to_absolute_path(cfg.data.path), to_absolute_path(cfg.training.output_dir)
    ).run(cfg.training.epochs, cfg.training.dropout)
    print(metrics)


if __name__ == "__main__":
    main()
