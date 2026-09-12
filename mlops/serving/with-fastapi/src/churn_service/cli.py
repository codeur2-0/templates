import hydra
from omegaconf import DictConfig

from .main import model
from .schemas import ChurnRequest


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    request = ChurnRequest(tenure=4, monthly_spend=29.0, support_tickets=3)
    print(model.predict_probability(request))


if __name__ == "__main__":
    main()
