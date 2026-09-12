import hydra
from omegaconf import DictConfig

from .schemas import PredictionRequest
from .service import RiskModelService


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    service = RiskModelService(cfg.service.model_version, cfg.service.threshold)
    print(
        service.predict(
            PredictionRequest(income=50000, loan_amount=8000, credit_score=700)
        ).model_dump()
    )


if __name__ == "__main__":
    main()
