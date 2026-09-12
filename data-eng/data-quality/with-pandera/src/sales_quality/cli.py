import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .pipeline import QualityPipeline


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    report = QualityPipeline(
        to_absolute_path(cfg.data.path), to_absolute_path(cfg.report.output_path)
    ).run()
    print({"valid": report.valid, "rows": report.rows, "errors": len(report.errors)})
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
