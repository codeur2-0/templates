import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .application import RAGApplication


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    app = RAGApplication(
        to_absolute_path(cfg.data.path),
        to_absolute_path(cfg.data.persist_dir),
        cfg.data.collection,
        cfg.retrieval.top_k,
    )
    print({"indexed": app.ingest()})
    answer = app.answer(cfg.query)
    print(answer.text)
    print({"sources": answer.sources})


if __name__ == "__main__":
    main()
