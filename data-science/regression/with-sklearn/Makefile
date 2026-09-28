# ===============================================================================================
# Estimation du prix de vente immobilier (AVM) avec scikit-learn
# ------------------------------------------------------------------------------------------------
# Raccourcis de développement. `make help` liste les cibles disponibles.
#
# Règle d'or du dépôt : en 5 commandes maximum, n'importe qui peut
#   générer les données -> entraîner -> évaluer -> visualiser -> prédire.
#       make install && make data && make train && make evaluate && make predict
# ===============================================================================================

PYTHON      ?= python
PIP         ?= pip
PROJECT     := real-estate-avm-sklearn
MODE        ?= train
OVERRIDES   ?=
DATA_ARGS   ?=
TRAIN_ARGS  ?=
EVAL_ARGS   ?=
PRED_ARGS   ?=
NOTEBOOKS   := $(sort $(wildcard notebooks/*.ipynb))

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test coverage data train evaluate predict \
        all smoke verify notebooks notebooks-update clean clean-artifacts tree hydra-help info

help: ## Affiche cette aide
	@echo ""
	@echo "  $(PROJECT) — cibles disponibles"
	@echo "  ------------------------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

info: ## Affiche la configuration résolue (Hydra)
	$(PYTHON) -m src.main --cfg job mode=train $(OVERRIDES)

install: ## Installe les dépendances runtime
	$(PIP) install -r requirements.txt

install-dev: ## Installe les dépendances runtime + dev (tests, lint, notebooks)
	$(PIP) install -r requirements-dev.txt

lint: ## Vérifie le style (ruff)
	$(PYTHON) -m ruff check src scripts tests

format: ## Formate le code (ruff format + fix)
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check --fix src scripts tests

typecheck: ## Vérifie le typage (mypy)
	$(PYTHON) -m mypy src scripts

test: ## Exécute les tests unitaires
	$(PYTHON) -m pytest -q $(OVERRIDES)

coverage: ## Exécute les tests avec couverture
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing --cov-report=xml

data: ## Génère le jeu de données synthétique (data/raw)
	$(PYTHON) scripts/generate_data.py $(DATA_ARGS)

train: ## Entraîne le modèle et sauvegarde les artefacts
	$(PYTHON) scripts/train.py $(TRAIN_ARGS)

evaluate: ## Évalue le modèle entraîné (rapports + figures + métriques)
	$(PYTHON) scripts/evaluate.py $(EVAL_ARGS)

predict: ## Prédit sur un échantillon ou un fichier
	$(PYTHON) scripts/predict.py $(PRED_ARGS)

all: ## Enchaîne : données -> train -> évaluation -> prédiction
	$(PYTHON) -m src.main mode=all $(OVERRIDES)

smoke: ## Exécution rapide de bout en bout (petit dataset, peu d'époques)
	$(PYTHON) -m src.main mode=all data.n_samples=600 ++train.epochs=3 $(OVERRIDES)

verify: ## Gate qualité complet : lint + type + tests + smoke
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) smoke

notebooks: ## Exécute tous les notebooks de bout en bout (vérification CI)
	$(PYTHON) scripts/run_notebooks.py

notebooks-update: ## Exécute les notebooks et réécrit leurs sorties
	$(PYTHON) scripts/run_notebooks.py --inplace

clean: ## Supprime les sorties temporaires (Hydra, caches)
	rm -rf outputs multirun .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

clean-artifacts: ## Supprime les artefacts et les données générées
	rm -rf artifacts/models/* artifacts/reports/* artifacts/figures/* artifacts/metrics/*
	rm -rf data/raw/* data/processed/* data/external/*
	@find artifacts data -type d -empty -exec touch {}/.gitkeep \; 2>/dev/null || true

tree: ## Affiche l'arborescence du projet
	@find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" \
		-o -name ".ruff_cache" -o -name "outputs" -o -name "multirun" -o -name ".ipynb_checkpoints" \) -prune \
		-o -type f -print | sort | sed 's|[^/]*/|  |g'

hydra-help: ## Affiche l'aide Hydra (groupes et overrides disponibles)
	$(PYTHON) -m src.main --help
