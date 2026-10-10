"""La fabrique des stratégies : quelle source d'hyper-paramètres gagne, et dans quel ordre.

Un projet de gabarit vit de cette règle, parce que tout le monde s'appuie dessus : un notebook qui
réduit l'architecture, un test qui veut une suite rapide, une grille d'hyper-paramètres qui compare
deux valeurs d'un réglage. L'ordre documenté est **défauts du registre → configuration à plat
(``model.params.units``) → bloc explicite (``model.params.training.units``) → surcharge passée à
``build_model``**, et ces tests le vérifient source par source.

Le test qui compte le plus est le premier : les réglages à plat de la configuration doivent
atteindre le modèle. Sans lui, une surcharge silencieusement ignorée fait tourner un notebook
avec l'architecture complète — c'est-à-dire avec un résultat qui ne correspond plus à ce qu'il
annonce.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models import build_model, describe_algorithm
from src.models.summarizer import EncoderDecoderSummarizer
from src.schemas.config import AppConfig, validate_config

#: Racine du projet : la configuration est celle de ``python -m src.main``, pas une copie.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def default_config() -> AppConfig:
    """Compose the configuration **without** the test overrides, to read the registry defaults."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        return validate_config(compose(config_name="config", overrides=["mode=train"]))


def test_a_flat_setting_reaches_the_model(app_config: AppConfig) -> None:
    """``model.params.layers``, ``units``, ``vocab_size`` : la configuration à plat est appliquée.

    ``app_config`` porte les surcharges minuscules de la suite de tests, écrites **à plat** :
    si elles n'atteignent pas le modèle, toute la suite tourne sur l'architecture de
    référence — plus lente, et surtout différente de ce que les tests annoncent.
    """
    model = build_model(app_config, algorithm="transformer_tiny")
    assert isinstance(model, EncoderDecoderSummarizer)
    assert model.layers == 1
    assert model.units == 48
    assert model.tokenizer is not None
    assert model.vocab_size == 320


def test_the_registry_defaults_apply_when_the_configuration_is_silent(
    default_config: AppConfig,
) -> None:
    """Sans réglage écrit, ce sont les défauts du registre qui s'appliquent, et ils sont publiés."""
    model = build_model(default_config, algorithm="transformer_tiny")
    assert isinstance(model, EncoderDecoderSummarizer)
    defaults = describe_algorithm("transformer_tiny")["default_params"]["training"]
    assert model.layers == defaults["layers"]
    assert model.units == defaults["units"]
    assert model.vocab_size == defaults["vocab_size"]


def test_a_block_setting_beats_the_flat_setting(default_config: AppConfig) -> None:
    """Un réglage écrit dans son bloc (``training.units``) l'emporte sur le nom écrit à plat."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        config = validate_config(
            compose(
                config_name="config",
                overrides=[
                    "mode=train",
                    "model.params.units=64",
                    "+model.params.training.units=96",
                ],
            )
        )
    model = build_model(config, algorithm="transformer_tiny")
    assert isinstance(model, EncoderDecoderSummarizer)
    assert model.units == 96


def test_an_explicit_override_wins_over_the_configuration(app_config: AppConfig) -> None:
    """La surcharge passée à ``build_model`` est la dernière source : elle gagne."""
    model = build_model(app_config, algorithm="transformer_tiny", params={"units": 160})
    assert isinstance(model, EncoderDecoderSummarizer)
    assert model.units == 160
    assert model.layers == 1  # le reste de la configuration est intact


def test_the_budget_is_shared_by_the_extractive_strategies(app_config: AppConfig) -> None:
    """Le budget de longueur est le même pour toutes les stratégies comparées.

    Une comparaison entre une extractive bornée à 60 jetons et un modèle borné à 90 ne compare pas
    deux stratégies : elle compare deux budgets.
    """
    served = build_model(app_config, algorithm="transformer_tiny")
    lead = build_model(app_config, algorithm="lead")
    textrank = build_model(app_config, algorithm="textrank")
    assert {served.max_output_tokens, lead.max_output_tokens, textrank.max_output_tokens} == {
        served.max_output_tokens
    }
    assert {served.compression, lead.compression, textrank.compression} == {served.compression}


def test_an_unknown_algorithm_is_refused(app_config: AppConfig) -> None:
    """Un algorithme inconnu lève une erreur explicite, avec la liste des noms valides."""
    with pytest.raises(ValueError, match="transformer_tiny"):
        build_model(app_config, algorithm="gpt-4")


def test_every_declared_algorithm_is_described(app_config: AppConfig) -> None:
    """Chaque stratégie annoncée est constructible et décrite (les notebooks lisent cette table)."""
    from src.models import available_algorithms

    names = available_algorithms()
    assert names == sorted(names)
    assert {"transformer_tiny", "lead", "textrank"} <= set(names)
    for name in names:
        description = describe_algorithm(name)
        assert description["family"] in {"learned", "extractive"}
        assert description["notes"]
        assert build_model(app_config, algorithm=name).strategy
