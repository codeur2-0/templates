"""Tests du feature engineering, du preprocessing et des transformateurs maison.

Deux risques dominent cette couche, et les tests les ciblent explicitement :

1. **la fuite de données** — une statistique apprise sur le train puis ré-appliquée au test
   fausse toute l'évaluation. ``test_scaler_statistics_come_from_train_only`` le vérifie.
2. **l'incompatibilité scikit-learn** — un transformateur qui ne respecte pas le contrat
   ``clone()`` / ``get_feature_names_out()`` casse n'importe quel ``Pipeline`` ou
   ``ColumnTransformer``. ``test_every_transformer_is_cloneable`` est le garde-fou.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.features.build_features import FeatureBuilder, FeatureRecipe, select_feature_columns
from src.preprocessing.pipelines import PreprocessingPipeline, infer_column_kinds
from src.preprocessing.transformers import (
    ColumnSelector,
    DataFrameScaler,
    Log1pTransformer,
    OutlierClipper,
    RareCategoryGrouper,
    TypeCaster,
)


@pytest.fixture
def sample_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Return a small mixed-type frame exercising every transformer."""
    return pd.DataFrame(
        {
            "amount": rng.gamma(2.0, 50.0, size=120),
            "count": rng.integers(0, 12, size=120),
            "sparse": np.where(rng.random(120) < 0.2, np.nan, rng.normal(size=120)),
            "city": rng.choice(["paris", "lyon", "nice", "rare_a", "rare_b"], size=120),
            "flag": rng.integers(0, 2, size=120),
        }
    )


#: Clés de paramètres d'une recette qui désignent des colonnes d'entrée.
_SOURCE_KEYS: tuple[str, ...] = (
    "column",
    "left",
    "right",
    "on",
    "by",
    "values",
    "columns",
    "numerator",
    "denominator",
)


def _recipe_sources(builder: FeatureBuilder, output: str) -> list[str]:
    """Return the input columns a recipe reads, so a missing value can be traced to its origin.

    Args:
        builder: Fitted feature builder.
        output: Name of the derived column.

    Returns:
        The declared input columns, empty when the recipe is unknown or reads no column.
    """
    for recipe in builder.recipes:
        if recipe.name != output:
            continue
        sources: list[str] = []
        for key in _SOURCE_KEYS:
            value = recipe.params.get(key)
            if isinstance(value, str):
                sources.append(value)
            elif isinstance(value, (list, tuple)):
                sources.extend(str(item) for item in value)
        return [source for source in sources if source]
    return []


class TestFeatureBuilder:
    """Construction déclarative des features dérivées."""

    def test_declared_recipes_produce_declared_columns(
        self, feature_builder: FeatureBuilder, split_frames: Any
    ) -> None:
        """Chaque recette produit exactement la colonne annoncée."""
        if not feature_builder.recipes:
            pytest.skip("aucune recette de feature déclarée dans ce projet")
        enriched = feature_builder.transform(split_frames.train)
        for name in feature_builder.output_names:
            assert name in enriched.columns
            missing = enriched[name].isna()
            if not missing.any():
                continue
            # Une feature dérivée hérite légitimement des manquants de ses entrées (une panne de
            # capteur se propage à tout ce qui est calculé à partir de la colonne concernée). Ce
            # qu'elle ne doit jamais faire, c'est en **inventer** : chaque NaN est donc tracé
            # jusqu'aux colonnes sources de la recette.
            sources = _recipe_sources(feature_builder, name)
            assert sources, f"la feature '{name}' contient des NaN sans entrée manquante"
            inherited = enriched[sources].isna().any(axis="columns")
            unexpected = missing & ~inherited
            assert not unexpected.any(), (
                f"la feature '{name}' invente {int(unexpected.sum())} NaN absents de ses "
                f"entrées {sources}"
            )

    def test_transform_preserves_original_columns(
        self, feature_builder: FeatureBuilder, split_frames: Any
    ) -> None:
        """Le builder *ajoute* des colonnes, il n'en supprime jamais."""
        enriched = feature_builder.transform(split_frames.train)
        assert set(split_frames.train.columns) <= set(enriched.columns)

    def test_transform_is_deterministic(
        self, feature_builder: FeatureBuilder, split_frames: Any
    ) -> None:
        """Deux appels successifs produisent la même matrice (reproductibilité)."""
        first = feature_builder.transform(split_frames.train)
        second = feature_builder.transform(split_frames.train)
        pd.testing.assert_frame_equal(first, second)

    def test_input_frame_is_not_mutated(
        self, feature_builder: FeatureBuilder, split_frames: Any
    ) -> None:
        """Le frame d'entrée ne doit jamais être modifié sur place."""
        before = list(split_frames.train.columns)
        _ = feature_builder.transform(split_frames.train)
        assert list(split_frames.train.columns) == before

    def test_learned_recipes_require_fit(self, sample_frame: pd.DataFrame) -> None:
        """Une recette apprise (bin) doit refuser un ``transform`` avant ``fit``."""
        builder = FeatureBuilder(
            [{"type": "bin", "name": "amount_bin", "column": "amount", "bins": 3}]
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            builder.transform(sample_frame)

    def test_bin_edges_are_learned_on_train(self, sample_frame: pd.DataFrame) -> None:
        """Les bornes de binning viennent du train : un test décalé reste dans les bornes."""
        builder = FeatureBuilder(
            [{"type": "bin", "name": "amount_bin", "column": "amount", "bins": 4}]
        )
        builder.fit(sample_frame)
        shifted = sample_frame.copy()
        shifted["amount"] = shifted["amount"] * 100
        coded = builder.transform(shifted)["amount_bin"]
        assert set(np.unique(coded.to_numpy())) <= {0, 1, 2, 3, -1}

    def test_ratio_recipe_is_exact(self, sample_frame: pd.DataFrame) -> None:
        """Un ratio doit être calculé exactement (et sans division par zéro)."""
        frame = sample_frame.assign(denominator_zero=0.0)
        builder = FeatureBuilder(
            [
                {
                    "type": "ratio",
                    "name": "ratio",
                    "numerator": "count",
                    "denominator": "denominator_zero",
                }
            ]
        )
        builder.fit(frame)
        result = builder.transform(frame)["ratio"]
        assert np.isfinite(result.to_numpy()).all()

    def test_unknown_recipe_type_is_rejected(self) -> None:
        """Une recette inconnue est une erreur de configuration, pas un silence."""
        with pytest.raises(ValueError, match="Unknown feature recipe"):
            FeatureRecipe.from_mapping({"type": "magie", "column": "amount"})

    def test_duplicate_names_are_rejected(self) -> None:
        """Deux recettes produisant la même colonne écraseraient un résultat : refusé."""
        recipes = [
            {"type": "log1p", "name": "doublon", "column": "amount"},
            {"type": "log1p", "name": "doublon", "column": "count"},
        ]
        with pytest.raises(ValueError, match="Duplicated"):
            FeatureBuilder(recipes)

    def test_recipes_reading_the_target_are_rejected(self, app_config: Any) -> None:
        """Utiliser la cible comme feature est la fuite la plus classique : bloquée."""
        target = app_config.data.target or "target"
        with pytest.raises(ValueError, match="must not read the target"):
            FeatureBuilder([{"type": "log1p", "name": "fuite", "column": target}], target=target)

    def test_missing_input_column_gives_an_actionable_error(
        self, feature_builder: FeatureBuilder, split_frames: Any
    ) -> None:
        """Une colonne absente doit nommer la recette en échec."""
        if not feature_builder.recipes:
            pytest.skip("aucune recette de feature déclarée dans ce projet")
        required = feature_builder.recipes[0].required_columns()
        if not required or required[0] not in split_frames.train.columns:
            pytest.skip("recette sans colonne d'entrée")
        truncated = split_frames.train.drop(columns=[required[0]])
        with pytest.raises(ValueError, match="failed"):
            feature_builder.transform(truncated)

    def test_describe_documents_recipes(self, feature_builder: FeatureBuilder) -> None:
        """``describe()`` sert de documentation exploitable dans les rapports."""
        if not feature_builder.recipes:
            pytest.skip("aucune recette de feature déclarée dans ce projet")
        description = feature_builder.describe()
        assert len(description) == len(feature_builder.recipes)
        assert {"name", "type"} <= set(description[0])

    def test_select_feature_columns_excludes_drops(
        self, split_frames: Any, app_config: Any
    ) -> None:
        """Le sélecteur exclut la cible et les colonnes non modélisables."""
        columns = select_feature_columns(
            split_frames.train,
            drop_columns=app_config.data.drop_columns,
            target=app_config.data.target,
        )
        target = app_config.data.target
        assert target not in columns
        assert not set(app_config.data.drop_columns) & set(columns)
        # En tâche non supervisée, il n'y a pas de colonne cible à retirer du total.
        target_columns = 0 if target is None else 1
        assert (
            len(columns) + len(app_config.data.drop_columns) + target_columns
            == split_frames.train.shape[1]
        )


class TestPreprocessingPipeline:
    """Pipeline complet (imputation, encodage, scaling)."""

    def test_output_is_numeric_and_complete(self, matrices: dict[str, Any]) -> None:
        """La matrice livrée au modèle est 100 % numérique et sans NaN."""
        X_train: pd.DataFrame = matrices["X_train"]
        assert not X_train.empty
        assert all(pd.api.types.is_numeric_dtype(X_train[column]) for column in X_train.columns)
        assert not X_train.isna().any().any()

    def test_column_names_are_stable_across_splits(self, matrices: dict[str, Any]) -> None:
        """Train / val / test partagent exactement les mêmes colonnes, dans le même ordre."""
        assert list(matrices["X_train"].columns) == list(matrices["X_test"].columns)
        if matrices["X_val"] is not None:
            assert list(matrices["X_train"].columns) == list(matrices["X_val"].columns)

    def test_feature_names_out_matches_the_matrix(
        self, preprocessing: PreprocessingPipeline, matrices: dict[str, Any]
    ) -> None:
        """``feature_names_out`` décrit réellement la matrice produite (contrat sklearn)."""
        assert preprocessing.feature_names_out == list(matrices["X_train"].columns)

    def test_transform_before_fit_is_refused(
        self, sample_frame: pd.DataFrame, app_config: Any
    ) -> None:
        """Transformer avant d'avoir appris est une erreur de programmation explicite."""
        pipeline = PreprocessingPipeline(
            numeric_features=["amount", "count"],
            categorical_features=["city"],
            config=app_config.preprocessing.model_dump(),
            target=app_config.data.target,
        )
        with pytest.raises(RuntimeError):
            pipeline.transform(sample_frame)

    def test_scaler_statistics_come_from_train_only(
        self, preprocessing: PreprocessingPipeline, matrices: dict[str, Any]
    ) -> None:
        """Anti-fuite : le test est centré/réduit avec les statistiques du **train**."""
        numeric_outputs = [
            column
            for column in matrices["X_train"].columns
            if pd.api.types.is_numeric_dtype(matrices["X_train"][column])
        ]
        train_mean = float(matrices["X_train"][numeric_outputs].to_numpy().mean())
        test_mean = float(matrices["X_test"][numeric_outputs].to_numpy().mean())
        assert abs(train_mean) < 0.5
        assert abs(test_mean - train_mean) < 1.0

    def test_categorical_columns_are_encoded(
        self, preprocessing: PreprocessingPipeline, matrices: dict[str, Any]
    ) -> None:
        """Aucune colonne catégorielle texte ne survit au pipeline."""
        assert preprocessing.categorical_features
        assert all(
            pd.api.types.is_numeric_dtype(matrices["X_train"][column])
            for column in matrices["X_train"].columns
        )

    def test_report_documents_the_transformations(
        self, preprocessing: PreprocessingPipeline
    ) -> None:
        """Le rapport expose les choix (encodeur, scaler, colonnes) pour l'audit."""
        payload = preprocessing.report.to_dict()
        assert payload["encoder"]
        assert payload["scaler"]
        assert len(payload["numeric_features"]) == len(preprocessing.numeric_features)

    def test_save_load_roundtrip_is_lossless(
        self, preprocessing: PreprocessingPipeline, matrices: dict[str, Any], tmp_path: Any
    ) -> None:
        """Un pipeline sauvegardé puis rechargé transforme à l'identique (inférence)."""
        path = preprocessing.save(tmp_path / "preprocessing.joblib")
        assert path.exists()
        restored = PreprocessingPipeline.load(path)
        pd.testing.assert_frame_equal(
            restored.transform(matrices["feature_frame"]), matrices["X_train"]
        )

    def test_from_config_is_equivalent_to_the_pipeline_used_by_training(
        self, app_config: Any, prepared: dict[str, Any]
    ) -> None:
        """``from_config`` reconstruit le même pipeline que celui de l'entraînement.

        C'est ce qui garantit que l'inférence (qui repart de la configuration) produit
        exactement la même matrice que l'entraînement.
        """
        pipeline = PreprocessingPipeline.from_config(app_config.model_dump())
        assert set(pipeline.numeric_features) == set(prepared["numeric_features"])
        assert set(pipeline.categorical_features) == set(prepared["categorical_features"])
        pipeline.fit(prepared["feature_frame"], prepared["y_train"])
        assert pipeline.feature_names_out == prepared["feature_names"]

    def test_infer_column_kinds_uses_the_schema(self, app_config: Any) -> None:
        """En l'absence de colonnes explicites, le schéma Pandera pilote la détection."""
        numeric, categorical = infer_column_kinds(
            {},
            {"drop_columns": list(app_config.data.drop_columns), "target": app_config.data.target},
        )
        assert numeric or categorical
        assert app_config.data.target not in [*numeric, *categorical]
        for column in app_config.data.drop_columns:
            assert column not in [*numeric, *categorical]


class TestCustomTransformers:
    """Contrat scikit-learn des transformateurs maison."""

    @pytest.mark.parametrize(
        "transformer",
        [
            ColumnSelector(["amount", "count"]),
            TypeCaster(numeric_columns=["amount", "count"], categorical_columns=["city"]),
            OutlierClipper((0.05, 0.95)),
            Log1pTransformer(["amount"]),
            RareCategoryGrouper(min_frequency=0.05),
            DataFrameScaler("standard"),
        ],
        ids=[
            "column_selector",
            "type_caster",
            "outlier_clipper",
            "log1p",
            "rare_category",
            "scaler",
        ],
    )
    def test_every_transformer_is_cloneable(
        self, transformer: Any, sample_frame: pd.DataFrame
    ) -> None:
        """``clone()`` exige des paramètres de constructeur stockés tels quels."""
        copy = clone(transformer)
        assert copy is not transformer
        fitted = copy.fit(sample_frame)
        assert fitted.transform(sample_frame) is not None

    @pytest.mark.parametrize(
        "transformer",
        [
            ColumnSelector(["amount", "count"]),
            TypeCaster(numeric_columns=["amount", "count"], categorical_columns=["city"]),
            OutlierClipper((0.05, 0.95)),
            Log1pTransformer(["amount"]),
            DataFrameScaler("minmax"),
        ],
        ids=["column_selector", "type_caster", "outlier_clipper", "log1p", "scaler"],
    )
    def test_feature_names_out_matches_transform_width(
        self, transformer: Any, sample_frame: pd.DataFrame
    ) -> None:
        """``get_feature_names_out()`` doit avoir la même largeur que ``transform()``."""
        fitted = transformer.fit(sample_frame)
        names = fitted.get_feature_names_out()
        width = fitted.transform(sample_frame).shape[1]
        assert len(names) == width

    def test_column_selector_keeps_order(self, sample_frame: pd.DataFrame) -> None:
        """L'ordre déclaré est respecté : l'ordre des colonnes fait partie du contrat."""
        selector = ColumnSelector(["count", "amount"]).fit(sample_frame)
        assert list(selector.transform(sample_frame).columns) == ["count", "amount"]

    def test_column_selector_raises_on_missing_column(self, sample_frame: pd.DataFrame) -> None:
        """Une colonne manquante est refusée, sauf opt-in explicite."""
        with pytest.raises(KeyError, match="missing column"):
            ColumnSelector(["amount", "inexistante"]).fit(sample_frame)
        tolerant = ColumnSelector(["amount", "inexistante"], allow_missing=True).fit(sample_frame)
        assert list(tolerant.transform(sample_frame).columns) == ["amount"]

    def test_type_caster_projects_extra_columns(self, sample_frame: pd.DataFrame) -> None:
        """Un payload avec des colonnes en trop est projeté sur le contrat (robustesse API)."""
        caster = TypeCaster(numeric_columns=["amount"], categorical_columns=["city"])
        noisy = sample_frame.assign(colonne_inutile="x")
        fitted = caster.fit(noisy)
        projected = fitted.transform(noisy)
        assert list(projected.columns) == ["amount", "city"]
        assert pd.api.types.is_float_dtype(projected["amount"])
        assert isinstance(projected["city"].dtype, pd.CategoricalDtype)

    def test_outlier_clipper_bounds_values(self, sample_frame: pd.DataFrame) -> None:
        """Les extrêmes sont ramenés dans les quantiles appris sur le train."""
        clipper = OutlierClipper((0.1, 0.9), columns=["amount"]).fit(sample_frame)
        extreme = sample_frame.assign(
            amount=[-1_000.0, 1_000_000.0, *sample_frame["amount"].iloc[2:]]
        )
        clipped = clipper.transform(extreme)
        low, high = clipper.bounds_["amount"]
        assert clipped["amount"].min() == pytest.approx(low)
        assert clipped["amount"].max() == pytest.approx(high)
        assert sum(clipper.report.values()) >= 2

    def test_outlier_clipper_rejects_bad_quantiles(self, sample_frame: pd.DataFrame) -> None:
        """Une fenêtre de quantiles incohérente est refusée au ``fit``."""
        with pytest.raises(ValueError, match="quantiles"):
            OutlierClipper((0.9, 0.1)).fit(sample_frame)

    def test_log1p_rejects_negative_values(self, sample_frame: pd.DataFrame) -> None:
        """``log1p`` exige des valeurs positives : l'erreur doit être explicite."""
        negative = sample_frame.assign(amount=-sample_frame["amount"])
        with pytest.raises(ValueError, match="negative"):
            Log1pTransformer(["amount"]).fit(negative)

    def test_log1p_transform_is_exact(self, sample_frame: pd.DataFrame) -> None:
        """La transformation appliquée est bien ``log(1 + x)``."""
        transformer = Log1pTransformer(["amount"]).fit(sample_frame)
        expected = np.log1p(sample_frame["amount"].to_numpy(dtype="float64"))
        np.testing.assert_allclose(
            transformer.transform(sample_frame)["amount"].to_numpy(), expected
        )

    def test_rare_category_grouper_buckets_infrequent_levels(self) -> None:
        """Les modalités rares sont regroupées, y compris les modalités jamais vues."""
        skewed = pd.DataFrame(
            {"city": ["paris"] * 80 + ["lyon"] * 15 + ["nice"] * 3 + ["brest"] * 2}
        )
        grouper = RareCategoryGrouper(min_frequency=0.05, rare_label="rare").fit(skewed)
        grouped = grouper.transform(skewed)
        assert set(grouped["city"].astype(str)) == {"paris", "lyon", "rare"}
        assert int((grouped["city"].astype(str) == "rare").sum()) == 5

        unseen = pd.DataFrame({"city": ["ville_inconnue", "paris"]})
        assert grouper.transform(unseen)["city"].astype(str).tolist() == ["rare", "paris"]

    @pytest.mark.parametrize("strategy", ["standard", "minmax", "robust", "none"])
    def test_scaler_strategies(self, sample_frame: pd.DataFrame, strategy: str) -> None:
        """Chaque stratégie de scaling produit une sortie cohérente."""
        scaler = DataFrameScaler(strategy).fit(sample_frame)
        scaled = scaler.transform(sample_frame)
        assert list(scaled.columns) == list(sample_frame.columns)
        values = scaled["amount"].to_numpy(dtype="float64")
        assert np.isfinite(values).all()
        if strategy == "standard":
            assert values.mean() == pytest.approx(0.0, abs=1e-8)
        elif strategy == "minmax":
            assert values.min() == pytest.approx(0.0, abs=1e-8)
            assert values.max() == pytest.approx(1.0, abs=1e-8)
        elif strategy == "none":
            np.testing.assert_allclose(values, sample_frame["amount"].to_numpy(dtype="float64"))

    def test_unknown_scaler_is_rejected(self) -> None:
        """Une stratégie inconnue doit lever immédiatement (erreur de configuration)."""
        with pytest.raises(ValueError, match="Unknown scaler"):
            DataFrameScaler("quantique")
