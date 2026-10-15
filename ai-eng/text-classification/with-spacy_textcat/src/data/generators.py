"""Générateur synthétique du corpus de classification de texte.

Un corpus synthétique de 1 200 tickets répartis sur six catégories de support — facturation,
livraison, produit défectueux, remboursement, compte client et « autre » — et trois styles
rédactionnels : cinquante pour cent des tickets réutilisent le vocabulaire de leur catégorie,
trente pour cent reformulent la même demande sans ce vocabulaire, vingt pour cent ajoutent une
formule de politesse et une signature partagées par toutes les classes. Chaque catégorie a son
vocabulaire propre, sauf « autre » : cette classe n'a aucun mot commun par construction, elle est
minoritaire (12 %) et c'est pour elle que la métrique principale est une F1 macro. La priorité
déclarée par le client est tirée indépendamment du libellé, et le canal d'arrivée n'est pas non
plus un signal : ce sont deux distracteurs déclarés, mesurés par un test de raccourci. Enfin,
chaque pool de formulations est coupé en deux : quelques tournures par couple (classe, style) sont
**réservées aux splits d'évaluation** et n'apparaissent jamais à l'entraînement, sans quoi le
score publié mesurerait une recopie de gabarits plutôt qu'une généralisation.

Le corpus est écrit à partir de **gabarits par classe**, et c'est ce qui rend la tâche honnête :

* chaque classe possède un vocabulaire propre (``facture``, ``prélèvement`` pour la facturation ;
  ``colis``, ``livraison`` pour la livraison), et un modèle lexical *doit* le retrouver — c'est le
  segment ``canonique``, où la précision est facile ;
* un tiers des tickets est écrit en **paraphrase** : la même demande, sans les mots de la classe.
  Ces tickets existent pour que le projet publie l'écart entre « lire un mot » et « comprendre une
  demande », au lieu de le supposer ;
* les formulations du test **ne sont jamais vues à l'entraînement** : chaque pool (classe, style)
  est coupé en deux, et ses dernières formulations sont réservées aux splits ``val``,
  ``calibration`` et ``test``. Sans cette réserve, le corpus se contenterait de vérifier qu'un
  modèle sait recopier des phrases qu'il a mémorisées — le score serait parfait et n'apprendrait
  rien. Ici, la F1 publiée mesure ce qui est *généralisé* ;
* une part des tickets est **bruitée** (formule de politesse, signature, doublon de phrase) : le
  bruit est le même pour toutes les classes, donc il ne peut pas être un signal — un modèle qui le
  suit se trompe, et la ventilation par style le montre ;
* la classe ``autre`` n'a **aucun vocabulaire commun** : c'est la classe difficile du corpus, et
  c'est pour elle que la métrique principale est une **F1 macro** (une exactitude globale
  récompenserait un modèle qui l'ignore).

Deux colonnes sont des distracteurs *déclarés* : ``priority`` (priorité déclarée par le client,
indépendante de la catégorie par construction) et ``source`` (canal d'arrivée). La suite de tests
vérifie que ``priority`` est indépendante du libellé ; la ventilation par ``source``, elle, est une
vraie lecture métier.

Le découpage (``train`` / ``val`` / ``calibration`` / ``test``) est **stratifié par classe** et
écrit dans le corpus : deux exécutions à graine fixée produisent les mêmes lignes, et le test ne
sert qu'une fois. Tout est semé : la même configuration donne des tables identiques au bit près.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.data.generator_base import BaseCorpusGenerator, GeneratedCorpus
from src.preprocessing.transformers import normalise_text, tokenize
from src.utils.config_access import as_mapping
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Splits and their target shares. ``calibration`` is small on purpose: it settles the class
#: weights and the confidence threshold of the notebook, not the model itself.
SPLIT_SHARES: dict[str, float] = {
    "train": 0.60,
    "val": 0.20,
    "calibration": 0.05,
    "test": 0.15,
}

#: Share of the tickets written in each editorial style (the class shares are separate).
STYLE_SHARES: dict[str, float] = {"canonique": 0.50, "paraphrase": 0.30, "bruite": 0.20}

#: Number of phrasings per (class, style) pool that the training split never sees. They are used
#: by ``val``, ``calibration`` and ``test`` only: c'est ce qui sépare « mémoriser » de « généraliser ».
PHRASING_HOLDOUT: int = 2

#: Relative weight of every class in the corpus: ``autre`` is genuinely minority.
CLASS_WEIGHTS: dict[str, float] = {
    "facturation": 0.20,
    "livraison": 0.20,
    "produit_defectueux": 0.18,
    "remboursement": 0.15,
    "compte_client": 0.15,
    "autre": 0.12,
}

#: Channels and their share. A channel is a *feature*, never a label: the chat produces shorter
#: tickets than the letter, but any channel can carry any category.
SOURCE_SHARES: dict[str, float] = {
    "formulaire": 0.45,
    "email": 0.30,
    "chat": 0.15,
    "courrier": 0.10,
}

#: Declared distractor: the priority is drawn independently of the label.
PRIORITY_SHARES: dict[str, float] = {"basse": 0.35, "normale": 0.45, "haute": 0.20}

#: Shared noise, identical for every class: greetings and signatures add length without information.
NOISE_PREFIXES: tuple[str, ...] = (
    "Bonjour, je vous contacte aujourd'hui car",
    "Bonsoir, désolé de vous déranger mais",
    "Bonjour à l'équipe, petite question :",
    "Rebonjour, je reviens vers vous :",
)

NOISE_SUFFIXES: tuple[str, ...] = (
    "Merci d'avance pour votre réponse.",
    "Je vous remercie par avance.",
    "Bonne journée à vous.",
    "Cordialement, Camille.",
)

#: Products and references the tickets quote, so the texts are not interchangeable.
PRODUCTS: tuple[str, ...] = (
    "casque Aria",
    "enceinte Onda",
    "montre Belis",
    "liseuse Silex",
    "caméra Vela",
    "routeur Nemo",
    "clavier Quartz",
    "écouteurs Iode",
)


@dataclass(frozen=True, slots=True)
class ClassTemplate:
    """Gabarits d'une catégorie : son vocabulaire, ses formulations canoniques, ses paraphrases.

    Attributes:
        label: Category of the support desk.
        display_name: French name used in the logs and the report.
        keywords: Terms that signal the category. They feed the rule-based baseline published in
            the metadata, and the generator checks that a *paraphrase* avoids them.
        canonical: Phrasings that reuse the keywords (the easy segment).
        paraphrase: Phrasings that ask the same thing without the keywords (the hard segment).
    """

    label: str
    display_name: str
    keywords: tuple[str, ...]
    canonical: tuple[str, ...]
    paraphrase: tuple[str, ...]


#: The six categories of the support desk, with their vocabulary.
TEMPLATES: tuple[ClassTemplate, ...] = (
    ClassTemplate(
        label="facturation",
        display_name="Facturation",
        keywords=("facture", "prélèvement", "montant", "facturé"),
        canonical=(
            "Le montant de la facture {order} ne correspond pas au prix annoncé.",
            "Un prélèvement a été effectué deux fois sur la facture {order}.",
            "Je n'ai jamais reçu la facture pour {product} commandé sous la référence {order}.",
            "La facture {order} fait apparaître des frais que je n'ai pas commandés.",
            "La facture {order} a été réglée deux fois par prélèvement.",
            "Le montant facturé pour {product} dépasse le devis signé ({order}).",
        ),
        paraphrase=(
            "On m'a débité plus que le prix affiché au moment de l'achat, commande {order}.",
            "Le total réglé pour {product} ne correspond pas à ce qui était convenu ({order}).",
            "Le document joint à ma commande {order} affiche un total qui m'étonne.",
            "Je constate un second encaissement sur ma carte pour {product}, dossier {order}.",
            "Un second débit apparaît sur mon relevé, alors que je n'ai commandé qu'une fois "
            "({order}).",
            "Le prix débité sur ma carte ne correspond pas à celui affiché pour {product} "
            "({order}).",
        ),
    ),
    ClassTemplate(
        label="livraison",
        display_name="Livraison",
        keywords=("colis", "livraison", "livré", "transporteur"),
        canonical=(
            "Mon colis {order} n'est toujours pas livré après deux semaines.",
            "La livraison de {product} a été annoncée pour lundi et rien n'est arrivé.",
            "Le transporteur indique mon colis comme livré, je ne l'ai pas reçu ({order}).",
            "La livraison a été repoussée trois fois sans explication pour {order}.",
            "Mon colis {order} est bloqué chez le transporteur depuis une semaine.",
            "La livraison de {product} est annoncée pour demain, sans aucune confirmation ({order}).",
        ),
        paraphrase=(
            "Cela fait quinze jours que j'attends mon achat {order} et personne ne me répond.",
            "Le suivi affiche « remis au destinataire » alors que je n'ai rien reçu ({order}).",
            "Je n'ai jamais rien vu arriver pour {product}, dossier {order}.",
            "Trois rendez-vous ont été décalés, et mon achat {order} reste introuvable.",
            "Toujours rien dans ma boîte aux lettres après trois semaines d'attente ({order}).",
            "Le suivi annonce une remise au destinataire, mais personne n'a rien apporté ({order}).",
        ),
    ),
    ClassTemplate(
        label="produit_defectueux",
        display_name="Produit défectueux",
        keywords=("défectueux", "panne", "cassé", "ne fonctionne"),
        canonical=(
            "Le {product} reçu est défectueux : la batterie ne tient pas une heure.",
            "Mon {product} est en panne après trois semaines d'utilisation.",
            "Le {product} est arrivé cassé, l'écran est fissuré ({order}).",
            "Le {product} ne fonctionne plus du tout depuis la mise à jour ({order}).",
            "Mon {product} est tombé en panne le jour même de la réception ({order}).",
            "Le {product} est cassé, et la panne revient à chaque allumage ({order}).",
        ),
        paraphrase=(
            "Au bout de trois semaines, mon achat refuse de s'allumer ({order}).",
            "L'appareil s'éteint tout seul dès que je le débranche, achat {order}.",
            "Il y a une fissure visible sur l'écran à la réception, commande {order}.",
            "Depuis vendredi plus rien ne s'allume, malgré le produit {product} ({order}).",
            "L'écran reste noir et l'appareil s'éteint dès que je le débranche ({order}).",
            "La batterie ne tient plus la journée et la coque présente une fissure, achat {order}.",
        ),
    ),
    ClassTemplate(
        label="remboursement",
        display_name="Remboursement",
        keywords=("remboursement", "rembourser", "remboursé", "somme"),
        canonical=(
            "Je demande le remboursement de {product} retourné la semaine dernière ({order}).",
            "Le remboursement annoncé n'est jamais arrivé sur mon compte ({order}).",
            "J'attends d'être remboursé de la totalité de la somme pour {order}.",
            "Le remboursement de {order} est incomplet par rapport au montant payé.",
            "Je n'ai toujours pas été remboursé du montant de {product} ({order}).",
            "Le remboursement de la somme versée pour {order} reste incomplet.",
        ),
        paraphrase=(
            "L'appareil est reparti chez vous mais l'argent n'est jamais revenu ({order}).",
            "Je souhaite récupérer les euros versés pour {product}, dossier {order}.",
            "On m'a promis un virement pour le retour de {order}, toujours rien.",
            "La restitution de ce que j'ai payé reste bloquée depuis un mois ({order}).",
            "L'argent versé pour {product} ne m'a jamais été restitué ({order}).",
            "Le virement promis après le retour de l'appareil n'arrive toujours pas ({order}).",
        ),
    ),
    ClassTemplate(
        label="compte_client",
        display_name="Compte client",
        keywords=("compte", "mot de passe", "identifiant", "connexion"),
        canonical=(
            "Je n'arrive plus à me connecter à mon compte, mon mot de passe est refusé.",
            "Mon identifiant ne fonctionne plus depuis la dernière mise à jour du site.",
            "Je souhaite supprimer mon compte client et mes données personnelles.",
            "Deux comptes portent mon adresse e-mail, je voudrais les fusionner.",
            "Mon mot de passe est refusé et je ne peux plus accéder à mon compte.",
            "Je demande la fermeture de mon compte client et la suppression des données ({order}).",
        ),
        paraphrase=(
            "Impossible d'accéder à mon espace depuis hier soir, l'écran me renvoie une erreur.",
            "Le site refuse mes accès alors que je saisis bien les mêmes informations.",
            "Je voudrais faire disparaître mon espace et les informations qu'il contient.",
            "Deux espaces existent à mon nom et j'aimerais n'en garder qu'un.",
            "Le site me renvoie une erreur dès que je saisis mes informations ({order}).",
            "Impossible d'ouvrir mon espace : le site refuse mes accès ({order}).",
        ),
    ),
    ClassTemplate(
        label="autre",
        display_name="Autre",
        keywords=(),
        canonical=(
            "Avez-vous des accessoires compatibles avec le {product} ?",
            "Proposez-vous des formations pour utiliser le {product} ?",
            "Pouvez-vous me dire où trouver la notice du {product} en français ?",
            "Je cherche un revendeur proche de chez moi pour le {product}.",
            "Quels accessoires sont vendus avec le {product} ?",
            "Existe-t-il une formation en ligne pour le {product} ({order}) ?",
        ),
        paraphrase=(
            "Y a-t-il des accessoires qui vont avec l'appareil acheté sous {order} ?",
            "Existe-t-il des sessions d'initiation pour ce que j'ai acheté ({order}) ?",
            "Où puis-je lire les explications en français de mon achat {order} ?",
            "Quel magasin peut me montrer ce produit près de chez moi ?",
            "Quels accessoires me conseillez-vous pour l'appareil reçu la semaine dernière ({order}) ?",
            "Peut-on me montrer le fonctionnement de mon achat dans une boutique proche ({order}) ?",
        ),
    ),
)


@dataclass(slots=True)
class SyntheticTicketGenerator(BaseCorpusGenerator):
    """Generate the labelled support-ticket corpus, its splits and its metadata.

    Attributes:
        dataset_name: Name used in logs and artefacts.
        n_documents: Target number of tickets.
        seed: Reproducibility seed.
        reference_date: Date of the last ticket (dates are drawn backwards).
        templates: Class templates.
        style_shares: Share of each editorial style.
        split_shares: Share of each split (stratified per class and per style).
        phrasing_holdout: Number of phrasings kept out of the training split, per pool.
        random_state: Seed generator, created in :meth:`__post_init__`.
    """

    dataset_name: str = "support_tickets"
    n_documents: int = 1200
    seed: int = 42
    reference_date: date = date(2025, 3, 3)
    templates: tuple[ClassTemplate, ...] = field(default=TEMPLATES, repr=False)
    style_shares: dict[str, float] = field(default_factory=lambda: dict(STYLE_SHARES), repr=False)
    split_shares: dict[str, float] = field(default_factory=lambda: dict(SPLIT_SHARES), repr=False)
    phrasing_holdout: int = PHRASING_HOLDOUT
    random_state: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Seed the generator and validate the shares."""
        self.random_state = np.random.default_rng(self.seed)
        for name, shares in (("style", self.style_shares), ("split", self.split_shares)):
            total = float(sum(shares.values()))
            if abs(total - 1.0) > 1e-6:
                msg = f"{name} shares must sum to 1, got {total:.4f} ({shares})"
                raise ValueError(msg)
        smallest = min(
            len(pool)
            for template in self.templates
            for pool in (template.canonical, template.paraphrase)
        )
        if not 1 <= self.phrasing_holdout <= smallest - 2:
            msg = (
                "phrasing_holdout must leave at least two phrasings for the training split and "
                f"one for the evaluation splits, got {self.phrasing_holdout} for pools of "
                f"{smallest}"
            )
            raise ValueError(msg)
        if self.n_documents < len(self.templates) * 4:
            msg = (
                f"n_documents must be at least {len(self.templates) * 4} to give every class a "
                f"handful of training rows, got {self.n_documents}"
            )
            raise ValueError(msg)

    @classmethod
    def from_config(cls, config: Any, *, seed: int | None = None) -> SyntheticTicketGenerator:
        """Build the generator from the ``data`` node of the configuration.

        Args:
            config: ``data`` configuration node (``seed``, ``n_samples`` and the ``corpus``
                sub-node are read).
            seed: Optional seed override.

        Returns:
            The configured generator.
        """
        settings = as_mapping(config)
        corpus = as_mapping(settings.get("corpus"))
        return cls(
            dataset_name=str(settings.get("dataset_name", "support_tickets")),
            n_documents=int(settings.get("n_samples", 1200)),
            seed=int(seed if seed is not None else settings.get("seed", 42)),
            reference_date=_parse_date(corpus.get("reference_date")) or date(2025, 3, 3),
            style_shares=dict(corpus.get("style_shares") or STYLE_SHARES),
            split_shares=dict(corpus.get("split_shares") or SPLIT_SHARES),
            phrasing_holdout=int(corpus.get("phrasing_holdout", PHRASING_HOLDOUT)),
        )

    # ------------------------------------------------------------------ génération --------
    def generate(self) -> GeneratedCorpus:
        """Generate the corpus, its splits and the metadata.

        Returns:
            The :class:`GeneratedCorpus` (``queries`` is empty: a classification corpus is one
            table, the labels live on the documents).
        """
        documents = self._documents()
        metadata = self._metadata(documents)
        logger.info(
            "Corpus '{}' generated | {} tickets | {} classes | par classe {}",
            self.dataset_name,
            len(documents),
            documents["label"].nunique(),
            documents["label"].value_counts().to_dict(),
        )
        return GeneratedCorpus(
            documents=documents,
            queries=pd.DataFrame(),
            metadata=metadata,
        )

    def _documents(self) -> pd.DataFrame:
        """Build the labelled corpus, class by class, style by style.

        Le split est tiré **avant** la formulation : c'est lui qui décide si la ligne a le droit de
        piocher dans les formulations vues à l'entraînement ou dans celles qui lui sont réservées.
        """
        counts = self._class_counts()
        rows: list[dict[str, Any]] = []
        splits: list[str] = []
        number = 1
        for template in self.templates:
            styles = [self._draw(self.style_shares) for _ in range(int(counts[template.label]))]
            assigned = self._split_styles(styles)
            for position, (style, split) in enumerate(zip(styles, assigned, strict=True)):
                rows.append(self._row(number, template, style, split, position))
                splits.append(split)
                number += 1
        frame = pd.DataFrame(rows)
        frame["text"] = frame["text"].astype(str)
        frame["n_tokens"] = frame["text"].map(lambda text: len(tokenize(text))).astype("int64")
        frame["split"] = splits
        return frame

    def _class_counts(self) -> dict[str, int]:
        """Distribute the target number of tickets across the classes."""
        weights = np.asarray([CLASS_WEIGHTS[template.label] for template in self.templates])
        weights = weights / weights.sum()
        raw = weights * float(self.n_documents)
        counts = np.floor(raw).astype(int)
        remainder = int(self.n_documents - int(counts.sum()))
        for position in np.argsort(-(raw - counts))[:remainder]:
            counts[int(position)] += 1
        return {
            template.label: int(count)
            for template, count in zip(self.templates, counts, strict=True)
        }

    def _row(
        self, number: int, template: ClassTemplate, style: str, split: str, position: int
    ) -> dict[str, Any]:
        """Build one ticket row.

        Args:
            number: Row number, used to number ``doc_id``.
            template: Class templates the row is drawn from.
            style: Editorial style of the row.
            split: Split the row belongs to; it selects the pool of phrasings the row may use.
            position: Position of the row inside its class (decides the casing of noisy tickets).

        Returns:
            One row of the corpus (``n_tokens`` and ``split`` are added by the caller).
        """
        product = str(self.random_state.choice(PRODUCTS))
        order = f"CMD-{int(self.random_state.integers(10_000, 99_999))}"
        pool = template.canonical if style == "canonique" else template.paraphrase
        learned = pool[: len(pool) - self.phrasing_holdout]
        # Les splits d'évaluation ne reçoivent que des formulations inédites : c'est la règle qui
        # empêche le corpus de publier un score de mémorisation déguisé en généralisation.
        available = learned if split == "train" else pool[len(learned) :]
        template_text = str(available[int(self.random_state.integers(len(available)))])
        slots = {"product": product, "order": order}
        text = template_text.format(**slots)
        if style == "bruite":
            prefix = str(self.random_state.choice(NOISE_PREFIXES))
            suffix = str(self.random_state.choice(NOISE_SUFFIXES))
            text = f"{prefix} {text.lower() if position % 2 else text} {suffix}"
        published = self.reference_date - timedelta(days=int(self.random_state.integers(0, 180)))
        return {
            "doc_id": f"TKT-{number:04d}",
            "text": text,
            "label": template.label,
            "source": self._draw(SOURCE_SHARES),
            "style": style,
            "priority": self._draw(PRIORITY_SHARES),
            "published_at": published,
        }

    def _split_styles(self, styles: Sequence[str]) -> list[str]:
        """Assign one split per row, stratified per editorial style.

        Le découpage est écrit dans le corpus plutôt que tiré à l'entraînement : deux exécutions
        voient donc exactement les mêmes lignes de test, et un ``make all`` rejoue le même chiffre.

        Chaque style reçoit **ses propres** effectifs par split (méthode du plus fort reste) avant
        d'être mélangé : une répartition par blocs globaux laisserait un style entier hors du split
        d'entraînement, et la ventilation par style du rapport comparerait alors des segments qui
        n'ont pas été appris dans les mêmes conditions.

        Args:
            styles: Editorial style of every row of one class.

        Returns:
            One split name per row, in the order given.
        """
        names = list(self.split_shares)
        assignment = np.empty(len(styles), dtype=object)
        for style in dict.fromkeys(styles):
            positions = np.asarray(
                [index for index, value in enumerate(styles) if value == style], dtype="int64"
            )
            shuffled = self.random_state.permutation(positions)
            sizes = _largest_remainder(len(positions), self.split_shares)
            start = 0
            for name in names:
                size = int(sizes[name])
                assignment[shuffled[start : start + size]] = name
                start += size
        return [str(value) for value in assignment]

    def _metadata(self, documents: pd.DataFrame) -> dict[str, Any]:
        """Describe the generated corpus, references included."""
        distribution = documents["label"].value_counts()
        majority_label = str(distribution.index[0])
        rule_accuracy = self._rule_baseline(documents)
        return {
            "dataset_name": self.dataset_name,
            "seed": int(self.seed),
            "n_documents": int(len(documents)),
            "n_classes": int(distribution.size),
            "majority_label": majority_label,
            "majority_share": float(distribution.iloc[0]) / float(len(documents)),
            "class_distribution": {str(key): int(value) for key, value in distribution.items()},
            "share_canonique": float((documents["style"] == "canonique").mean()),
            "share_paraphrase": float((documents["style"] == "paraphrase").mean()),
            "share_bruite": float((documents["style"] == "bruite").mean()),
            "phrasing_holdout": int(self.phrasing_holdout),
            # Les splits d'évaluation n'utilisent que des formulations inédites : le lire ici évite
            # de confondre le score du modèle avec celui d'un recopieur de phrases.
            "phrasing_holdout_definition": (
                "formulations par classe et par style réservées aux splits val, calibration et "
                "test : le modèle ne les voit jamais à l'entraînement"
            ),
            "n_phrasings": {
                "canonique": [len(template.canonical) for template in self.templates],
                "paraphrase": [len(template.paraphrase) for template in self.templates],
            },
            # La référence « règles maison » : compter les mots-clés de chaque classe suffit-il ?
            # Le chiffre est mesuré ici, publié avec le corpus, et le rapport le reprend.
            "rule_baseline_accuracy": rule_accuracy,
            "rule_baseline_definition": (
                "classe dont les mots-clés sont les plus fréquents dans le texte ; "
                "égalités tranchées par l'ordre alphabétique"
            ),
            "split_sizes": {
                str(key): int(value)
                for key, value in documents["split"].value_counts().items()
            },
            "mean_tokens": float(documents["n_tokens"].mean()),
        }

    def _rule_baseline(self, documents: pd.DataFrame) -> float:
        """Accuracy of the keyword rule the project publishes as a reference.

        Args:
            documents: Labelled corpus.

        Returns:
            The share of tickets the rule classifies correctly. A model that does not beat this
            number has learned nothing that a support manager could not write down in ten lines.
        """
        keywords = {
            template.label: template.keywords
            for template in self.templates
            if template.keywords
        }
        correct = 0
        for record in documents.to_dict(orient="records"):
            # La règle cherche les mots-clés comme des sous-chaînes du texte normalisé : elle
            # attrape « factures » comme « facture », exactement ce qu'écrirait un gestionnaire de
            # support dans dix lignes — et c'est la référence que le modèle doit battre.
            haystack = normalise_text(str(record["text"])).lower()
            scores = {
                label: sum(1 for term in terms if term in haystack)
                for label, terms in keywords.items()
            }
            best = max(scores.values(), default=0)
            if best == 0:
                predicted = "autre"
            else:
                predicted = sorted(label for label, score in scores.items() if score == best)[0]
            correct += int(predicted == str(record["label"]))
        return float(correct) / float(len(documents))

    def _draw(self, shares: Mapping[str, float]) -> str:
        """Draw one key from a weighted mapping, deterministically."""
        names = list(shares)
        weights = np.asarray([shares[name] for name in names], dtype="float64")
        return str(names[int(self.random_state.choice(len(names), p=weights / weights.sum()))])


def keyword_baseline_accuracy(documents: pd.DataFrame) -> float:
    """Mesure la règle de mots-clés de la famille sur un corpus quelconque.

    Args:
        documents: Labelled corpus.

    Returns:
        The accuracy of the rule (see :meth:`SyntheticTicketGenerator._rule_baseline`).
    """
    return SyntheticTicketGenerator()._rule_baseline(documents)  # noqa: SLF001 - mesure publique


def _largest_remainder(total: int, shares: Mapping[str, float]) -> dict[str, int]:
    """Split an integer count according to shares, using the largest remainder method.

    Args:
        total: Number of rows to distribute.
        shares: Relative shares, summing to one.

    Returns:
        One integer per share key, summing exactly to ``total``. A share too small to claim a
        single row gets zero, which is what keeps the ``calibration`` split (5 %) honest on a
        small corpus instead of quietly stealing a row from the training split.
    """
    exact = {name: shares[name] * float(total) for name in shares}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    missing = int(total) - sum(counts.values())
    ranked = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in ranked[:missing]:
        counts[name] += 1
    return counts


def _parse_date(value: Any) -> date | None:
    """Parse an ISO date, returning ``None`` when the value is absent."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def class_keywords() -> dict[str, Sequence[str]]:
    """Return the keyword list of every class (used by the notebooks and the report)."""
    return {template.label: template.keywords for template in TEMPLATES}


__all__ = [
    "CLASS_WEIGHTS",
    "PHRASING_HOLDOUT",
    "PRIORITY_SHARES",
    "PRODUCTS",
    "SOURCE_SHARES",
    "SPLIT_SHARES",
    "STYLE_SHARES",
    "ClassTemplate",
    "SyntheticTicketGenerator",
    "TEMPLATES",
    "class_keywords",
    "keyword_baseline_accuracy",
]
