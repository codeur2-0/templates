"""Vocabulaire du corpus d'entités nommées, et rien d'autre.

Ce module est le **seul** endroit où le vocabulaire du corpus est écrit : les noms de produits, les
transporteurs, les familles de produits, les villes et les mois. Le générateur y puise ses surfaces,
la suite de tests y vérifie que ce qui est écrit dans les textes vient bien du vocabulaire déclaré,
et le rapport peut donc dire *ce qui* a été réservé aux splits d'évaluation au lieu de le laisser
deviner.

Trois listes jouent un rôle différent, et la distinction est le cœur du projet :

* :data:`PRODUITS` et :data:`TRANSPORTEURS` sont les surfaces **vues à l'entraînement** : un modèle
  qui apprend des noms peut les retrouver ;
* :data:`PRODUITS_RESERVES` et :data:`TRANSPORTEURS_RESERVES` sont **réservés aux splits
  d'évaluation** : jamais présents dans le train. Un modèle qui n'apprend que des listes ne peut pas
  les retrouver ; un modèle qui apprend la *forme* d'un nom d'entité en retrouve une partie — c'est
  exactement l'écart que le projet mesure ;
* :data:`FAMILLES` et :data:`VILLES` sont des **distracteurs déclarés** : ils apparaissent dans les
  textes et ne sont jamais annotés. « le casque » n'est pas un produit (« Casque audio Aria 3 » en
  est un), et « Lyon » n'appartient à aucune des cinq catégories. Sans ces pièges, un modèle qui
  surligne tous les noms communs du domaine obtiendrait une précision apparente flatteuse.

Les gabarits de rédaction (:data:`TEMPLATES_REDIGE`, :data:`TEMPLATES_ABREGE`) sont eux aussi
déclarés ici : chaque `{slot}` dont le nom est un type d'entité est annoté, les autres
(``{numero}``, ``{ville}``) sont recopiés littéralement. Un numéro de facture n'est pas une
référence de commande, et une ville n'est pas une entité du corpus.
"""

from __future__ import annotations

from typing import Any, Final

#: Types d'entités du corpus, dans l'ordre d'affichage des rapports.
LABELS: Final[tuple[str, ...]] = ("produit", "commande", "montant", "date", "transporteur")

#: Ordre de résolution des chevauchements (du plus spécifique au plus général) : quand deux règles
#: proposent des spans qui se recouvrent, la première de cette liste gagne.
LABEL_PRIORITY: Final[tuple[str, ...]] = ("commande", "date", "montant", "produit", "transporteur")

#: Produits **vus à l'entraînement** : ``(forme rédigée, forme abrégée)``.
PRODUITS: Final[tuple[tuple[str, str], ...]] = (
    ("Casque audio Aria 3", "casque Aria 3"),
    ("Clavier mécanique Nova 7", "clavier Nova 7"),
    ("Enceinte Bluetooth Pulse", "enceinte Pulse"),
    ("Souris ergonomique Vega", "souris Vega"),
    ("Écran 27 pouces Orion", "écran Orion"),
    ("Imprimante laser Kappa", "imprimante Kappa"),
    ("Tablette graphique Sigma", "tablette Sigma"),
    ("Routeur Wi-Fi Delta", "routeur Delta"),
    ("Disque SSD Vega 2 To", "SSD Vega 2 To"),
    ("Station d'accueil Atlas", "dock Atlas"),
    ("Webcam studio Iota", "webcam Iota"),
    ("Micro USB Theta", "micro Theta"),
    ("Batterie externe Zeta", "batterie Zeta"),
    ("Câble HDMI Omega", "câble Omega"),
    ("Onduleur 900 VA Rho", "onduleur Rho"),
    ("Barrette mémoire Nu 16 Go", "barrette Nu 16 Go"),
    ("Boîtier mini-ITX Mu", "boîtier Mu"),
    ("Refroidisseur liquide Xi", "refroidisseur Xi"),
)

#: Produits **réservés aux splits d'évaluation** : jamais vus à l'entraînement.
PRODUITS_RESERVES: Final[tuple[tuple[str, str], ...]] = (
    ("Enceinte nomade Zéphyr 2", "enceinte Zéphyr 2"),
    ("Souris verticale Lyra", "souris Lyra"),
    ("Écran incurvé 32 pouces Tau", "écran Tau"),
    ("Clavier compact Psi 60", "clavier Psi 60"),
    ("Imprimante couleur Chi 400", "imprimante Chi 400"),
    ("Casque de studio Omicron", "casque Omicron"),
)

#: Transporteurs (organisations) vus à l'entraînement.
TRANSPORTEURS: Final[tuple[str, ...]] = (
    "Chronopost",
    "DHL",
    "Colissimo",
    "GLS",
    "Mondial Relay",
    "UPS",
    "FedEx",
    "Bpost",
)

#: Transporteurs réservés aux splits d'évaluation : jamais vus à l'entraînement.
TRANSPORTEURS_RESERVES: Final[tuple[str, ...]] = ("TNT Express", "DPD", "Geodis", "GLS Express")

#: Familles de produits : présentes dans les textes, **jamais annotées**.
FAMILLES: Final[tuple[str, ...]] = (
    "casque",
    "clavier",
    "enceinte",
    "souris",
    "écran",
    "imprimante",
    "tablette",
    "routeur",
    "disque",
    "câble",
    "batterie",
    "webcam",
)

#: Villes : présentes dans les textes, **jamais annotées** (hors taxonomie).
VILLES: Final[tuple[str, ...]] = ("Lyon", "Nantes", "Lille", "Toulouse", "Bordeaux", "Strasbourg")

#: Mois français utilisés par les dates rédigées.
MOIS: Final[tuple[str, ...]] = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)

#: Canaux d'arrivée d'un message, avec leur part cible.
CANAUX: Final[tuple[str, ...]] = ("formulaire", "email", "chat", "courrier")

#: Styles rédactionnels : ``redige`` écrit les surfaces complètes, ``abrege`` les abrège
#: (« Casque audio Aria 3 » devient « casque Aria 3 », « CMD-1234 » devient « cmd 1234 »).
STYLES: Final[tuple[str, ...]] = ("redige", "abrege")

#: Splits du corpus et leurs parts cibles. ``calibration`` sert au notebook de calibration de la
#: confiance, pas à l'entraînement.
SPLIT_SHARES: Final[dict[str, float]] = {
    "train": 0.60,
    "val": 0.20,
    "calibration": 0.05,
    "test": 0.15,
}

#: Parts cibles des deux styles rédactionnels.
STYLE_SHARES: Final[dict[str, float]] = {"redige": 0.55, "abrege": 0.45}

#: Part des mentions qui utilisent une surface **réservée** dans un split d'évaluation.
PART_RESERVEE: Final[float] = 0.75

#: Probabilité de piocher une surface réservée quand la ligne y a droit (un tirage laisse une part
#: de mentions ordinaires dans les splits d'évaluation : sans elles, l'écart mesurerait un tirage).
RESERVED_DRAW: Final[float] = 0.65

#: Gabarits rédigés : les slots nommés d'après un type d'entité sont annotés.
TEMPLATES_REDIGE: Final[tuple[str, ...]] = (
    "Bonjour, ma commande {commande} passée le {date} n'est jamais arrivée : {transporteur} "
    "m'indique que le colis est bloqué en agence.",
    "Je vous écris au sujet de la commande {commande} livrée par {transporteur} : le {produit} "
    "est arrivé endommagé le {date}.",
    "La commande {commande} du {date} est incomplète : il manque le {produit}, et la facture "
    "n° {numero} mentionne {montant}.",
    "Le {produit} commandé le {date} sous la référence {commande} a été retourné à {transporteur} "
    "sans aucune explication.",
    "Bonjour, {transporteur} a bien livré le {produit} de la commande {commande}, mais la facture "
    "de {montant} est erronée.",
    "Ma commande {commande} a été annulée le {date} alors que j'avais réglé {montant} par carte "
    "bancaire.",
    "Le {produit} de la commande {commande} est en panne depuis le {date} : j'ai saisi "
    "{transporteur} et j'attends une réponse.",
    "Vous m'annoncez un remboursement de {montant} le {date} pour la commande {commande}, mais je "
    "n'ai rien reçu à ce jour.",
    "La livraison du {produit} prévue le {date} par {transporteur} a été reportée, et la commande "
    "{commande} reste ouverte.",
    "Bonjour, je souhaite retourner le {produit} reçu à {ville} le {date} : la commande "
    "{commande} ne correspond pas à ce que j'ai commandé.",
    "Le colis de la commande {commande} a été déposé chez le voisin par {transporteur} le {date} : "
    "je demande le remboursement de {montant}.",
    "Je n'ai pas reçu le {produit} de la commande {commande}, et {transporteur} évoque un problème "
    "d'adresse depuis le {date}.",
    "Après trois relances pour la commande {commande}, {transporteur} m'annonce une livraison le "
    "{date} : le dossier {numero} n'avance pas.",
    "Le {produit} livré le {date} ne correspond pas à la commande {commande} : je demande un "
    "échange ou le remboursement de {montant}.",
)

#: Gabarits abrégés : mêmes entités, surfaces plus courtes, ponctuation absente ou réduite.
TEMPLATES_ABREGE: Final[tuple[str, ...]] = (
    "cmd {commande} pas recue, {transporteur} dit colis bloque. je veux le remboursement de "
    "{montant}",
    "bonjour, souci sur {commande} du {date} : le {produit} est casse (voir facture {numero})",
    "{commande} livree par {transporteur} le {date} : {produit} manquant, merci de me dire pour "
    "les {montant}",
    "je renvoie le {produit} de {commande} recu le {date} chez {transporteur}",
    "client depuis 2025, commande {commande} toujours en attente, {montant} debites le {date} par "
    "{transporteur}",
    "colis {commande} bloque chez {transporteur} depuis le {date}, le {produit} devait arriver a "
    "{ville}",
    "2eme relance pour {commande} : le {produit} est hs, je demande {montant} de dedommagement",
    "toujours rien pour {commande} du {date}, {produit} jamais expedie, {transporteur} injoignable",
    "facture {numero} : {montant} preleves pour {commande} mais {produit} non livre le {date}",
    "besoin d'un retour pour le {produit} ({commande}) recu le {date} via {transporteur}",
    "le {produit} de {commande} est arrive casse le {date}, {transporteur} refuse le litige",
    "merci de rembourser {montant} sur {commande} : {produit} en rupture depuis le {date}",
)

#: Slots qui ne sont **pas** des entités : ils sont recopiés littéralement dans les textes.
DISTRACTOR_SLOTS: Final[tuple[str, ...]] = ("numero", "ville")


def templates_by_style() -> dict[str, tuple[str, ...]]:
    """Return the declared templates, indexed by editorial style.

    Returns:
        Mapping of style name to the templates of that style.
    """
    return {"redige": TEMPLATES_REDIGE, "abrege": TEMPLATES_ABREGE}


def vocabulary_report() -> dict[str, Any]:
    """Describe the declared vocabulary (used by the metadata and the reports).

    Returns:
        Counts per list, plus the reserved surfaces and the declared distractors.
    """
    return {
        "n_produits": len(PRODUITS),
        "n_produits_reserves": len(PRODUITS_RESERVES),
        "n_transporteurs": len(TRANSPORTEURS),
        "n_transporteurs_reserves": len(TRANSPORTEURS_RESERVES),
        "n_familles": len(FAMILLES),
        "n_villes": len(VILLES),
        "n_templates_redige": len(TEMPLATES_REDIGE),
        "n_templates_abrege": len(TEMPLATES_ABREGE),
        "reserved_products": [surface for pair in PRODUITS_RESERVES for surface in pair],
        "reserved_carriers": list(TRANSPORTEURS_RESERVES),
    }


__all__ = [
    "CANAUX",
    "DISTRACTOR_SLOTS",
    "FAMILLES",
    "LABELS",
    "LABEL_PRIORITY",
    "MOIS",
    "PART_RESERVEE",
    "PRODUITS",
    "PRODUITS_RESERVES",
    "RESERVED_DRAW",
    "SPLIT_SHARES",
    "STYLES",
    "STYLE_SHARES",
    "TEMPLATES_ABREGE",
    "TEMPLATES_REDIGE",
    "TRANSPORTEURS",
    "TRANSPORTEURS_RESERVES",
    "VILLES",
    "templates_by_style",
    "vocabulary_report",
]
