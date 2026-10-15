"""Vocabulaire du corpus de comptes-rendus d'intervention.

Tout ce qui est écrit dans les documents vient d'ici. La pièce maîtresse est la notion de
**profil d'équipement** : un équipement, son symptôme, sa cause probable, l'action qui la corrige et
les références de pièce qui peuvent être consommées. Le générateur tire **un profil par document**,
puis écrit un compte-rendu cohérent autour de lui.

Pourquoi cette structure plutôt que des listes indépendantes ? Parce qu'un corpus
généré par tirages indépendants produit des phrases *grammaticalement* correctes et
*techniquement* absurdes : une fuite sur un automate, un joint remplacé sur un capteur.
Un lecteur le remarque en trois lignes, et le projet perd sa crédibilité de référence.
Ici, la cohérence est une propriété de construction, et la suite de tests la vérifie.

Le décor — numéros de ticket, de téléphone, de parking, horaires de présence — est un **distracteur
déclaré** : il apparaît dans les documents et jamais dans les résumés de référence. Un modèle qui le
recopie produit un résumé plus long et moins couvert, un modèle qui l'invente est détecté comme non
supporté.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EquipmentProfile:
    """Un équipement, sa défaillance et les opérations qui lui correspondent.

    Attributes:
        equipment: Équipement tel qu'il est désigné dans le texte (article compris).
        symptom: Symptôme observable de cette défaillance.
        cause: Cause identifiée au diagnostic.
        action: Action corrective réalisée par le technicien.
        parts: Références de pièce consommables (la première est la pièce principale).
        family: Famille technique de l'équipement (utilisée par les notebooks pour segmenter).
    """

    equipment: str
    symptom: str
    cause: str
    action: str
    parts: tuple[str, ...]
    family: str


#: Profils d'équipement : chaque document est écrit autour d'un seul d'entre eux.
EQUIPMENT_PROFILES: tuple[EquipmentProfile, ...] = (
    EquipmentProfile(
        equipment="la pompe P-12",
        symptom="une fuite au niveau du joint",
        cause="un joint d'étanchéité usé",
        action="le remplacement du joint",
        parts=("JNT-204", "JNT-311"),
        family="hydraulique",
    ),
    EquipmentProfile(
        equipment="le convoyeur C-3",
        symptom="des vibrations anormales",
        cause="un roulement desserré",
        action="le resserrage du roulement",
        parts=("RLM-118", "RLM-220"),
        family="mécanique",
    ),
    EquipmentProfile(
        equipment="l'automate AU-7",
        symptom="un défaut de communication",
        cause="un câble de liaison abîmé",
        action="le remplacement du câble de liaison",
        parts=("CBL-045", "CBL-072"),
        family="électrique",
    ),
    EquipmentProfile(
        equipment="le variateur V-2",
        symptom="une surchauffe du moteur",
        cause="un ventilateur encrassé",
        action="le nettoyage du ventilateur",
        parts=("VNT-330", "FLT-512"),
        family="électrique",
    ),
    EquipmentProfile(
        equipment="le capteur CP-15",
        symptom="une dérive de mesure",
        cause="un encrassement de la lentille",
        action="le nettoyage de la lentille",
        parts=("LNT-140", "JNT-204"),
        family="instrumentation",
    ),
    EquipmentProfile(
        equipment="la vanne VN-4",
        symptom="une perte de pression",
        cause="un filtre colmaté",
        action="le remplacement du filtre",
        parts=("FLT-512", "FLT-530"),
        family="hydraulique",
    ),
    EquipmentProfile(
        equipment="le groupe froid GF-1",
        symptom="une chute de rendement",
        cause="un échangeur entartré",
        action="le détartrage de l'échangeur",
        parts=("DTR-127", "FLT-530"),
        family="thermique",
    ),
    EquipmentProfile(
        equipment="la presse PR-6",
        symptom="un bruit de frottement",
        cause="une courroie détendue",
        action="la tension de la courroie",
        parts=("CRR-241", "CRR-255"),
        family="mécanique",
    ),
    EquipmentProfile(
        equipment="le moteur MT-9",
        symptom="un arrêt intempestif",
        cause="une alimentation instable",
        action="le contrôle de l'alimentation",
        parts=("ALM-075", "CBL-072"),
        family="électrique",
    ),
    EquipmentProfile(
        equipment="la centrale hydraulique CH-2",
        symptom="une chute de débit",
        cause="un réglage de pression erroné",
        action="le réglage de la pression",
        parts=("PRS-018", "FLT-512"),
        family="hydraulique",
    ),
    EquipmentProfile(
        equipment="le doseur DO-8",
        symptom="un écart de dosage",
        cause="une vis d'Archimède usée",
        action="le remplacement de la vis d'Archimède",
        parts=("VIS-063", "JNT-311"),
        family="mécanique",
    ),
    EquipmentProfile(
        equipment="l'armoire AR-5",
        symptom="un déclenchement répété",
        cause="un disjoncteur sous-dimensionné",
        action="le remplacement du disjoncteur",
        parts=("DSJ-088", "ALM-075"),
        family="électrique",
    ),
)

#: Actions complémentaires, non saillantes : elles s'ajoutent à l'action corrective principale sans
#: lui être liées, et donnent au document des faits exacts que le résumé n'a pas à rapporter.
SECONDARY_ACTIONS: tuple[str, ...] = (
    "le contrôle des fixations",
    "le relevé des températures",
    "un essai de fonctionnement à vide",
    "le graissage des guidages",
    "la vérification des serrages",
    "le contrôle de l'isolement électrique",
)

#: Durées d'intervention, écrites en toutes lettres et en chiffres : les deux formes coexistent dans
#: un vrai compte-rendu, et la couverture les traite comme deux faits distincts.
DURATIONS: tuple[str, ...] = (
    "45 minutes",
    "1 heure 30",
    "2 heures",
    "25 minutes",
    "3 heures",
    "1 heure",
    "50 minutes",
    "4 heures",
)

#: Statut de remise en service prononcé à la fin de l'intervention.
STATUSES: tuple[str, ...] = (
    "la remise en service immédiate",
    "une mise en essai prolongée",
    "un arrêt maintenu jusqu'à la prochaine visite",
    "une remise en service sous surveillance",
    "un redémarrage différé",
)

#: Décor : sites anonymisés où se déroulent les interventions.
SITES: tuple[str, ...] = ("SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F")

#: Techniciens (rôles anonymes, jamais de nom de personne réel). Les valeurs sont écrites **sans
#: article** : les gabarits les emploient après « du », et « du le technicien » serait une faute de
#: génération.
TECHNICIANS: tuple[str, ...] = (
    "technicien de maintenance",
    "électromécanicien de permanence",
    "technicien d'astreinte",
    "agent de maintenance préventive",
)

#: Nature de l'intervention : c'est le premier segment du rapport.
INTERVENTION_TYPES: tuple[str, ...] = ("depannage", "maintenance", "installation", "expertise")

#: Urgence déclarée : distracteur côté modèle, segment côté rapport.
URGENCIES: tuple[str, ...] = ("basse", "normale", "haute")

#: Types de fait rapportables et ordre d'écriture dans la table des faits. L'ordre est celui du
#: raisonnement du technicien : sur quoi, quel symptôme, quelle cause, quelle action, quelle pièce,
#: combien de temps, et dans quel état le matériel est laissé.
FACT_TYPES: tuple[str, ...] = ("equipement", "symptome", "cause", "action", "piece", "duree", "statut")

#: Gabarits de phrase du document, par type de fait. ``{equipment}`` et consorts sont remplacés par
#: une valeur du vocabulaire ; la phrase produite est la phrase du document **et** la trace du fait.
#: Deux règles d'écriture, vérifiées par les tests : aucun gabarit ne place une valeur immédiatement
#: après une préposition qui exigerait sa contraction (« à le », « de les »), et l'initiale de
#: chaque phrase est remise en majuscule au moment de l'assemblage — un compte-rendu qui commence
#: par « la pompe P-12 » serait un artefact de génération, pas un texte.
DOCUMENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "equipement": (
        "Sur le site, le contrôle a porté sur {equipment} à l'arrivée du {technician}.",
        "L'intervention porte sur {equipment} dans le cadre du plan de maintenance préventive.",
        "Le diagnostic a commencé par un relevé complet sur {equipment}.",
    ),
    "symptome": (
        "{equipment} présentait {symptom} depuis le début de la semaine.",
        "Le client a signalé {symptom} sur {equipment}.",
        "À l'arrivée, {equipment} montrait {symptom} de façon répétée.",
    ),
    "cause": (
        "Le diagnostic a identifié {cause} comme origine du problème.",
        "Après démontage, {cause} explique l'anomalie constatée.",
        "La cause retenue est {cause}, confirmée par un essai à vide.",
    ),
    "action": (
        "Action réalisée sur {equipment} : {action}.",
        "Le compte-rendu mentionne {action} parmi les opérations effectuées.",
        "Parmi les opérations figure {action}.",
    ),
    "piece": (
        "La pièce {part} a été remplacée et référencée au stock.",
        "Le magasin a fourni {part} pour cette intervention.",
        "Le remplacement a consommé la référence {part}.",
    ),
    "duree": (
        "L'intervention a duré {duration}.",
        "Le temps passé sur site est de {duration}.",
        "La durée totale enregistrée est de {duration}.",
    ),
    "statut": (
        "En fin d'intervention, le compte-rendu retient {status}.",
        "Le compte-rendu se termine par {status}.",
        "La conclusion du technicien est {status}.",
    ),
}

#: Phrases de décor : elles ne portent aucun fait rapportable et servent de distracteurs.
FILLER_SENTENCES: tuple[str, ...] = (
    "Le client était absent lors du passage et a été informé par téléphone.",
    "Le ticket {ticket} a été ouvert le matin par le service client.",
    "Les horaires de présence du site sont de 6 heures à 22 heures.",
    "Le parking du bâtiment {building} était occupé par une livraison.",
    "Une réunion de production était en cours dans l'atelier voisin.",
    "Le plan de prévention a été visé avant le début des travaux.",
    "La météo annonçait de la pluie, la zone a été bâchée par précaution.",
    "Le superviseur du site a été prévenu de la fin des opérations.",
    "Le relevé des compteurs a été transmis au service technique.",
    "Un contrôle visuel des protections a été effectué avant l'essai.",
)

#: Gabarits des phrases du **résumé de référence**, par type de fait. Les valeurs y sont reprises à
#: l'identique, mais la phrase est réécrite : c'est ce qui empêche un modèle de recopier le document
#: et d'obtenir un ROUGE de 1,0. Chaque type a **deux** formulations, dont une seule est utilisée
#: par document : deux résumés ne se ressemblent donc pas, et un modèle ne peut pas apprendre la
#: phrase du résumé par cœur.
SUMMARY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "equipement": (
        "L'intervention a ciblé {equipment}.",
        "Le matériel concerné est {equipment}.",
    ),
    "symptome": (
        "Le symptôme constaté était {symptom}.",
        "L'anomalie signalée est {symptom}.",
    ),
    "cause": (
        "L'origine identifiée est {cause}.",
        "La cause retenue est {cause}.",
    ),
    "action": (
        "L'opération réalisée a été {action}.",
        "Le compte-rendu retient {action}.",
    ),
    "piece": (
        "La pièce remplacée porte la référence {part}.",
        "La référence {part} a été consommée.",
    ),
    "duree": (
        "Le temps passé sur site a été de {duration}.",
        "La durée d'intervention est de {duration}.",
    ),
    "statut": (
        "Le matériel est laissé avec {status}.",
        "La conclusion retenue est {status}.",
    ),
}

#: Formules d'ouverture du résumé, par registre. Le registre change la formulation sans changer les
#: faits : c'est la part de réécriture publiée avec le corpus.
SUMMARY_OPENERS: tuple[str, ...] = (
    "Compte-rendu de l'intervention.",
    "Synthèse de l'intervention réalisée.",
    "Résumé à l'attention du responsable de site.",
)

#: Distracteurs du corpus : des valeurs qui ressemblent à des faits et n'en sont pas.
DISTRACTORS: tuple[str, ...] = (
    "TKT-4471",
    "TKT-8820",
    "01 44 55 66 77",
    "PC-31",
    "BAT-2",
)

#: Parts des splits. ``calibration`` sert au notebook 04 à régler le budget de décodage, ``test`` ne
#: sert qu'une fois.
SPLIT_SHARES: dict[str, float] = {
    "train": 0.60,
    "val": 0.20,
    "calibration": 0.10,
    "test": 0.10,
}

#: Nombre de phrases d'un document selon sa nature (le corpus doit rester lisible et compact).
SENTENCE_RANGE: dict[str, tuple[int, int]] = {
    "depannage": (9, 14),
    "maintenance": (10, 15),
    "installation": (11, 16),
    "expertise": (9, 13),
}

#: Nombre de phrases du résumé : bornes du contrat (une par type de fait saillant, plus
#: l'ouverture).
SUMMARY_SENTENCE_RANGE: tuple[int, int] = (2, 8)

#: Durée maximale d'un résumé rapportée à son document (borne du contrat). Le générateur ne coupe
#: **jamais** une phrase porteuse de fait pour respecter cette borne : il allonge le document (une
#: phrase de décor de plus), parce qu'un résumé de référence qui omettrait un fait saillant rendrait
#: la couverture de référence inatteignable et la mesure trompeuse.
MAX_COMPRESSION: float = 0.6


def profiles_by_family() -> dict[str, tuple[str, ...]]:
    """Group the equipment profiles by technical family.

    Returns:
        Mapping ``famille -> équipements``, useful to the notebooks that segment the corpus.
    """
    grouped: dict[str, list[str]] = {}
    for profile in EQUIPMENT_PROFILES:
        grouped.setdefault(profile.family, []).append(profile.equipment)
    return {name: tuple(values) for name, values in grouped.items()}


def profile_of(equipment: str) -> EquipmentProfile | None:
    """Return the profile of an equipment.

    Args:
        equipment: Equipment designation, as written in the corpus.

    Returns:
        The matching profile, or ``None`` when the equipment is unknown.
    """
    for profile in EQUIPMENT_PROFILES:
        if profile.equipment == equipment:
            return profile
    return None


def fact_values(fact_type: str) -> tuple[str, ...]:
    """Return the flat vocabulary of one fact type (used by the tests and the notebooks).

    Args:
        fact_type: One of :data:`FACT_TYPES`.

    Returns:
        The distinct values of that type across the profiles, plus the secondary pools for the
        actions. An unknown type yields an empty tuple.
    """
    if fact_type == "equipement":
        return tuple(profile.equipment for profile in EQUIPMENT_PROFILES)
    if fact_type == "symptome":
        return tuple(profile.symptom for profile in EQUIPMENT_PROFILES)
    if fact_type == "cause":
        return tuple(profile.cause for profile in EQUIPMENT_PROFILES)
    if fact_type == "action":
        return tuple(
            dict.fromkeys(
                [profile.action for profile in EQUIPMENT_PROFILES] + list(SECONDARY_ACTIONS)
            )
        )
    if fact_type == "piece":
        return tuple(
            dict.fromkeys(part for profile in EQUIPMENT_PROFILES for part in profile.parts)
        )
    if fact_type == "duree":
        return DURATIONS
    if fact_type == "statut":
        return STATUSES
    return ()


def vocabulary_report() -> dict[str, float | int | list[str] | dict[str, float]]:
    """Describe the vocabulary (published in the generation metadata).

    Returns:
        Profile count, per-type sizes, declared distractors, split shares and equipment families.
    """
    report: dict[str, float | int | list[str] | dict[str, float]] = {
        "n_profiles": len(EQUIPMENT_PROFILES),
        "n_equipments": len(fact_values("equipement")),
        "n_symptoms": len(fact_values("symptome")),
        "n_causes": len(fact_values("cause")),
        "n_actions": len(fact_values("action")),
        "n_parts": len(fact_values("piece")),
        "n_durations": len(DURATIONS),
        "n_statuses": len(STATUSES),
        "n_sites": len(SITES),
        "n_intervention_types": len(INTERVENTION_TYPES),
        "n_filler_sentences": len(FILLER_SENTENCES),
        "n_fact_types": len(FACT_TYPES),
        "families": sorted(profiles_by_family()),
        "distractors": list(DISTRACTORS),
        "split_shares": {name: float(share) for name, share in SPLIT_SHARES.items()},
    }
    return report


def preferred_types(intervention_type: str) -> Sequence[str]:
    """Return the fact types an intervention of that nature tends to produce.

    Ce n'est pas une contrainte de génération, seulement un ordre de priorité : une
    expertise produit plus de causes que d'actions, une installation plus de pièces que de
    symptômes. Les notebooks l'affichent pour expliquer la distribution des faits du corpus.

    Args:
        intervention_type: Nature de l'intervention.

    Returns:
        The preferred fact types, most informative first.
    """
    preferences: dict[str, tuple[str, ...]] = {
        "depannage": ("equipement", "symptome", "cause", "action", "piece", "duree", "statut"),
        "maintenance": ("equipement", "action", "piece", "duree", "statut", "cause", "symptome"),
        "installation": ("equipement", "piece", "action", "duree", "statut", "symptome", "cause"),
        "expertise": ("equipement", "cause", "symptome", "duree", "statut", "action", "piece"),
    }
    return preferences.get(intervention_type, FACT_TYPES)


__all__ = [
    "DISTRACTORS",
    "DOCUMENT_TEMPLATES",
    "DURATIONS",
    "EQUIPMENT_PROFILES",
    "FACT_TYPES",
    "FILLER_SENTENCES",
    "INTERVENTION_TYPES",
    "MAX_COMPRESSION",
    "SECONDARY_ACTIONS",
    "SENTENCE_RANGE",
    "SITES",
    "SPLIT_SHARES",
    "STATUSES",
    "SUMMARY_OPENERS",
    "SUMMARY_SENTENCE_RANGE",
    "SUMMARY_TEMPLATES",
    "TECHNICIANS",
    "URGENCIES",
    "EquipmentProfile",
    "fact_values",
    "preferred_types",
    "profile_of",
    "profiles_by_family",
    "vocabulary_report",
]
