"""Couche d'inférence de la reconnaissance d'entités.

Deux modules, séparés par responsabilité :

* :mod:`src.inference.extraction` — la mécanique partagée : extraire les mentions d'un lot message
  par message en chronométrant chaque appel, et dédoublonner. Le trainer, l'évaluateur et le service
  l'utilisent, donc la latence publiée est mesurée de la même façon partout ;
* :mod:`src.inference.predictor` — le service : recharger un artefact, lire son entrée (fichier ou
  échantillon du corpus), extraire, ajouter le verdict quand la référence est connue et écrire une
  table validée par le contrat Pandera avant persistance.
"""

from src.inference.extraction import MENTION_COLUMNS, deduplicate, extract_one_by_one
from src.inference.predictor import EntityPredictor

__all__ = ["MENTION_COLUMNS", "EntityPredictor", "deduplicate", "extract_one_by_one"]
