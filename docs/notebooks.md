# Convention des notebooks

Les notebooks sont des **carnets de démonstration**, pas le lieu de la logique métier. Ils doivent :

1. commencer par le contexte et la question métier ;
2. charger les classes de `src/` plutôt que recopier le pipeline ;
3. valider les données et afficher la taille du jeu, les distributions et les limites ;
4. documenter chaque décision (split temporel, métrique, seuil, traitement des valeurs manquantes) ;
5. terminer par des conclusions, risques et prochaines étapes.

Avant de partager un notebook : exécuter toutes les cellules, supprimer les sorties trop volumineuses et vérifier qu'il ne contient ni secret ni donnée personnelle. Pour une présentation, `jupyter nbconvert --to html notebooks/01_*.ipynb` produit un artefact partageable.
