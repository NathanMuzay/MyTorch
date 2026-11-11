# Entraînement et évaluation du réseau

`src/my_torch.py` utilise le réseau `Sequential` de la bibliothèque `src/neural_network/`.
Sans option, le programme affiche l'aide.

## Entraîner

```sh
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn --classes 5 --epochs 25
```

L'entraînement appelle `src/training/train.py` : encodage FEN en 773 entrées,
échantillonnage équilibré, séparation entraînement/validation 85/15, couches
denses 512, 256 puis 128 avec BatchNorm, ReLU et Dropout, sortie softmax,
optimiseur Adam et planification CosineAnnealingLR. La perte par défaut est
l'entropie croisée catégorielle (CCE), sans pondération de classe ajoutée.
Le modèle entraîné est sauvegardé au format `.nn`.

La symétrie de couleurs est activée par défaut (`--augment-colors`). Après la
partition, elle ajoute les positions d'entraînement retournées sur les rangées,
avec couleurs, trait, roques et annotations White/Black échangés. Les variantes
qui reproduisent une entrée de validation sont exclues. La validation reste
inchangée. `--no-augment-colors` retrouve l'entraînement sans cette augmentation.
Sur le corpus actuel, les 51 000 exemples d'entraînement deviennent 101 995
exemples, pour toujours 9 000 exemples de validation. Il y a donc environ deux
fois plus de lots par époque, avec un besoin mémoire accru. Le corpus encodé
complet est libéré après la partition et l'augmentation évite une copie filtrée
complète supplémentaire pour limiter ce surcoût.

Les options `--batch-size`, `--max-per-class` et `--lr` règlent l'entraînement.
`--no-bn` désactive BatchNorm ; `--loss focal` sélectionne la Focal Loss et
`--boost-mate` ajoute les pondérations tactiques pour les Check et Checkmate.
L'échantillonnage parcourt le corpus entier, puis crée une partition stratifiée.
La graine `--seed` vaut 42 par défaut. Le taux initial vaut 0,001, le Dropout
`--dropout` vaut 0,25 et la régularisation `--weight-decay` vaut 0,0005.
`--patience 6` arrête l'entraînement après six époques sans amélioration.
Le meilleur réseau, avec ses statistiques BatchNorm, est restauré avant la
sauvegarde. `--monitor val_accuracy` est le défaut et retient la meilleure
précision ; `--monitor val_loss` privilégie la perte. `--no-boost-mate` désactive
explicitement les pondérations tactiques.
À précision de validation exactement égale, le réseau avec la perte la plus
faible est retenu. Cette égalité ne réinitialise pas la patience : seule une
amélioration du critère principal le fait.
`--label-smoothing` propose un lissage optionnel des cibles d'entraînement CCE,
désactivé par défaut (`0`). Par exemple, `0.05` transforme une cible cinq classes
en `0.96` pour la bonne classe et `0.01` pour chacune des autres. Les cibles de
validation et de test restent inchangées. Cette option est expérimentale : elle
n'a pas amélioré la précision du pilote décrit dans le rapport. Avec la Focal
Loss, le lissage doit rester à `0`.
Le bilan distingue la meilleure précision observée, le minimum de perte et
l'époque retenue. Il affiche les métriques par classe du modèle restauré et
les conserve, avec la matrice de confusion, dans les métadonnées du `.nn`.
Les métadonnées conservent également l'activation de la symétrie, les effectifs
avant/après augmentation et le nombre de variantes exclues.
`--classes 3` regroupe les annotations White/Black en `Nothing`, `Check` et
`Checkmate`. Avec 5 classes, les annotations de couleur sont conservées.
Sans `--output`, le modèle est enregistré dans `my_torch_network_full.nn`.
Sans `--dataset`, le fichier utilisé est `dataset/cleaned/train.txt` ; s'il
manque, le corpus d'exemple `dataset/examples/train_chessboards.txt` est utilisé en
3 classes. Un chemin personnalisé manquant provoque une erreur.

## Encodage tactique optionnel

`--encoding basic` reste le défaut et conserve les 773 entrées historiques.
`--encoding tactical` utilise le schéma version 1 de 1 333 entrées :

| Partie | Indices Python | Nombre | Représentation |
| --- | --- | ---: | --- |
| Encodage historique | `[0:773]` | 773 | Pièces, trait, roques inchangés |
| Prise en passant | `[773:837]` | 64 | Case cible, ou tout à zéro |
| Cartes d'attaque | `[837:965]` | 128 | 64 cases × 2 couleurs |
| Lignes autour des rois | `[965:1269]` | 304 | 2 rois × 8 directions × (12 pièces + 7 distances) |
| Voisinage des rois | `[1269:1333]` | 64 | 2 rois × 8 cases × 4 indicateurs |

Les attaques sont géométriques : pions en diagonale, sauts des cavaliers,
voisinage des rois, rayons des fous/tours/dames arrêtés après la première
pièce. Les pièces clouées contribuent à ces cartes ; les cases occupées et
défendues y figurent aussi. Le voisinage décrit une case présente sur le
plateau, une pièce amie, une pièce adverse et une attaque adverse après
retrait du roi de sa case actuelle. Il ne représente pas les coups légaux.
Les huit directions suivent l'ordre `(-1,-1), (-1,0), (-1,1), (0,-1),
(0,1), (1,-1), (1,0), (1,1)`, en coordonnées de tableau depuis a8.
Toutes les entrées sont binaires. Aucune annotation attendue, classification
par règles ou recherche de coups légaux n'est fournie au réseau.

La première couche passe de `773 → 512` à `1333 → 512` ; les couches cachées,
la sortie et les hyperparamètres restent identiques. Cette couche comporte
286 720 poids supplémentaires. Les modèles tactiques doivent être entraînés
avec ces entrées : les anciens poids ne sont pas convertis.
La symétrie de couleurs transforme également toutes les nouvelles entrées.
Ses exclusions de validation utilisent la même identité conservative dans
les deux encodages : pièces, trait et roques, en ignorant la prise en passant
et les compteurs. Une variante avec seulement une autre prise en passant
ne réintroduit donc pas un échiquier de validation dans l'entraînement.

Les modèles sauvegardent `encoding` et `encoding_version` dans leurs
métadonnées. L'évaluation, l'analyseur (prédiction et réentraînement) et le
bonus jouable sélectionnent cet encodage automatiquement. Les anciens
modèles sans ces champs gardent l'encodage de 773 entrées. Une dimension
incompatible ou une version inconnue est rejetée.

## Comparer les encodages

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m src.experiments.compare_encodings \
  --output-dir /tmp/torch_encoding_comparison --max-per-class 1000 \
  --epochs 20 --init-seeds 42 43 44
```

Le dossier de sortie doit être nouveau. Le script enregistre les six modèles
et `results.json`, sans toucher aux modèles existants. Les positions sont
échantillonnées et séparées avec la même graine (`--seed 42`) pour chaque
variante. `--init-seed` change uniquement l'initialisation et l'aléatoire de
l'entraînement ; par défaut, il reprend `--seed` pour conserver le protocole
historique. Le script vérifie les empreintes SHA-256 des partitions (entrées
historiques et annotations, dans leur ordre) ainsi que les effectifs et
exclusions d'augmentation. Les plafonds d'époques et réglages sont identiques ;
l'arrêt anticipé peut sélectionner des époques différentes. Les résultats
d'un petit corpus ne remplacent pas une comparaison sur les 60 000 positions.

Pour augmenter le corpus d'entraînement avec une validation fixe :

```sh
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt \
  --validation-dataset dataset/cleaned/validation.txt \
  --max-per-class 30000 --encoding tactical --output model_tactical_large.nn
```

Cette option utilise toutes les positions reconnues du fichier de validation,
sans sous-échantillonnage ni nouvelle séparation 85/15. Les positions
d'entraînement correspondant à une entrée de validation sont retirées avant
l'augmentation, avec l'identité conservative décrite plus haut. Le fichier
de validation doit rester le même pendant les essais. Ce protocole diffère
du score historique à 75,27 % : réentraîner aussi la référence `basic` avec
le même fichier pour comparer. Le fichier de test reste réservé à la mesure
finale, après le choix de la configuration.

## Évaluer

```sh
python3 src/my_torch.py -eval --model model.nn --dataset dataset/cleaned/test.txt
```

L'évaluation charge les classes et les poids du modèle, puis affiche la perte
et le pourcentage de prédictions correctes sur toutes les positions du fichier.
Elle utilise `Sequential.evaluate()` sans entraîner ni sauvegarder le modèle.
Le bilan comporte également précision, rappel et F1 par classe, macro-F1 et
matrice de confusion, avec les classes attendues en lignes et prédites en colonnes.
Les nouveaux modèles sauvegardent aussi la configuration de perte ; elle est
rechargée pour l'évaluation. Les anciens `.nn` sans cette configuration utilisent
la CCE par défaut : les valeurs de pertes de familles différentes ne sont pas
directement comparables.
`--batch-size` règle la taille des lots (32 par défaut). Chaque ligne doit
contenir une FEN et son annotation attendue. Un fichier vide, une FEN invalide
ou une annotation inconnue provoque une erreur avec le code 84.
Sans `--dataset`, le fichier utilisé est `dataset/cleaned/test.txt`.

Les formes `--train`, `--eval` et `python3 -m src.my_torch` sont aussi disponibles.
`-clean` conserve le nettoyage des données et ses options `--input`/`--output`.

Voir le [rapport d'entraînement et de corrections](training_report.md) pour
le journal de référence, les vérifications numériques et la commande de comparaison.
