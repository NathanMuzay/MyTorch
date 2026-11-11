# Compétences développées - MyTorch

Bilan établi le 11 novembre 2026 à partir des sources, configurations, tests et documents présents dans ce dépôt.

## Présentation

Bibliothèque de réseaux de neurones écrite en Python avec NumPy, appliquée à la classification de positions d’échecs. Le README cite Esteban Kenzi et Nathan Muzay. Le projet n’utilise pas PyTorch malgré son nom.

## Durée du projet

- Début du calendrier pédagogique : **12/10/2026**.
- Échéance indiquée : **08/11/2026**.
- Fenêtre prévue, dates incluses : **28 jours calendaires, soit 4 semaines** (27 jours d’écart entre les deux dates).
- Source : capture du tableau « Projets terminés » fournie par Nathan ; lecture de la date au pictogramme vert comme début et du drapeau rouge comme échéance. La date orange n’est pas utilisée comme début de développement.
- Historique du dépôt source local consulté : premier commit listé le **2026-10-12**, dernier commit listé le **2026-11-09**. Ces dates peuvent inclure des travaux après l’échéance et ne mesurent pas le temps de travail effectif.
- Heures investies, répartition individuelle et date exacte de fin de travail : **non renseignées**. Le commit d’import du portfolio ne représente pas la durée du projet.

## Lecture des niveaux de maîtrise

Ce bilan inventorie les compétences mobilisées et étayées par les livrables. Une analyse de dépôt ne permet pas de certifier l’acquisition individuelle, l’autonomie ni l’auteur de chaque réalisation. Les niveaux ci-dessous qualifient la profondeur d’utilisation visible **dans ce projet**, pas une certification ou un pourcentage de maîtrise personnelle de Nathan. Leur attribution individuelle reste à confirmer par ses contributions et une démonstration orale/pratique.

| Niveau | Signification dans ce document |
| --- | --- |
| Ciblé | Utilisation circonscrite à une fonction ou un outil ; expérience générale non établie. |
| Opérationnel | Utilisation concrète dans plusieurs tâches ou dans une fonctionnalité intégrée. |
| Approfondi | Plusieurs mécanismes internes sont implémentés ou articulés, avec des éléments de test ou d’analyse ; cela ne signifie pas expert du domaine. |

## Technologies et profondeur d’utilisation

| Technologie ou domaine | Niveau observé | Utilisation démontrée | Preuve dans le dépôt |
| --- | --- | --- | --- |
| Python 3 | Approfondi | Architecture en packages, classes abstraites, CLI, fichiers et pipelines | [src](src) |
| NumPy | Approfondi | Calcul matriciel, vectorisation, gradients, poids et encodages | [src/neural_network](src/neural_network) |
| Réseaux denses et rétropropagation | Approfondi | Forward/backward, pertes, optimisation et apprentissage supervisé | [src/neural_network/layers.py](src/neural_network/layers.py) |
| Adam, SGD, RMSprop | Approfondi | Implémentation des mises à jour et comparaison expérimentale | [src/neural_network/optimizers.py](src/neural_network/optimizers.py) |
| Dropout, BatchNorm, ordonnancement du taux | Opérationnel | Régularisation, états train/eval et schedulers | [src/neural_network](src/neural_network) |
| FEN et moteur de règles d’échecs | Opérationnel | Parsing, encodage, attaques et classification déterministe | [src/chess_engine](src/chess_engine) |
| SQLite, fichiers texte | Ciblé | Nettoyage et déduplication des corpus | [src/data/cleaning.py](src/data/cleaning.py) |
| JSON, Base64, format .nn | Opérationnel | Sérialisation des poids et métadonnées des réseaux | [src/neural_network/serialization.py](src/neural_network/serialization.py) |
| unittest, Makefile | Opérationnel | Tests des briques et lanceurs reproductibles | [tests](tests) |
| Tkinter, SVG | Ciblé | Visualisation de positions et génération de graphiques | [bonus/visualize.py](bonus/visualize.py) |
| Git, GitHub, Markdown | Opérationnel | Livraison de modèles, documentation et scripts de maintenance | [tools](tools) |

## Inventaire détaillé des compétences

### Mathématiques et bibliothèque neuronale

- Implémenter une couche dense, biais, produit matriciel et dérivées.
- Propager le gradient à travers une succession de couches.
- Implémenter ReLU, LeakyReLU, Sigmoid, Tanh et Softmax.
- Prendre en compte la stabilité numérique des activations et des pertes.
- Implémenter entropie croisée catégorielle/binaire, erreur quadratique et Focal Loss.
- Construire SGD avec momentum, Adam avec correction du biais et RMSprop.
- Initialiser les poids avec He, Xavier, LeCun et des variantes normales/uniformes.
- Implémenter Dropout et distinguer entraînement et inférence.
- Mettre en œuvre BatchNorm1d et ses statistiques.
- Composer un modèle Sequential et des interfaces de couches, pertes et optimiseurs.
- Faire varier le taux d’apprentissage avec StepLR, ExponentialLR et CosineAnnealingLR.
- Explorer MC-Dropout pour obtenir plusieurs prédictions stochastiques.

### Données et représentation

- Parser et valider une position FEN.
- Encoder pièces, joueur au trait et droits de roque dans 773 entrées.
- Construire une variante tactique à 1 333 entrées avec cartes d’attaque et informations supplémentaires.
- Séparer encodage, règles déterministes et prédiction du modèle.
- Générer des exemples de positions et leurs étiquettes.
- Nettoyer les corpus, identifier les lignes invalides et dédupliquer.
- Équilibrer l’échantillonnage par classe avec reservoir sampling.
- Créer une séparation stratifiée et reproductible entre apprentissage et validation.
- Contrôler les chevauchements de positions et conserver des empreintes de partitions.
- Augmenter les données par symétrie de couleurs avec adaptation des labels.
- Encoder des ensembles de positions via un point d’entrée parallélisable.
- Gérer trois ou cinq classes et leur correspondance.

### Entraînement et évaluation

- Entraîner par mini-lots et suivre pertes et scores.
- Utiliser early stopping et conserver le meilleur état du modèle.
- Comparer les configurations de profondeur, largeur, régularisation et optimisation.
- Évaluer avec matrice de confusion, précision, rappel et F1 par classe.
- Comparer deux encodages sur une partition commune.
- Distinguer performance de validation, test indépendant et généralisation.
- Développer des scripts de recherche d’hyperparamètres avec mutation et croisement.
- Produire des courbes SVG et consigner le protocole d’un benchmark.
- Diagnostiquer des erreurs de gradient et des biais de comparaison dans un rapport d’entraînement.

### Outils, interfaces et livraison

- Générer des réseaux depuis une configuration textuelle.
- Valider la cohérence entre encodage, dimension d’entrée et modèle.
- Sauvegarder et restaurer architecture, poids, types/tableaux et métadonnées.
- Livrer plusieurs modèles préentraînés distincts des réseaux aléatoires générés.
- Fournir des modes nettoyage, apprentissage, évaluation et prédiction en CLI.
- Gérer erreurs et code de retour 84 ; conserver des messages exploitables.
- Afficher une position avec Tkinter et exporter une représentation SVG.
- Développer des bonus de visualisation et de jeu autour du moteur d’échecs.
- Tester couches, règles, augmentation, nettoyage, configurations et parcours CLI.
- Documenter mathématiques, choix, résultats, commandes et limites en équipe.

## Limites et compétences non attestées

- PyTorch, TensorFlow et scikit-learn ne sont pas utilisés : ce projet ne prouve pas leur maîtrise.
- Les résultats historiques du README et du rapport ne sont pas des mesures réexécutées pour ce bilan.
- Les 94,26 % mentionnés dans le README concernent un protocole de validation décrit ; ce n’est ni un score universel ni une mesure du niveau personnel.
- CNN, Transformers, entraînement GPU distribué, MLOps de production et apprentissage par renforcement ne sont pas attestés par ce projet.

## Compétences transversales

- Traduire un sujet en fonctionnalités et organiser un livrable exécutable.
- Décomposer un problème, isoler les responsabilités et rechercher les causes d’erreurs.
- Lire et produire une documentation technique, des exemples et des commandes reproductibles.
- Utiliser Git pour versionner et livrer le travail.
- Relier un choix technique à ses contraintes et expliquer ses limites.

La gestion personnelle du temps, la répartition des tâches, le leadership et l’autonomie ne peuvent pas être déduits des seuls fichiers ; ils demandent un retour d’expérience individuel.

## Vérification et preuves

Les niveaux reposent sur une inspection des sources et sur les tests/rapports déjà présents. Aucun résultat d’exécution nouveau, taux de couverture ou succès de CI n’est revendiqué dans ce document. Les liens ci-dessus permettent de retrouver les implémentations. Les rapports historiques doivent être lus avec leur date, leurs conditions et leurs limites.

## Points à préciser pour une présentation individuelle

- Modules conçus, codés, testés ou documentés personnellement par Nathan.
- Contributions réalisées en autonomie, en binôme ou avec accompagnement.
- Difficulté rencontrée, diagnostic et solution qu’il peut expliquer sans support.
- Capacité actuelle à reproduire ou adapter les réalisations dans un autre projet.
- Temps réellement consacré et résultats officiellement évalués, si disponibles.
