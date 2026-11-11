# Backlog & Issues GitHub - MY_TORCH_

Ce document répertorie l'ensemble des User Stories et Tâches techniques découpées pour le développement du projet **MY_TORCH_**.

---

## 📌 Milestone 1 : Socle Neural Network & Mathématiques (`my_torch`)

### Issue #1 : [CORE] Implémentation du moteur matriciel & couche abstraite `Layer`
- **Labels** : `core`, `enhancement`
- **Description** :
  Créer la classe de base abstraite `Layer` et l'architecture pour le passage avant (`forward`) et arrière (`backward`).
- **Tâches** :
  - Définir `Layer.forward(input_data)` et `Layer.backward(output_gradient)`.
  - Assurer la conservation des états et des activations pour la rétropropagation.
  - Documenter les dimensions d'entrée/sortie \((B, D_{in}) \to (B, D_{out})\).
- **Critères d'acceptation** :
  - Tests unitaires validant l'interface commune de toutes les futures couches.

---

### Issue #2 : [CORE] Couche Dense (Fully Connected) & Initialiseurs de poids
- **Labels** : `core`, `math`
- **Description** :
  Implémenter la couche linéaire dense \(Y = X \cdot W + b\) et les stratégies d'initialisation des poids.
- **Tâches** :
  - Initialisation **He / Kaiming** (uniforme & normale) pour activations ReLU.
  - Initialisation **Xavier / Glorot** pour activations Sigmoid / Tanh.
  - Initialisation **LeCun**.
  - Calcul du gradient \(\frac{\partial L}{\partial W} = X^T \cdot dY\) et \(\frac{\partial L}{\partial b} = \sum dY\).
  - Calcul de la rétropropagation du gradient d'entrée \(dX = dY \cdot W^T\).
- **Critères d'acceptation** :
  - Gradient checking numérique vérifiant l'exactitude mathématique de \(dW, db, dX\) à \(10^{-5}\) près.

---

### Issue #3 : [CORE] Fonctions d'activation et leurs dérivées
- **Labels** : `core`, `math`
- **Description** :
  Implémenter les fonctions d'activation non-linéaires et leurs dérivées pour le backpropagation.
- **Tâches** :
  - **ReLU** : \(f(x) = \max(0, x)\), \(f'(x) = 1_{x > 0}\).
  - **LeakyReLU** : \(f(x) = x\) si \(x > 0\) sinon \(\alpha x\).
  - **Sigmoid** : \(f(x) = \frac{1}{1 + e^{-x}}\), \(f'(x) = f(x)(1 - f(x))\).
  - **Tanh** : \(f(x) = \tanh(x)\), \(f'(x) = 1 - \tanh^2(x)\).
  - **Softmax** : version numériquement stable soustrayant le \(\max\).
- **Critères d'acceptation** :
  - Tests unitaires comparant la dérivée analytique aux différences finies numériques.
  - Absence d'overflow / NaN sur des entrées extrêmes (\(\pm 1000\)).

---

### Issue #4 : [CORE] Fonctions de perte (Losses)
- **Labels** : `core`, `math`
- **Description** :
  Implémenter les fonctions de coût et leurs gradients par rapport aux prédictions.
- **Tâches** :
  - **CategoricalCrossEntropy** combinée avec Softmax (\(dZ = \hat{y} - y\)).
  - **MeanSquaredError** (MSE) : \(\frac{1}{N} \sum (\hat{y} - y)^2\).
  - **BinaryCrossEntropy** (BCE).
- **Critères d'acceptation** :
  - La perte décroît de façon monotone sur un exemple appris par cœur.
  - Stabilité numérique avec clipping \(\epsilon = 10^{-15}\).

---

### Issue #5 : [CORE] Optimiseurs (SGD, Momentum, Adam, RMSprop)
- **Labels** : `core`, `optimization`
- **Description** :
  Implémenter les algorithmes d'optimisation et la mise à jour des paramètres.
- **Tâches** :
  - **SGD** basique et avec momentum \(\beta\).
  - **Adam** avec moments d'ordre 1 (\(m_t\)) et 2 (\(v_t\)), correction de biais et \(\epsilon\).
  - **RMSprop**.
  - Gestion du weight decay (régularisation L2) dans la mise à jour des poids.
- **Critères d'acceptation** :
  - Convergence rapide sur un problème synthétique non-linéaire (ex: XOR).

---

### Issue #6 : [CORE] Régularisation & MC-Dropout
- **Labels** : `core`, `optimization`
- **Description** :
  Ajouter la gestion du Dropout et MC-Dropout pour lutter contre l'overfitting comme exigé par le sujet.
- **Tâches** :
  - Couche `Dropout` avec probabilité de masque \(p\) et scaling \(1 / (1 - p)\).
  - Mode entraînement (`training=True`) vs mode inférence (`training=False`).
  - Option **MC-Dropout** (Monte Carlo Dropout) à l'inférence pour estimer l'incertitude.
- **Critères d'acceptation** :
  - Les activations masquées ont une moyenne préservée.
  - Inférence reproductible et stable en mode normal.

---

### Issue #7 : [CORE] Conteneur de modèle `Sequential` & Boucle d'entraînement
- **Labels** : `core`, `architecture`
- **Description** :
  Construire la classe orchestratrice du réseau de neurones (`Sequential`).
- **Tâches** :
  - Méthodes `add(layer)`, `forward(X)`, `backward(loss_grad)`.
  - Méthode `fit(X_train, y_train, epochs, batch_size, validation_data, callbacks)`.
  - Gestion des mini-batches avec mélange aléatoire (shuffle).
  - Calcul des métriques par époque (loss train, loss val, accuracy train, accuracy val).
  - Mécanisme d'arrêt précoce (**Early Stopping**).
- **Critères d'acceptation** :
  - Entraînement complet fonctionnel avec suivi clair des métriques par époque.

---

### Issue #8 : [CORE] Sérialisation du réseau (`.nn`) & Générateur `my_torch_generator`
- **Labels** : `core`, `cli`
- **Description** :
  Permettre la sauvegarde/chargement d'un modèle ainsi que la génération de modèles vierges via configuration.
- **Tâches** :
  - Spécifier un format `.nn` clair, autonome et lisible (JSON ou archive structurée) contenant l'architecture, hyperparamètres et tenseurs de poids.
  - Créer `my_torch_generator config_file_1 nb_1 [...]` capable de générer `nb_i` réseaux aléatoires vierges à partir d'un fichier de conf.
  - Implémenter `model.save(filepath)` et `model.load(filepath)`.
- **Critères d'acceptation** :
  - Un modèle sauvegardé puis rechargé produit des sorties identiques à \(10^{-9}\) près.
  - `./my_torch_generator basic_network.conf 3` génère 3 fichiers `.nn` valides.

---

## 📌 Milestone 2 : Échiquier, FEN & Feature Engineering (`chess_engine`)

### Issue #9 : [CHESS] Parser FEN robuste et validation
- **Labels** : `chess`, `parsing`
- **Description** :
  Implémenter un parseur de notation Forsyth–Edwards (FEN) avec validation stricte de la conformité.
- **Tâches** :
  - Parser les 6 champs de la FEN : disposition des pièces, trait (`w`/`b`), roques, prise en passant, demi-coups, coup total.
  - Valider l'intégrité de l'échiquier (présence d'un roi de chaque camp, 8 rangées valides, pions aux rangées autorisées).
  - Lever une exception explicite si la FEN est malformée.
- **Critères d'acceptation** :
  - Succès sur toutes les positions standard valides.
  - Rejet propre de FENs tronquées ou absurdes.

---

### Issue #10 : [CHESS] Encodeur d'échiquier en vecteur d'entrées
- **Labels** : `chess`, `ml`
- **Description** :
  Transformer une FEN en tenseur d'entrée optimisé pour le réseau de neurones.
- **Tâches** :
  - 12 canaux 8x8 pour les pièces (P, N, B, R, Q, K pour blanc et noir).
  - Canaux de métadonnées : joueur au trait, droits de roque, case en passant.
  - Extraction de features avancées : case des rois, attaques immédiates.
  - Normalisation des valeurs d'entrée.
- **Critères d'acceptation** :
  - Tenseur de forme fixe et constante pour toute position valide.
  - Temps d'encodage inférieur à 0.5 ms par position.

---

### Issue #11 : [CHESS] Moteur de règles d'échecs déterministe (Ground Truth)
- **Labels** : `chess`, `validation`
- **Description** :
  Implémenter la détection déterministe de l'état du jeu (`Nothing`, `Check`, `Checkmate`, `Stalemate`) pour labelliser les données et valider les modèles.
- **Tâches** :
  - Détection des attaques sur une case donnée (pions, cavaliers, fous/dames, tours/dames, rois).
  - Test si le roi du joueur au trait est en échec.
  - Génération des coups légaux pour vérifier si l'échec est imparable (`Checkmate`).
  - Détection de l'absence de coup légal sans être en échec (`Stalemate`).
- **Critères d'acceptation** :
  - Test sur une suite de positions de référence (mat du berger, mat du sot, positions nulles, parties d'échecs connues) à 100% de concordance.

---

## 📌 Milestone 3 : Exécutable `my_torch_analyzer` & Intégration CLI

### Issue #12 : [CLI] Interface en ligne de commande de l'analyseur
- **Labels** : `cli`, `epitech`
- **Description** :
  Implémenter le binaire `my_torch_analyzer` selon les spécifications strictes du sujet.
- **Tâches** :
  - Support de `./my_torch_analyzer --help`.
  - Support de `./my_torch_analyzer --predict LOADFILE CHESSFILE`.
  - Support de `./my_torch_analyzer --train [--save SAVEFILE] LOADFILE CHESSFILE`.
  - En mode `--predict`, affiche ligne par ligne la prédiction dans l'ordre exact du fichier d'entrée (`Checkmate`, `Check`, `Nothing`).
- **Critères d'acceptation** :
  - Conforme mot pour mot à la sortie demandée dans la documentation du sujet.

---

### Issue #13 : [EPITECH] Gestion stricte des erreurs et codes de retour (84 / 0)
- **Labels** : `epitech`, `error-handling`
- **Description** :
  Assurer que toute anomalie produit un message sur la sortie d'erreur (`stderr`) et quitte avec le code 84.
- **Tâches** :
  - Gestion des fichiers inexistants, illisibles ou corrompus.
  - Gestion des arguments inconnus ou en nombre incorrect.
  - Gestion des FEN invalides ou lignes vides.
  - Sortie avec 0 en cas de déroulement sans erreur.
- **Critères d'acceptation** :
  - Tous les tests de robustesse (fichiers manquants, arguments invalides) retournent le code 84 sur `stderr`.

---

### Issue #14 : [BUILD] Makefile conforme EPITECH
- **Labels** : `epitech`, `build`
- **Description** :
  Créer le Makefile avec les règles requises : `all`, `clean`, `fclean`, `re`, `tests_run`.
- **Tâches** :
  - Génère `my_torch_analyzer` et `my_torch_generator` avec permissions d'exécution.
  - `clean` nettoie les fichiers temporaires et caches `__pycache__`.
  - `fclean` nettoie les exécutables et fichiers résiduels.
  - `re` enchaîne `fclean` et `all`.
- **Critères d'acceptation** :
  - `make && make clean && make fclean && make re` s'exécute sans erreur sur l'environnement cible.

---

## 📌 Milestone 4 : Dataset, Entraînement & Modèle Pré-entraîné

### Issue #15 : [DATA] Script de génération de datasets équilibrés
- **Labels** : `data`, `scripts`
- **Description** :
  Créer `scripts/generate_dataset.py` pour générer des milliers d'échiquiers FEN labellisés.
- **Tâches** :
  - Équilibrage des classes : ~33% `Nothing`, ~33% `Check`, ~33% `Checkmate`.
  - Génération aléatoire ou basée sur des parties réelles avec perturbations.
  - Format de sortie compatible avec `CHESSFILE` (`FEN label`).
- **Critères d'acceptation** :
  - Dataset de 10 000+ positions généré sans doublons aberrants et 100% vérifié par le moteur de règles.

---

### Issue #16 : [TRAIN] Entraînement et livraison de `my_torch_network_basic.nn`
- **Labels** : `ml`, `deliverable`
- **Description** :
  Entraîner le réseau de neurones supervisé et livrer le modèle pré-entraîné obligatoire dans le dépôt.
- **Tâches** :
  - Entraîner avec Adam, mini-batches, régularisation L2 et dropout.
  - Validation sur un split de test indépendant.
  - Sauvegarder sous le nom obligatoire `my_torch_network_basic.nn`.
- **Critères d'acceptation** :
  - Le modèle est présent à la racine du dépôt.
  - Précision élevée sur la classification des 3 états d'échiquier.

---

## 📌 Milestone 5 : Documentation, Benchmarks & Bonus

### Issue #17 : [DOC] Documentation README & Justification des hyperparamètres
- **Labels** : `documentation`
- **Description** :
  Rédiger la documentation professionnelle exigée par le sujet.
- **Tâches** :
  - Description de l'architecture, formules mathématiques (descente de gradient, rétropropagation, activations).
  - Justification des choix d'hyperparamètres (learning rate, taille de batch, fonctions de coût).
  - Guide d'utilisation détaillé avec exemples.
- **Critères d'acceptation** :
  - README clair, illustré et complet.

---

### Issue #18 : [BENCH] Benchmarks et courbes d'apprentissage visuelles
- **Labels** : `benchmarks`, `visualization`
- **Description** :
  Créer `scripts/benchmark.py` pour générer des métriques et graphiques professionnels.
- **Tâches** :
  - Tracé des courbes de loss (train vs validation) au fil des époques.
  - Tracé des courbes d'accuracy.
  - Matrice de confusion pour les 3 classes (`Nothing`, `Check`, `Checkmate`).
  - Comparaison d'optimiseurs (SGD vs Adam) et d'initialisations (Random vs He).
- **Critères d'acceptation** :
  - Graphiques exportés en PNG dans `docs/benchmarks/`.

---

### Issue #19 : [BONUS] Détection des pats (Stalemate) & Distinction par couleur
- **Labels** : `bonus`
- **Description** :
  Implémenter les extensions optionnelles prévues dans le sujet.
- **Tâches** :
  - Détection du `Stalemate`.
  - Prédiction de la couleur gagnante / en échec (`Checkmate Black`, `Check White`, etc.).
  - Stocker les fichiers et Makefile spécifique dans le répertoire `bonus/`.
- **Critères d'acceptation** :
  - Dossier `bonus/` autonome et documenté.
