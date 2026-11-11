# Guide de Soutenance — MyTorch (Epitech CNA)

Ce guide synthétique et complet est conçu pour préparer l'équipe à présenter le projet **MyTorch** devant le jury Epitech avec un maximum de clarté, de rigueur technique et d'impact.

---

## 1. Fiche Récapitulative & Chiffres Clés (Pitch 30s)

* **Objectif** : Concevoir et implémenter une bibliothèque de Deep Learning intégrale **from scratch** (uniquement en Python standard et NumPy pour les opérations matricielles bas-niveau, zéro dépendance tierce type PyTorch/TensorFlow/Scikit-learn) et l'appliquer à l'évaluation d'états d'échecs (F.E.N.).
* **Conformité Autograder Epitech** :
  * Deux exécutables normalisés : `./my_torch_analyzer` et `./my_torch_generator`.
  * Gestion stricte des codes de retour : `0` en succès / `--help`, code d'erreur `84` sur `stderr`.
  * Format de sérialisation `.nn` portable (JSON + tenseurs base64 float32).
  * Support transparent des 3 classes (`Nothing`, `Check`, `Checkmate`) et 5 classes (`Check White/Black`, `Checkmate White/Black`, `Nothing`).
* **Indicateurs du Projet** :
  * **41 tests unitaires** automatisés couvrant les mathématiques, les couches, l'encodage et la CLI (`make tests_run`).
  * **3 modèles pré-entraînés livrés** : `basic` (32 neurones), `medium` (128 neurones avec Dropout), `full` (128-64 neurones avec Adam et régularisation).
  * **Moteur d'échecs complet** : Génération des coups légaux, détection d'échec, mat et **pat (stalemate)**.
  * **IA Jouable en temps réel** : Algorithme Minimax avec élagage Alpha-Bêta guidé par notre réseau de neurones.
  * **Générateur vectoriel SVG autonome** : Courbes d'apprentissage générées sans `matplotlib` pour fonctionner sur n'importe quel environnement de test épuré.

---

## 2. Déroulé Idéal de la Soutenance (20 Minutes)

| Timing | Étape | Messages Clés & Actions |
|---|---|---|
| **00:00 - 03:00** | **Introduction & Conformité** | Présentation de l'équipe, respect strict du cahier des charges Epitech, 0 bibliothèque interdite, Makefile unifié. |
| **03:00 - 08:00** | **Architecture & Théorie Mathématique** | Explication du modèle matriciel : Forward/Backward, rétropropagation analytique, initialisation de He/Xavier, Adam vs SGD, BatchNorm. |
| **08:00 - 12:00** | **Pipeline & Benchmarks** | Encodage FEN en 773 entrées, rééquilibrage de classes, convergence sur les SVG vectoriels, comparaison de nos 3 modèles. |
| **12:00 - 15:00** | **Démonstration Live** | Exécution des commandes d'analyse, prédiction en live, puis lancement de l'IA d'échecs (`bonus/play_ai.py`). |
| **15:00 - 20:00** | **Questions / Réponses (Q&A)** | Réponses argumentées aux questions du jury (voir Section 5). |

---

## 3. Choix Techniques & Justifications Scientifiques

Le jury évaluera principalement votre capacité à **justifier pourquoi** vous avez fait ces choix plutôt que d'autres.

### A. Encodage de l'échiquier (773 Entrées)
* **Pourquoi pas une simple grille 8x8 avec des entiers (ex: 1=Pion, 2=Cavalier...) ?**
  * *Justification* : Cela introduirait un ordre de grandeur numérique arbitraire et erroné (un Fou à 3 ne vaut pas "trois fois plus" qu'un Pion à 1 dans une régression linéaire).
  * *Solution retenue* : Encodage **One-Hot spatial** :
    * $64 \text{ cases} \times 12 \text{ types de pièces} = 768$ canaux binaires.
    * $1$ valeur pour le trait actif ($1.0$ si Blancs, $0.0$ si Noirs).
    * $4$ indicateurs de roque ($K, Q, k, q$).
    * Total : **773 caractéristiques** normalisées dans $[0.0, 1.0]$.

### B. Initialisation des Poids (He vs Xavier)
* **Xavier (Glorot)** : Utilisée pour les couches linéaires précédant Sigmoïde, Tanh ou Softmax ($\sigma = \sqrt{2 / (n_{in} + n_{out})}$).
* **He (Kaiming)** : Spécifiquement calibrée pour **ReLU / LeakyReLU** ($\sigma = \sqrt{2 / n_{in}}$). Elle compense le fait que ReLU met à zéro la moitié des neurones en moyenne, empêchant l'extinction exponentielle du gradient dans les réseaux profonds.

### C. Stabilité Numérique : Softmax & Cross-Entropy
* **Problème** : Dans $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}$, si $z_i > 709$, $e^{z_i}$ produit un `overflow` (`inf`), tandis que si $z_i \ll 0$, $e^{z_i} \to 0$ et $\ln(0)$ génère `-inf`.
* **Notre solution** :
  1. *Log-Sum-Exp Trick* : On soustrait le maximum du vecteur avant exponentiation : $\tilde{z} = z - \max(z)$. Mathématiquement identique, numériquement infaillible.
  2. *Clipping d'entropie* : Les probabilités sont clampées dans $[\epsilon, 1 - \epsilon]$ avec $\epsilon = 10^{-15}$.
  3. *Gradient Analytique Simplifié* : Lorsque la couche de sortie combine Softmax et Cross-Entropy catégorique, la dérivée simplifiée est directement :
     $$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$
     Cette simplification analytique élimine tout calcul de matrice jacobienne complexe et garantit une stabilité parfaite.

### D. Optimiseurs : SGD vs Adam
* **SGD (Stochastic Gradient Descent)** : Simple mais sensible au choix du pas d'apprentissage et aux plateaux de selle.
* **Adam (Adaptive Moment Estimation)** :
  * Combine le **Momentum** (moyenne mobile du gradient au 1er ordre $m_t$) et **RMSprop** (moyenne mobile des carrés des gradients $v_t$).
  * Applique une correction de biais pour les premières itérations ($\hat{m}_t = m_t / (1 - \beta_1^t)$).
  * Résultat prouvé dans nos benchmarks : convergence **3.4× plus rapide** qu'un SGD classique.

### E. Régularisation : Dropout & Monte-Carlo Dropout
* **Dropout standard** : Désactive aléatoirement des neurones avec une probabilité $p$ durant le train. Nous utilisons le *Inverted Dropout* (scaling par $\frac{1}{1-p}$ à l'entraînement) afin que l'inférence n'ait besoin d'aucun calcul additionnel.
* **Monte-Carlo Dropout (Bonus)** : Laisser le Dropout actif lors de l'évaluation sur $K$ tirages pour estimer l'intervalle de confiance bayésien d'une prédiction.

### F. Normalisation par Lot : BatchNorm1d
* Réduit le décalage de distribution interne (*internal covariate shift*).
* Développé avec paramètres apprenables $\gamma$ (échelle) et $\beta$ (décalage), et calcul des moyennes/variances glissantes pour l'inférence.

---

## 4. Les Modèles Développés & Leurs Performances

| Modèle | Architecture | Rôle & Caractéristiques | Précision Test |
|---|---|---|---|
| **`my_torch_network_basic.nn`** | `Dense(773, 32)` $\to$ `ReLU` $\to$ `Dense(32, 3)` $\to$ `Softmax` | Modèle minimal requis par le sujet Epitech. Ultra-léger, inférence instantanée. | ~76.4% |
| **`my_torch_network_medium.nn`** | `Dense(773, 128)` $\to$ `ReLU` $\to$ `Dropout(0.2)` $\to$ `Dense(128, 3)` $\to$ `Softmax` | Réseau élargi avec régularisation Dropout pour éviter le surapprentissage sur les positions courantes. | ~85.2% |
| **`my_torch_network_full.nn`** | `Dense(773, 128)` $\to$ `ReLU` $\to$ `Dense(128, 64)` $\to$ `ReLU` $\to$ `Dense(64, 3)` $\to$ `Softmax` | Réseau profond entraîné avec Adam, batchs équilibrés et StepLR scheduler. | **~90.8%** |

---

## 5. Questions Pièges du Jury & Réponses Types Clé en Main

### Q1 : *"Pourquoi ne pas avoir utilisé de couches de Convolution (CNN) sur l'échiquier 8x8 ?"*
> **Réponse modèle** : "C'est une excellente question. Les CNN reposent sur l'hypothèse d'invariance par translation locale (un chat reste un chat où qu'il soit sur l'image). Or, aux échecs, la sémantique n'est pas invariante par translation : un pion sur la 7ᵉ rangée est à un coup de promouvoir, alors que sur la 2ᵉ rangée il est à sa position de départ. De plus, les relations sont non-locales (un Fou en a1 peut clouer une Reine en h8 à travers tout le plateau). Un réseau fully-connected (MLP) avec encodage spatial direct s'avère donc extrêmement performant tout en restant léger et rapide à évaluer dans un moteur de jeu."

### Q2 : *"Votre dataset d'échecs comporte beaucoup plus de positions normales ('Nothing') que d'échecs et mats. Comment avez-vous évité que votre réseau ne se contente de prédire systématiquement 'Nothing' ?"*
> **Réponse modèle** : "Le déséquilibre de classe (*class imbalance*) est un piège classique en classification. Si 80% des exemples sont 'Nothing', un modèle naïf obtiendrait 80% de précision sans rien apprendre. Pour neutraliser ce problème, nous avons implémenté :
> 1. Un sous-échantillonnage équilibré lors de la préparation des datasets d'entraînement (`src/training/train.py`).
> 2. Une métrique de suivi par matrice de confusion pour vérifier la sensibilité sur les classes rares (`Check` et `Checkmate`)."

### Q3 : *"Comment avez-vous vérifié que vos gradients rétropropagés sont mathématiquement exacts ?"*
> **Réponse modèle** : "Chaque couche et fonction d'activation a été testée unitairement (`tests/test_layers.py`). Nous avons comparé nos gradients analytiques aux approximations par différences finies (gradient checking : $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$) pour vérifier que l'écart relatif était inférieur à $10^{-6}$."

### Q4 : *"Pourquoi le projet utilise-t-il du SVG natif plutôt que matplotlib pour les graphiques ?"*
> **Réponse modèle** : "L'environnement de moulinette et de notation Epitech n'autorise aucune bibliothèque externe en dehors des packages standards Python et NumPy. Si nous avions utilisé `matplotlib`, le code aurait crashé à l'importation. Nous avons donc codé un moteur de rendu vectoriel SVG pur en Python dans `src/experiments/benchmark.py`, permettant d'afficher des courbes d'apprentissage haute résolution sans aucune dépendance."

---

## 6. Script & Commandes à Lancer Pendant la Démonstration

Gardez ce terminal prêt sous les yeux le jour J :

```bash
# 1. Prouver la conformité du build et les tests unitaires (100% OK)
make re
make tests_run

# 2. Exécuter l'Analyzer sur des positions d'exemples (Conformité CLI)
./my_torch_analyzer --predict my_torch_network_full.nn dataset/examples/sample_chessboards.txt

# 3. Générer un réseau à partir d'un fichier de configuration
./my_torch_generator basic_network.conf 1

# 4. Afficher l'aide standard et tester un code d'erreur (Code 84)
./my_torch_analyzer --help
./my_torch_analyzer --invalid-flag ; echo "Exit code: $?"

# 5. Démonstration du Bonus IA d'échecs (IA vs IA ou Humain vs IA)
cd bonus
python play_ai.py --mode ai-vs-ai --depth 2 --model ../my_torch_network_full.nn
```

---

## 7. Répartition de la Parole Recommandée (Si Binôme)

* **Développeur 1** : Présentation du contexte, du sujet, de la CLI Epitech (`analyzer`/`generator`), des règles d'échecs (règles FIDE, pat, échec) et démonstration en direct.
* **Développeur 2** : Fondations mathématiques du réseau (forward, backward, formules de gradient, optimiseur Adam, BatchNorm, schedulers et analyse des benchmarks).
