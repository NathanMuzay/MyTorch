# Fiche de Synthèse pour la Soutenance — MY_TORCH_

> **Projet :** G-CNA-501 — MyTorch (Epitech Promo 2026)  
> **Auteurs :** Esteban Kenzi & Nathan Muzay  
> **Objectif :** Framework de Deep Learning *from scratch* (zéro framework externe) et classifieur d'états d'échecs FEN (*Nothing*, *Check*, *Checkmate*).

---

## 1. Pitch Synthétique (30 secondes d'intro)

*« MyTorch est un framework complet de Deep Learning développé intégralement from scratch en Python et NumPy, sans PyTorch, TensorFlow ou Scikit-Learn. Nous avons implémenté l'ensemble de la chaîne mathématique — de la rétropropagation du gradient matricielle jusqu'à l'optimiseur Adam et la BatchNorm. Pour valider notre framework, nous l'avons appliqué à la prédiction de situations d'échecs complexes à partir de positions FEN, atteignant près de 76% de précision de validation grâce à des innovations comme la Focal Loss multi-classes et l'augmentation par symétrie de couleur. »*

---

## 2. Vue d'Ensemble & Architecture Modulaire (`src/`)

Le projet est strictement partitionné pour respecter les principes SOLID et la conformité Epitech :

```
G-CNA-501-MAR-5-1-mytorch-1/
├── Makefile                     # Build unifié (my_torch_analyzer, my_torch_generator, visu)
├── src/
│   ├── neural_network/          # Cœur du framework Deep Learning
│   │   ├── model.py             # Conteneur Sequential (forward, backward, early stopping)
│   │   ├── layers.py            # Dense, Dropout, BatchNorm1d
│   │   ├── activations.py       # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax stable
│   │   ├── losses.py            # CategoricalCrossEntropy, FocalLoss, MSE, BCE
│   │   ├── optimizers.py        # Adam (moments adaptatifs), SGD (momentum), RMSprop
│   │   ├── schedulers.py        # StepLR, CosineAnnealingLR, ExponentialLR
│   │   └── initializers.py      # HeNormal (Kaiming), Xavier (Glorot), LeCun
│   ├── chess_engine/            # Moteur d'échecs déterministe & règles FIDE
│   │   ├── fen_parser.py        # Parsing robuste et validation de chaînes FEN
│   │   ├── encoders.py          # Encodage spatial One-Hot en 773 entrées
│   │   └── rules.py             # Détection échec, mat, pat, roque, clouages
│   ├── data/                    # Pipeline de données
│   │   ├── cleaning.py          # Nettoyage et filtrage de positions valides
│   │   ├── augmentation.py      # Symétrie miroir & inversion des couleurs
│   │   └── labels.py            # Gestion des classes (3 classes ou 5 avec couleur)
│   ├── training/                # Entraînement et métriques
│   │   ├── train.py             # Boucle d'entraînement, stratification, early stopping
│   │   └── evaluate.py          # Matrice de confusion, F1-score macro, Precision, Recall
│   ├── cli/                     # Points d'entrée Epitech conformes
│   │   ├── analyzer.py          # my_torch_analyzer (--train, --predict)
│   │   └── generator.py         # my_torch_generator (génération depuis .conf)
│   └── experiments/             # Benchmarks & tracés vectoriels SVG sans matplotlib
├── bonus/                       # Bonus à fort impact
│   ├── visualize.py             # Visualiseur d'échiquier Tkinter & export SVG autonome
│   ├── play_ai.py               # IA jouable Minimax Alpha-Bêta guidée par le réseau
│   └── genetic_tuner.py         # Optimisation d'hyperparamètres par algorithme génétique
└── tests/                       # Suite de 59 tests unitaires (couverture complète)
```

---

## 3. Détails Techniques & Justifications Scientifiques

### A. Représentation du Plateau (Feature Engineering — 773 Caractéristiques)
* **Pourquoi pas une matrice 8x8 d'entiers (1=Pion, 2=Cavalier...) ?**
  * Une numérotation arbitraire crée une fausse relation d'ordre métrique qui fausse la régression linéaire.
* **La solution : Encodage spatial One-Hot normalisé dans $[0.0, 1.0]$ :**
  1. **64 cases $\times$ 12 types de pièces** (6 blanches, 6 noires) = **768 canaux binaires**.
  2. **1 valeur pour le joueur actif** ($1.0$ si Blancs, $0.0$ si Noirs).
  3. **4 indicateurs de droits de roque** ($K, Q, k, q$).
  * Total strict : **773 entrées**.

### B. Innovations du Pipeline de Données
1. **Augmentation par Symétrie de Couleur (`src/data/augmentation.py`)** :
   * Chaque position d'échecs admet une position jumelle inversée (inversion Blanc/Noir des pièces, permutation des rangées 1-8 et inversion du trait).
   * **Gain :** Doublement virtuel du dataset sans risque d'incohérence tactique et neutralisation du biais de couleur.
2. **Stratified Split & Reservoir Sampling** :
   * Découpage train/validation garantissant exactement le même ratio de classes rares (*Checkmate*) dans chaque split.
3. **Encodage Multicœur Parallèle** :
   * Traitement via `ThreadPoolExecutor` réduisant le temps d'encodage de 30 000 positions de **5.2s à moins de 2s**.

### C. Fondations Mathématiques du Réseau
1. **Rétropropagation Analytique Simplifiée :**
   * Pour la combinaison `Softmax` + `CategoricalCrossEntropy`, la matrice jacobienne se simplifie élégamment en :
     $$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$
     Cette simplification évite les instabilités numériques et accélère le calcul.
2. **Stabilité Numérique (Log-Sum-Exp Trick) :**
   * Avant l'exponentielle dans le Softmax, on soustrait $\max(z)$ pour éviter tout overflow (`inf`).
   * Les probabilités sont clampées dans $[\epsilon, 1 - \epsilon]$ avec $\epsilon = 10^{-15}$.
3. **Initialisations Adaptées :**
   * **He (Kaiming)** ($\sigma = \sqrt{2 / n_{\text{in}}}$) pour les couches suivies de ReLU/LeakyReLU (évite l'extinction exponentielle du gradient).
   * **Xavier (Glorot)** pour les couches de sortie linéaires/Softmax.
4. **Normalisation Interne (`BatchNorm1d`) :**
   * Paramètres apprenables $\gamma$ et $\beta$, moyennes/variances mobiles pour l'inférence. Réduit l'*Internal Covariate Shift*.
5. **Optimiseur Adam :**
   * Combine Momentum (1er ordre) et RMSprop (2d ordre) avec correction de biais ($\hat{m}_t, \hat{v}_t$). Convergence **3.4× plus rapide** que SGD.
6. **Planificateurs de Pas (`CosineAnnealingLR`) :**
   * Décroissance cosinus permettant de sortir des cols pour converger vers un minimum global stable.

### D. Gestion du Déséquilibre & Affinement des Pertes
1. **Multi-Class Focal Loss** :
   $$\mathcal{L}_{\text{Focal}} = - \alpha_t (1 - p_t)^\gamma \ln(p_t) \quad (\gamma = 2.0)$$
   * 60% des erreurs provenaient de la frontière floue entre *Check* et *Checkmate*.
   * La Focal Loss écrase le gradient des positions évidentes ($p_t \approx 0.95$) de **99.75%** et concentre le gradient sur les cas litigieux.
2. **Label Smoothing** :
   * $y_{\text{smooth}} = (1 - \epsilon)y + \frac{\epsilon}{K}$ avec $\epsilon = 0.05$ à $0.1$.
   * Empêche le modèle de devenir surconfiant et d'exploser les poids sur les cas limites.

---

## 4. Évolution des Performances (Les 4 Phases de Développement)

| Métrique | Phase 1 (Baseline) | Phase 2 (Dropout + He) | Phase 3 (BatchNorm + Cosine) | Phase 4 (Deep-Wide + Focal) |
|---|:---:|:---:|:---:|:---:|
| **Architecture** | `Dense(773, 32)` | `Dense(773, 128)` | `Dense(256) + Dense(128)` | `Dense(512) + Dense(256) + Dense(128)` |
| **Paramètres** | 24 835 | 107 523 | 248 197 | **560 389** |
| **Fonction de Perte** | Cross-Entropy | Cross-Entropy | Weighted Cross-Entropy | **Multi-Class Focal Loss** |
| **Précision Test** | **35.5 %** | **50.2 %** | **66.2 %** | **69.8 %** (5 classes) |
| **Précision Validation**| **41.2 %** | **54.8 %** | **71.9 %** | **75.9 %** |
| **Détection du Mat** | **23.9 %** | **52.0 %** | **70.9 %** | **75.7 %** |

---

## 5. Les Bonus Réalisés (Différenciateurs Majeurs)

1. **Visualiseur Autonome Tkinter & Export SVG** (`visu` / `bonus/visualize.py`) :
   * Visualisation interactive de n'importe quel fichier de positions FEN.
   * Génération de fichiers SVG purs sans dépendre de `matplotlib` (moulinette-proof).
2. **IA d'Échecs Jouable en Temps Réel** (`bonus/play_ai.py`) :
   * Algorithme Minimax avec élagage Alpha-Bêta couplé à l'évaluation du réseau de neurones. Modes IA vs IA et Humain vs IA.
3. **Métaheuristique Génétique** (`bonus/genetic_tuner.py`) :
   * Optimisation automatique des hyperparamètres (learning rate, dropout, dimensions de couches).

---

## 6. Plan de Présentation Soutenance (20 Minutes Chrono)

| Timing | Étape | Intervenant & Contenu |
|---|---|---|
| **00:00 - 03:00** | **Introduction & Conformité** | Contexte du projet, respect des normes Epitech, 0 lib externe, Makefile et code 84. |
| **03:00 - 08:00** | **Théorie Mathématique & Core** | Forward/Backward matriciel, He initialization, Adam, BatchNorm1d, stabilité numérique. |
| **08:00 - 12:00** | **Data Engineering & Modèles** | Encodage 773 features, augmentation miroir, Focal Loss et gains chiffrés des 4 phases. |
| **12:00 - 15:00** | **Démonstration Live** | Exécution de `analyzer`, `generator`, tests et lancement de l'échiquier / IA jouable. |
| **15:00 - 20:00** | **Questions & Réponses (Q&A)** | Réponses argumentées aux questions du jury. |

---

## 7. Commandes Clés pour la Démonstration Live

```bash
# 1. Compilation et validation des tests
make re
make tests_run

# 2. Prédiction avec le modèle final haute performance
./my_torch_analyzer --predict my_torch_network_full.nn dataset/examples/sample_chessboards.txt

# 3. Génération d'une architecture depuis un fichier .conf
./my_torch_generator basic_network.conf 1

# 4. Démonstration de la gestion des erreurs (Code 84 obligatoire Epitech)
./my_torch_analyzer ; echo "Exit code: $?"
./my_torch_analyzer --invalid-flag ; echo "Exit code: $?"

# 5. Démonstration du visualiseur SVG / interface
./visu dataset/examples/sample_chessboards.txt -1

# 6. Démonstration de l'IA jouable en direct
python bonus/play_ai.py --mode ai-vs-ai --depth 2 --model my_torch_network_full.nn
```

---

## 8. Réponses Clés aux Questions Pièges du Jury

* **Q : Pourquoi un MLP plutôt qu'un CNN pour un échiquier ?**  
  * *R :* Les CNN reposent sur l'invariance par translation spatiale locale. Or aux échecs, la position n'est pas invariante (un pion en 2e rangée n'a rien à voir avec un pion en 7e rangée), et les interactions sont globales (un Fou en a1 peut attaquer une pièce en h8). Un MLP sur 773 entrées est donc optimal et beaucoup plus rapide.
* **Q : Comment avez-vous garanti l'exactitude de vos gradients ?**  
  * *R :* Par Gradient Checking numérique (différences finies centrées : $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$). Tous nos tests unitaires valident un écart relatif $< 10^{-6}$.
* **Q : Comment traitez-vous le déséquilibre de classe ?**  
  * *R :* Stratified splitting, Focal Loss ($\gamma = 2.0$) pour focaliser sur les frontières dures, et augmentation par symétrie de couleur.
