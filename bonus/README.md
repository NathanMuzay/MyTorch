# 🏆 Bonus Directory - MY_TORCH_ < A CLASH OF KINGS />

Ce répertoire regroupe l'ensemble des fonctionnalités et extensions avancées développées pour le projet **MY_TORCH_**, conformément aux spécifications du sujet EPITECH (page 2 : *« All the bonus files (including a potential specific Makefile) should be in a directory named bonus »*).

---

## 📋 Sommaire des Bonus Développés

1. [♟️ IA d'Échecs Jouable Complète (`play_ai.py`)](#1-️-ia-déchecs-jouable-complète-play_aipy)
2. [👁️ Visualiseur Graphique Interactif & Export SVG (`visualize.py`)](#2-️-visualiseur-graphique-interactif--export-svg-visualizepy)
3. [⚖️ Détection et Gestion du Pat (*Stalemate*)](#3-️-détection-et-gestion-du-pat-stalemate)
4. [🧬 Optimisation d'Hyperparamètres par Métaheuristique](#4--optimisation-dhyperparamètres-par-métaheuristique)
5. [🎨 Classification Fine par Couleur (5 Classes)](#5--classification-fine-par-couleur-5-classes)
6. [🛠️ Makefile Dédié](#6-️-makefile-dédié)

---

## 1. ♟️ IA d'Échecs Jouable Complète (`play_ai.py`)

Une intelligence artificielle complète permettant de jouer aux échecs contre l'ordinateur dans le terminal :
- **Algorithme Minimax avec élagage Alpha-Bêta** (recherche en profondeur paramétrable).
- **Évaluation hybride** combinant :
  - La valeur matérielle et le positionnement des pièces.
  - L'évaluation tactique en temps réel fournie par le réseau de neurones pré-entraîné (`my_torch_network_full.nn` ou `medium.nn`) pour détecter les menaces d'échec et de mat.
- **Affichage dynamique de l'échiquier** en console (compatible ASCII et glyphes Unicode).

### Utilisation :
```bash
# Lancer une partie interactive (joueur Blanc par défaut)
python bonus/play_ai.py

# Jouer avec les Noirs contre l'IA
python bonus/play_ai.py --color b

# Choisir la profondeur de recherche (ex: profondeur 3)
python bonus/play_ai.py --depth 3

# Démonstration automatique (l'IA joue contre elle-même pendant 4 coups)
python bonus/play_ai.py --auto 4
```

---

## 2. 👁️ Visualiseur Graphique Interactif & Export SVG (`visualize.py`)

Outil de rendu graphique de positions FEN :
- Affichage dans une fenêtre Tkinter dédiée avec indication du joueur au trait et de l'annotation.
- Export vectoriel propre au format **SVG**.

### Utilisation :
```bash
# Visualiser la 30e position d'un fichier de test
python bonus/visualize.py dataset/cleaned/test.txt 30

# Exporter en SVG sans ouvrir la fenêtre
python bonus/visualize.py dataset/cleaned/test.txt 30 --output /tmp/position.svg --no-open
```

---

## 3. ⚖️ Détection et Gestion du Pat (*Stalemate*)

Spécifié en page 6 du sujet : *« Handling stalemates is considered a bonus, as there are quite few configurations »*.
- Intégration dans le moteur de règles [src/chess_engine/rules.py](../src/chess_engine/rules.py) de la fonction `is_stalemate()`.
- Détecte lorsqu'un joueur au trait n'a aucun coup légal possible mais que son roi n'est pas attaqué.
- Couvert par les tests unitaires dans `tests/test_rules.py`.

---

## 4. 🧬 Optimisation d'Hyperparamètres par Métaheuristique

Conforme à la page 7 & 8 du sujet (*« Hyperparameter Optimization and Metaheuristics »*) :
- Script [src/experiments/hyperparameter_tuning.py](../src/experiments/hyperparameter_tuning.py) basé sur un **Algorithme Génétique**.
- Explore l'espace combinatoire (taux d'apprentissage, tailles de lots, dimensions des couches cachées, taux de dropout, optimiseurs Adam/SGD/RMSprop, initialisations He/Xavier).
- Sélectionne les meilleurs individus par tournoi, croisement (*crossover*) et mutation stochastique.
- Rapport consigné dans [doc/hyperparameter_tuning.md](../doc/hyperparameter_tuning.md).

---

## 5. 🎨 Classification Fine par Couleur (5 Classes)

Au-delà de la classification classique en 3 états (`Nothing`, `Check`, `Checkmate`), le modèle complet [my_torch_network_full.nn](../my_torch_network_full.nn) distingue la couleur qui porte l'attaque :
- `Nothing`
- `Check White`
- `Check Black`
- `Checkmate White`
- `Checkmate Black`

---

## 6. 🛠️ Makefile Dédié

Conformément à la recommandation EPITECH (page 2) :
```bash
cd bonus/

# Compiler les binaires 'play' et 'visu'
make

# Lancer le test d'intégrité de l'IA
make test

# Nettoyer les exécutables bonus
make fclean
```
