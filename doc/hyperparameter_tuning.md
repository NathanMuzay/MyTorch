# Optimisation des Hyperparamètres par Métaheuristique (Algorithme Génétique)

Ce rapport documente la recherche automatisée des meilleurs hyperparamètres pour l'architecture **MyTorch**, conformément aux exigences des pages 7 et 8 du sujet (*Hyperparameter Optimization and Metaheuristics*).

## Espace de Recherche Exploré

- **Taux d'apprentissage (\(lr\))** : `0.0005`, `0.001`, `0.002`, `0.005`
- **Taille de lot (*batch size*)** : `16`, `32`, `64`
- **Architecture cachée** : Couche 1 (`64`, `128`, `192`), Couche 2 (`32`, `64`, `96`)
- **Taux de Dropout** : `0.0`, `0.1`, `0.2`
- **Optimiseurs** : `Adam`, `RMSprop`, `SGD` (avec momentum)
- **Initialisation** : `He`, `Xavier`

## Meilleure Configuration Découverte

- **Validation Accuracy** : **63.89%**
- **Optimiseur** : `rmsprop`
- **Taux d'apprentissage (\(lr\))** : `0.002`
- **Taille de lot** : `64`
- **Couches Cachées** : `192 → 32`
- **Taux de Dropout** : `0.1`
- **Initialiseur de poids** : `xavier`

> **Justification théorique :** L'optimiseur `rmsprop` combiné à l'initialisation `xavier` et à un taux de dropout de `0.1` permet une vitesse de convergence optimale tout en maintenant la variance de généralisation sous contrôle.
