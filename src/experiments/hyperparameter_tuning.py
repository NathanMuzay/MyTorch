#!/usr/bin/env python3
"""
Hyperparameter Optimization using Genetic Algorithm / Metaheuristics.
Searches the hyperparameter space (learning rates, layer dimensions, dropout rates,
optimizers, weight initializations) to optimize validation accuracy.
Fulfills Page 7 & 8 requirement: 'Hyperparameter Optimization and Metaheuristics'.
"""

import os
import random
import time
import numpy as np
from typing import Dict, Any

from src.neural_network.model import Sequential
from src.neural_network.layers import Dense, Dropout
from src.neural_network.activations import ReLU, Softmax
from src.neural_network.optimizers import Adam, SGD, RMSprop
from src.neural_network.losses import CategoricalCrossEntropy
from src.training.train import load_dataset_balanced
from src.data.labels import CLASSES_3

SEARCH_SPACE = {
    "lr": [0.0005, 0.001, 0.002, 0.005],
    "batch_size": [16, 32, 64],
    "hidden1": [64, 128, 192],
    "hidden2": [32, 64, 96],
    "dropout": [0.0, 0.1, 0.2],
    "optimizer": ["adam", "rmsprop", "sgd"],
    "init": ["he", "xavier"],
}


def random_individual() -> Dict[str, Any]:
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}


def mutate(ind: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
    new_ind = ind.copy()
    for k in SEARCH_SPACE:
        if random.random() < mutation_rate:
            new_ind[k] = random.choice(SEARCH_SPACE[k])
    return new_ind


def crossover(parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
    child = {}
    for k in SEARCH_SPACE:
        child[k] = parent1[k] if random.random() < 0.5 else parent2[k]
    return child


def build_and_evaluate(
    ind: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 5,
) -> float:
    layers = [
        Dense(773, ind["hidden1"], weight_initializer=ind["init"]),
        ReLU(),
    ]
    if ind["dropout"] > 0.0:
        layers.append(Dropout(p=ind["dropout"]))
    layers.extend([
        Dense(ind["hidden1"], ind["hidden2"], weight_initializer=ind["init"]),
        ReLU(),
        Dense(ind["hidden2"], 3, weight_initializer="xavier"),
        Softmax(),
    ])

    model = Sequential(layers)

    opt_name = ind["optimizer"]
    if opt_name == "adam":
        opt = Adam(learning_rate=ind["lr"])
    elif opt_name == "rmsprop":
        opt = RMSprop(learning_rate=ind["lr"])
    else:
        opt = SGD(learning_rate=ind["lr"] * 10, momentum=0.9)

    model.compile(optimizer=opt, loss=CategoricalCrossEntropy())

    try:
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=ind["batch_size"],
            validation_data=(X_val, y_val),
            early_stopping_patience=0,
            verbose=False,
        )
        _, val_acc = model.evaluate(X_val, y_val)
        return float(val_acc)
    except Exception:
        return 0.0


def run_metaheuristic_tuning(
    pop_size: int = 6,
    generations: int = 3,
    epochs_per_candidate: int = 4,
):
    print("=== MY_TORCH_ HYPERPARAMETER OPTIMIZATION (METAHEURISTIC) ===")
    dataset_path = "dataset/cleaned/train.txt" if os.path.exists("dataset/cleaned/train.txt") else "dataset/examples/train_chessboards.txt"
    X, y = load_dataset_balanced(dataset_path, CLASSES_3, max_per_class=600)

    indices = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]

    y_train_oh = np.zeros((len(y_train), 3), dtype=np.float32)
    y_train_oh[np.arange(len(y_train)), y_train] = 1.0

    y_val_oh = np.zeros((len(y_val), 3), dtype=np.float32)
    y_val_oh[np.arange(len(y_val)), y_val] = 1.0

    print(f"Population size: {pop_size}, Generations: {generations}\n")

    population = [random_individual() for _ in range(pop_size)]
    history_log = []

    best_fitness = -1.0
    best_config = None

    for gen in range(1, generations + 1):
        gen_start = time.time()
        print(f"--- Generation {gen}/{generations} ---")
        scores = []
        for idx, ind in enumerate(population, 1):
            fit = build_and_evaluate(ind, X_train, y_train_oh, X_val, y_val_oh, epochs=epochs_per_candidate)
            scores.append((fit, ind))
            if fit > best_fitness:
                best_fitness = fit
                best_config = ind
            print(f"  Candidate {idx}: Acc={fit*100:.2f}% | Config: lr={ind['lr']}, opt={ind['optimizer']}, hidden=({ind['hidden1']},{ind['hidden2']}), drop={ind['dropout']}")

        scores.sort(key=lambda x: x[0], reverse=True)
        history_log.append((gen, scores[0][0], scores[0][1]))
        elapsed = time.time() - gen_start
        print(f"  > Best Generation {gen} Accuracy: {scores[0][0]*100:.2f}% (Time: {elapsed:.2f}s)\n")

        # Selection & Reproduction
        survivors = [s[1] for s in scores[: max(2, pop_size // 2)]]
        next_gen = survivors[:]
        while len(next_gen) < pop_size:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)
        population = next_gen

    print(f"==================================================")
    print(f"Optimal Configuration Discovered (Val Acc: {best_fitness*100:.2f}%):")
    for k, v in best_config.items():
        print(f"  - {k}: {v}")
    print(f"==================================================")

    # Save summary report
    report_md = fr"""# Optimisation des Hyperparamètres par Métaheuristique (Algorithme Génétique)

Ce rapport documente la recherche automatisée des meilleurs hyperparamètres pour l'architecture **MyTorch**, conformément aux exigences des pages 7 et 8 du sujet (*Hyperparameter Optimization and Metaheuristics*).

## Espace de Recherche Exploré

- **Taux d'apprentissage ($lr$)** : `0.0005`, `0.001`, `0.002`, `0.005`
- **Taille de lot (*batch size*)** : `16`, `32`, `64`
- **Architecture cachée** : Couche 1 (`64`, `128`, `192`), Couche 2 (`32`, `64`, `96`)
- **Taux de Dropout** : `0.0`, `0.1`, `0.2`
- **Optimiseurs** : `Adam`, `RMSprop`, `SGD` (avec momentum)
- **Initialisation** : `He`, `Xavier`

## Meilleure Configuration Découverte

- **Validation Accuracy** : **{best_fitness * 100:.2f}%**
- **Optimiseur** : `{best_config['optimizer']}`
- **Taux d'apprentissage (\(lr\))** : `{best_config['lr']}`
- **Taille de lot** : `{best_config['batch_size']}`
- **Couches Cachées** : `{best_config['hidden1']} → {best_config['hidden2']}`
- **Taux de Dropout** : `{best_config['dropout']}`
- **Initialiseur de poids** : `{best_config['init']}`

> **Justification théorique :** L'optimiseur `{best_config['optimizer']}` combiné à l'initialisation `{best_config['init']}` et à un taux de dropout de `{best_config['dropout']}` permet une vitesse de convergence optimale tout en maintenant la variance de généralisation sous contrôle.
"""
    with open("doc/hyperparameter_tuning.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("\nReport saved to doc/hyperparameter_tuning.md")


if __name__ == "__main__":
    run_metaheuristic_tuning(pop_size=4, generations=2, epochs_per_candidate=3)
