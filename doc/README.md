# Apprendre à construire MY_TORCH

Le [nettoyage des données](datacleaning.md) documente `src/data/cleaning.py`, ses choix
et son lancement avec `python3 src/my_torch.py -clean`.

Le guide [entraînement et évaluation du réseau](neural_network.md) décrit
les modes `-train` et `-eval` de `src/my_torch.py`, reliés à la bibliothèque `src/neural_network/`.

Le [rapport d'entraînement](training_report.md) conserve les essais successifs,
analyse le surapprentissage et détaille les corrections et leur vérification.
Il compare notamment la référence à 75,27 % au modèle tactique à **94,26 %**
sur les mêmes 9 000 positions de validation.

Commence par [le cours complet](cours_reseau_neuronal.md). Il part des opérations d'un neurone, construit l'entraînement, puis traite les échiquiers, les expériences et la justification des choix.

Le compagnon [XOR sans bibliothèque externe](exemples/xor_sans_lib.py) est exécutable depuis la racine du projet :

```sh
python3 doc/exemples/xor_sans_lib.py
```

Il montre un réseau réellement entraîné, une vérification numérique de ses gradients et une sauvegarde/relecture. Il sert d'exercice pédagogique ; ce n'est pas l'analyseur d'échecs demandé par le sujet.

Après chaque chapitre, réalise l'exercice et vérifie le résultat attendu. Termine par les tableaux de décisions et les questions de soutenance : comprendre une formule ne suffit pas pour justifier une architecture.

---

## Documents Clés pour la Soutenance et l'Évaluation

* [Rapport d'Évolution des Algorithmes](evolution_report.md) : Historique des 4 itérations, justifications des choix, matrice de confusion et preuves visuelles des gains de précision (35% à >75%).
* [Guide de Soutenance](defense_guide.md) : Déroulé minute par minute, justifications scientifiques et questions/réponses types pour le jury.
* [Rapport de Benchmarks](benchmarks.md) : Courbes d'apprentissage vectorielles SVG et analyses de convergence.
* [Optimisation par Algorithmes Génétiques](hyperparameter_tuning.md) : Recherche automatisée des meilleurs hyperparamètres.


