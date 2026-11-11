# Données FEN

Toutes les données sont regroupées ici, séparément du code Python de `src/`.

- `examples/` : petits corpus versionnés utilisés par les tests et les exemples
  CLI. `sample_chessboards.txt` contient des FEN sans annotation et
  `train_chessboards.txt` des positions annotées en trois classes. Ce dernier
  sert aussi de corpus de secours quand le fichier d'entraînement par défaut
  `cleaned/train.txt` est absent.
- `check/`, `checkmate/`, `nothing/` : corpus bruts locaux, ignorés par Git.
- `for_train/` : fichiers sources d'entraînement, de validation et de test.
- `cleaned/` : résultats du nettoyage, dont `train.txt`, `validation.txt`,
  `test.txt` et le corpus global `cleaned.txt`.

Les noms des fichiers de `for_train/` sont conservés par le nettoyage.
Le nettoyage exige un nouveau dossier de sortie :

```bash
python3 src/my_torch.py -clean --input dataset --output dataset/cleaned_v2
```

Les petits exemples permettent de vérifier les commandes ; ils ne remplacent
pas un corpus d'entraînement et de test représentatif.
