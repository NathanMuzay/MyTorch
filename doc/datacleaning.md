# Nettoyage des données d'échecs

`src/data/cleaning.py` contient uniquement le nettoyage : aucun réseau, entraînement,
encodage numérique ou découpage train/validation/test. `src/my_torch.py` appelle le nettoyage uniquement avec l’option `-clean`. Le code utilise uniquement la bibliothèque standard Python 3.10+.

## Utilisation

Architecture :

```text
src/
    data/cleaning.py  # Validation, normalisation et déduplication
    my_torch.py    # Point d’entrée : nettoyage, entraînement et évaluation
bonus/
    visualize.py  # Visualiseur de positions FEN
```

Depuis la racine du dépôt :

```sh
python3 src/my_torch.py -clean --input dataset --output dataset/cleaned
```

L’exécution sans option affiche l’aide. Avec `-clean`, `--input` et `--output`
permettent de choisir les chemins, relatifs au répertoire de lancement.
`python3 -m src.my_torch -clean` est également pris en charge.

La commande crée un **nouveau** dossier. Elle refuse un dossier de sortie déjà
existant, même vide, pour éviter d'écraser un résultat. Pour relancer, choisir
un autre chemin. En cas d'erreur d'accès disque, une sortie partielle peut
subsister : utiliser également un nouveau dossier après correction.
Les erreurs de lancement renvoient 84 ; les lignes rejetées sont une issue
normale du nettoyage et sont comptées dans le bilan retourné en mémoire.

Fichiers produits :

| Fichier | Contenu et utilité |
| --- | --- |
| `cleaned.txt` | Une FEN normalisée et son annotation par ligne, au format des sources. |
| `train.txt`, `test.txt`, `validation.txt`, etc. | Version nettoyée de chaque fichier `*.txt` présent directement dans `for_train/`, avec son nom d'origine. |

Seuls les `.txt` sont écrits. Les compteurs restent disponibles dans le
dictionnaire retourné par `clean_dataset()` ; la commande affiche le nombre de
positions conservées dans chaque fichier. Aucun JSON, JSONL ou journal de
provenance n'est produit, pour garder uniquement les données utiles au réseau.

## Choix, raisons et possibilités d'évolution

### 1. Séparer les sources et les résultats

Les fichiers `*.txt` directement dans `check/`, `checkmate/` et `nothing/`
alimentent le corpus global `cleaned.txt`. Les fichiers `*.txt` directement dans
`for_train/` sont nettoyés séparément, avec les mêmes règles, puis écrits à la
racine du dossier de sortie sous leur nom d'origine. Ils ne sont pas ajoutés au
corpus global : cela évite de compter deux fois les mêmes sources.
Les sous-dossiers ne sont pas parcourus récursivement. L'annotation de la ligne
fait foi ; le nom du dossier n'est pas utilisé pour corriger une classe.

Le découpage existant est conservé : aucune position n'est déplacée entre
entraînement, validation et test. La déduplication et l'exclusion des conflits
s'effectuent **dans chaque fichier** de `for_train/`, indépendamment des autres.
Les éventuels doublons ou conflits entre ces jeux ne sont donc pas éliminés.
Pour une future évaluation du modèle, un contrôle des chevauchements entre jeux
restera nécessaire. Les compteurs de ces fichiers figurent sous `for_train` dans
le bilan en mémoire ; les compteurs globaux habituels concernent `cleaned.txt`.

Les fichiers sources ne sont jamais modifiés. La sortie ne peut pas se trouver
dans l'un de ces dossiers ni dans `for_train/`. Pour un projet plus grand, cette
séparation permettra de reconstruire un jeu nettoyé et de comparer plusieurs
versions de la politique de nettoyage en conservant les mêmes données brutes.

### 2. Définir un contrat de ligne explicite

Une ligne contient six champs FEN, puis exactement une des annotations :
`Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black`.
La casse reste stricte : une faute d'annotation est signalée au lieu d'être
interprétée silencieusement. Les couleurs sont conservées pour ne pas perdre
d'information ; un futur entraînement à trois classes pourra les regrouper.

Le validateur vérifie :

- huit rangées de huit cases et uniquement les symboles de pièces autorisés ;
- exactement un roi de chaque couleur et aucun pion sur la première ou la
  dernière rangée ;
- le trait `w` ou `b`, les symboles et l'absence de doublon des droits de roque ;
- une case en passant `-` ou `a3`…`h3` / `a6`…`h6`, avec la rangée cohérente
  avec le joueur au trait ;
- des compteurs entiers ASCII, un demi-coup positif ou nul et un numéro de
  coup supérieur ou égal à 1 ;
- une couleur qui donne échec ou mat opposée au joueur au trait, conformément
  à la convention des données observées.

Ces vérifications empêchent les entrées manifestement mal formées de parvenir
au futur encodeur. Dans un projet plus grand, `parse_line()` constituera un
contrat réutilisable pour l'import de nouvelles sources ou une API.

**Limite : ceci ne prouve pas la légalité de la position ni son annotation.**
On ne calcule ni les attaques, ni les coups légaux, ni les échecs et mats.
Les droits de roque ne sont pas vérifiés contre les pièces, et la possibilité
effective d'une prise en passant n'est pas calculée. Une annotation erronée
mais structurellement cohérente peut donc rester. Un futur validateur métier
pourra compléter cette étape sans changer les fichiers produits.

### 3. Normaliser sans inventer de données

Les espaces, tabulations et fins de ligne deviennent des espaces simples.
Les compteurs perdent leurs zéros initiaux, les droits de roque sont ordonnés
`KQkq` et les cases vides consécutives d'une rangée sont regroupées.
On ne change ni les pièces, ni le trait, ni l'annotation. Les lignes blanches
sont comptées puis ignorées. Le fichier est lu en UTF-8 strict ligne par ligne ;
une ligne mal encodée est rejetée et comptée comme invalide.

Cela donne une représentation stable pour comparer les observations. À plus
grande échelle, cette stabilité facilitera les contrôles de qualité et les
imports depuis plusieurs outils, sans imputation arbitraire des données manquantes.

### 4. Dédupliquer par état de jeu

La clé comprend l'échiquier, le trait, les droits de roque et la case en passant.
Les deux compteurs FEN sont ignorés **pour la clé seulement** : pour la cible
actuelle échec/mat/rien, changer le numéro du coup ne doit pas créer un nouvel
exemple. La première FEN complète rencontrée est conservée, compteurs compris.
Les fichiers sont parcourus par chemin trié et les sorties par clé triée :
à entrées identiques, le contenu nettoyé et les décisions sont reproductibles.
Le chemin absolu de la source dans le bilan en mémoire dépend de l'environnement.

Pour un projet plus grand, regrouper ces répétitions avant un futur découpage
évitera qu'un même état exact soit à la fois dans l'entraînement et le test,
ce qui pourrait gonfler les scores. Cela ne supprime pas toutes les corrélations
entre positions proches d'une même partie : il faudrait des identifiants de
partie pour garantir leur séparation.

La clé garde les droits de roque et la case en passant pour rester prudente
sur les possibilités de jeu. Elle ne fusionne donc pas nécessairement toutes
les positions équivalentes au sens des règles de répétition. Si la cible future
intègre la règle des cinquante coups ou l'historique des répétitions, cette
politique devra être revue et les données brutes retraitées.

### 5. Exclure les contradictions sans voter

Si une clé porte des annotations différentes, **aucun exemplaire de cette clé**
ne sort dans `cleaned.txt`, même si une annotation est majoritaire. Les doublons
ne sont pas des votes indépendants : une erreur copiée cent fois ne doit pas
écraser une autre annotation. Les sources restent disponibles pour une
vérification métier ultérieure. On ne rééquilibre pas les classes et on ne tire
pas d'échantillon : ces décisions relèveront du futur protocole d'entraînement.

### 6. Simplifier les sorties et limiter la mémoire

Le dossier de sortie contient uniquement les données nettoyées en TXT. On ne
conserve ni le détail des rejets ni la correspondance entre les lignes nettoyées
et leurs origines. Cela simplifie les fichiers à manipuler pour l'entraînement.
Pour un projet plus grand nécessitant un audit des données, il faudra réintroduire
une journalisation et relancer le nettoyage depuis les sources intactes.

Les lignes sont lues progressivement et un index SQLite temporaire conserve
les clés sur disque. Cela évite un dictionnaire Python contenant tout le corpus
en RAM et ne demande aucune dépendance supplémentaire. Le disque temporaire
doit néanmoins pouvoir accueillir l'index, en plus des fichiers de sortie.
SQLite reste un choix local : un projet distribué pourrait remplacer cet index
par une base partagée en conservant le contrat de validation.

### 7. Lire les compteurs correctement

`valid_lines` compte les lignes structurellement valides, y compris les doublons
et contradictions. `duplicate_lines` compte les annotations identiques à la
première annotation d'une clé ; `conflicting_lines` celles qui en diffèrent,
même si la clé a déjà été mise à l'écart. Ces compteurs décrivent les observations,
tandis que `quarantined_positions` compte les clés uniques contradictoires.

Les relations de contrôle sont :

```text
lines_read = blank_lines + invalid_lines + valid_lines
valid_lines = unique_positions + duplicate_lines + conflicting_lines
unique_positions = kept_positions + quarantined_positions
```

La répartition finale des classes permet de repérer un déséquilibre avant de
choisir, plus tard, une stratégie d'entraînement adaptée.

## Vérification

```sh
python3 -m unittest discover -s tests -p 'test_datacleaning.py' -v
```

Ces tests couvrent la validation, la normalisation, les doublons entre fichiers,
la quarantaine des contradictions, l'absence de fichiers autres que TXT, la protection des sorties
et le lancement avec `-clean` (script et module). Ils vérifient aussi qu'aucun
nettoyage n'est déclenché sans cette option.

Vérification sur le corpus local lors de l'implémentation : 723 019 lignes lues,
7 460 doublons, 715 559 positions conservées, aucune ligne structurellement
invalide ni contradiction détectée. Les trois relations de contrôle ci-dessus
ont également été vérifiées. Ce constat ne valide pas les annotations au sens
des règles d'échecs et pourra changer avec une autre version des sources.
