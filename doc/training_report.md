# Rapport d'entraînement : référence et corrections

Date : 3 novembre 2026. Ce rapport conserve le journal fourni par l'utilisateur,
explique ses limites et décrit les changements apportés au code. Le résultat
de référence est **76,60 % en validation**. Le nouvel entraînement complet sur
60 000 positions n'a pas encore été exécuté : aucun score supérieur à cette
référence n'est annoncé.

## 1. Configuration de référence

```bash
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn --classes 5
```

| Paramètre | Valeur du journal et du code avant correction |
| --- | --- |
| Positions retenues | 60 000 : 12 000 par classe |
| Classes | Nothing, Check White, Check Black, Checkmate White, Checkmate Black |
| Encodage | 773 valeurs par position |
| Partition | 85 % entraînement / 15 % validation : 51 000 / 9 000 |
| Architecture | Dense 773 → 512 → 256 → 128 → 5, BatchNorm et ReLU |
| Dropout | 0,15 puis 0,10 ; aucun après la troisième couche cachée |
| Perte | Focal Loss, gamma = 2 |
| Pondérations | [1 ; 1,2 ; 1,2 ; 2 ; 2] |
| Optimiseur | Adam, taux initial 0,002, régularisation L2 0,0001 |
| Planification | CosineAnnealingLR, horizon 20, minimum 0,00001 |
| Lots | 64 positions |
| Arrêt anticipé | Patience 10 sur val_loss ; sans restauration des meilleurs poids |

La validation affichée est une partition interne du fichier `train.txt`.
Le fichier `dataset/cleaned/validation.txt` n'était pas utilisé par cette commande.
La graine, les indices retenus et les poids initiaux du lancement fourni ne sont
pas disponibles dans le journal. Les résultats ci-dessous ne constituent pas
une mesure sur le fichier de test indépendant.

## 2. Résultats fournis, époque par époque

| Époque | Perte entraînement | Précision entraînement | Perte validation | Précision validation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,4878 | 60,18 % | 0,3657 | 66,33 % |
| 2 | 0,3544 | 67,67 % | 0,3390 | 67,12 % |
| 3 | 0,3353 | 69,03 % | 0,3414 | 68,80 % |
| 4 | 0,3298 | 69,38 % | 0,3238 | 69,08 % |
| 5 | 0,3183 | 70,22 % | 0,3338 | 66,21 % |
| 6 | 0,3065 | 71,07 % | 0,3053 | 70,00 % |
| 7 | 0,2953 | 72,63 % | 0,3030 | 70,30 % |
| 8 | 0,2822 | 73,44 % | 0,2944 | 72,18 % |
| 9 | 0,2627 | 75,19 % | 0,2827 | 72,52 % |
| 10 | 0,2443 | 76,85 % | 0,2722 | 73,84 % |
| 11 | 0,2253 | 78,58 % | 0,2708 | 75,01 % |
| 12 | 0,2023 | 80,73 % | **0,2680** | 75,46 % |
| 13 | 0,1770 | 83,03 % | 0,2735 | 75,08 % |
| 14 | 0,1534 | 85,45 % | 0,2802 | 76,13 % |
| 15 | 0,1305 | 87,72 % | 0,2867 | 75,84 % |
| 16 | 0,1070 | 90,05 % | 0,3112 | 75,92 % |
| 17 | 0,0844 | 92,33 % | 0,3251 | 76,01 % |
| 18 | 0,0683 | 94,02 % | 0,3311 | 76,30 % |
| 19 | 0,0582 | 95,00 % | 0,3540 | 76,26 % |
| 20 | **0,0515** | **95,67 %** | 0,3680 | **76,60 %** |

### Interprétation

- La précision de validation progresse de 10,27 points entre les époques 1 et 20.
- Le meilleur minimum de perte apparaît à l'époque 12. Ensuite, la perte de
  validation augmente de 0,2680 à 0,3680, soit environ 37,3 %, tandis que la
  perte d'entraînement continue de baisser : le surapprentissage devient visible.
- L'écart final entre les précisions affichées atteint 19,07 points. Cet écart
  reste indicatif : la métrique d'entraînement est accumulée pendant les mises
  à jour, avec Dropout actif, alors que la validation utilise le mode inférence.
- La meilleure précision reste celle de l'époque 20. **Minimiser la perte et
  maximiser la précision ne sélectionnent donc pas le même modèle.** Arrêter à
  l'époque 12 aurait retenu 75,46 %, pas 76,60 %. La remontée de la perte peut
  notamment refléter des erreurs devenant plus confiantes ; le journal seul
  ne fournit pas une étude de calibration.
- L'ancienne patience de 10 ne pouvait pas arrêter ce lancement avant sa fin :
  seules huit époques sans nouveau minimum suivent l'époque 12.

## 3. Problèmes vérifiés et corrections

### Rétropropagation des pertes

Dans l'ancien code, la CCE retournait déjà un gradient par rapport aux logits
`(P - Y) / N`, puis `Sequential.fit()` le faisait encore traverser
`Softmax.backward()`. Cela appliquait une deuxième fois la dérivée de softmax.
La Focal Loss suivait le même chemin et omettait la dérivée de son facteur
`(1 - p_t)^gamma`. Enfin, la CCE pondérée multipliait les composantes du gradient
par les poids de toutes les classes au lieu d'appliquer correctement la
pondération de la cible.

Les [pertes](../src/neural_network/losses.py) proposent maintenant deux chemins
explicites : `backward()` par rapport aux probabilités et `backward_softmax()`
par rapport aux logits. Le [modèle](../src/neural_network/model.py) utilise le
chemin fusionné pour une CCE ou une Focal Loss avec softmax final, puis saute
uniquement cette dernière couche lors de la rétropropagation. Les autres
combinaisons conservent le parcours de toutes les couches.

Pour une cible one-hot, le gradient focal corrigé est :

```text
dL/dz_j = alpha_t / N
          * [(1 - p_t)^gamma - gamma * p_t * (1 - p_t)^(gamma - 1) * ln(p_t)]
          * (p_j - y_j)
```

Le deuxième terme du facteur entre crochets manquait auparavant. La définition
de la perte vient de l'[article original sur la Focal Loss](https://arxiv.org/abs/1708.02002).
Les dérivées sont vérifiées localement par différences finies, indépendamment
de la taille ou de la forme du gradient.

### Sélection du modèle

L'entraînement restaure désormais les meilleurs paramètres et les moyennes et
variances courantes de BatchNorm, même lorsqu'il atteint la limite d'époques.
La patience par défaut passe à 4. Le choix repose sur `val_loss` par défaut ;
`--monitor val_accuracy` permet de privilégier la proportion de bonnes réponses.
Cette distinction entre arrêt et restauration est aussi décrite dans la
[documentation EarlyStopping](https://keras.io/api/callbacks/early_stopping/),
citée comme référence conceptuelle ; l'implémentation du projet reste en NumPy.

La restauration concerne le réseau destiné à l'inférence. Elle ne constitue
pas un checkpoint complet pour reprendre exactement l'entraînement : les
moments internes d'Adam ne sont pas restaurés ni sauvegardés dans le `.nn`.

### Données et reproductibilité

- Les premiers exemples de chaque classe sont remplacés par un échantillonnage
  en réservoir parcourant le fichier entier, avec la même limite par classe.
  Cela réduit le biais dû à l'ordre des lignes et demande une lecture complète
  du corpus avant l'encodage.
- La partition interne est stratifiée par classe, disjointe en indices et
  reproductible. Sur 60 000 exemples équilibrés, elle conserve 51 000 positions
  d'entraînement et 9 000 de validation.
- `--seed 42` fixe l'échantillonnage, la partition, l'initialisation et les tirages
  d'entraînement, à environnement numérique identique. Cela ne remplace pas une
  vérification des doublons ou des parties communes entre fichiers de données.

### Régularisation, métriques et sauvegarde

- Dropout passe par défaut à 0,25 après chacune des trois couches cachées.
- La régularisation L2 passe à 0,0005 et le taux initial à 0,001. Ce sont des
  points de départ à comparer en validation, pas une garantie d'amélioration.
- La perte d'entraînement est pondérée par le nombre d'exemples des lots : un
  dernier petit lot ne compte plus autant qu'un lot complet.
- Le `.nn` conserve la configuration de perte, les paramètres de lancement,
  l'historique, l'époque retenue et les métriques finales restaurées. L'évaluation
  recharge ainsi la même perte. Les anciens modèles sans cette information
  continuent d'utiliser la CCE par défaut : leur perte ne doit pas être comparée
  directement aux valeurs focales du journal.

## 4. Vérifications après modification

Les 48 tests automatisés passent, dont sept nouveaux tests couvrant : gradients
numériques des probabilités et logits, mise à jour effective des poids,
restauration des paramètres et statistiques BatchNorm, sauvegarde/rechargement
de la perte, pondération des lots, partition stratifiée et échantillonnage reproductible.

Un essai CLI supplémentaire a vérifié les nouvelles options, les cinq classes,
la sélection sur `val_accuracy`, la CCE sans surpondération et les métadonnées
après rechargement, avec des modèles temporaires.

### Essai comparatif réduit : 1 000 positions

Un essai local a utilisé 200 positions par classe, tirées de `train.txt` avec la
graine 42, puis la même partition stratifiée pour les deux variantes :
850 positions d'entraînement et 150 de validation, lots de 64, limite 20 époques.
La variante historique reconstitue les deux Dropout 0,15/0,10, le taux 0,002,
la L2 0,0001, le gradient focal incomplet suivi d'une seconde dérivée softmax,
la patience 10 et la conservation du dernier modèle. La variante corrigée
utilise les nouveaux réglages et restaure le meilleur modèle selon `val_loss`.
Les poids initiaux utilisent la même graine ; les tirages de Dropout évoluent
différemment puisque la régularisation a changé.

| Mesure | Variante historique reconstituée | Variante corrigée |
| --- | ---: | ---: |
| Époques exécutées | 13 | 9 |
| Modèle évalué | Dernière époque, 13 | Meilleure perte, époque 5 |
| Perte de validation focale | 1,1827 | 0,8953 |
| Précision de validation | 38,00 % | 39,33 % |
| Bonnes réponses sur 150 | 57 | 59 |

Le gain observé est de **deux positions, soit 1,33 point**, sur un seul petit
échantillon. Plusieurs changements sont combinés ; cet essai ne permet pas
d'attribuer le gain à une modification précise, ni de conclure à une amélioration
statistiquement établie. Il valide le fonctionnement de la chaîne corrigée.
**Ces scores sur 1 000 positions ne sont pas comparables aux 76,60 % du
lancement sur 60 000 positions.** Le jeu de test indépendant n'a pas été utilisé.

## 5. Relancer l'expérience complète

Pour comparer en priorité la précision de validation, avec un nouveau fichier :

```bash
python3 src/my_torch.py -train \
  --dataset dataset/cleaned/train.txt \
  --output model_improved.nn \
  --classes 5 --epochs 20 --seed 42 \
  --lr 0.001 --dropout 0.25 --weight-decay 0.0005 \
  --patience 4 --monitor val_accuracy
```

Pour privilégier la perte, remplacer `--monitor val_accuracy` par
`--monitor val_loss`. La commande d'origine reste valide et prend les nouveaux
réglages par défaut. `--no-boost-mate` désactive les pondérations tactiques ;
`--loss cce` permet une comparaison avec l'entropie croisée corrigée.

Choisir les réglages sur la validation, puis évaluer le modèle final sur le
fichier de test, après vérification des recouvrements entre corpus :

```bash
python3 src/my_torch.py -eval --model model_improved.nn --dataset dataset/cleaned/test.txt
```

Le prochain bilan devra relever la précision globale, le rappel de chaque
classe, notamment les Checkmate, et les résultats de plusieurs graines. Ne pas
sélectionner les hyperparamètres à partir du score du fichier de test.

## 6. Deuxième lancement complet : régression et nouvelle correction

Cette section est ajoutée à la suite de l'historique précédent. Elle analyse
le nouvel entraînement complet communiqué par l'utilisateur et les changements
effectués après ce résultat. Les réglages des sections précédentes décrivent
la première révision ; les valeurs ci-dessous décrivent la deuxième.

### 6.1. Résultats du lancement reçu

```bash
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn --classes 5
```

Les 60 000 positions sont réparties de façon stratifiée : 51 000 pour
l'entraînement et 9 000 pour la validation, avec la graine 42. Les métadonnées
du `model.nn` local confirment le critère `val_loss`, la sélection de l'époque 9,
les 13 époques exécutées et la précision finale arrondie à 65,76 %.

| Époque | Perte entraînement | Précision entraînement | Perte validation | Précision validation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,6691 | 48,58 % | 0,4618 | 56,03 % |
| 2 | 0,4571 | 59,41 % | 0,4256 | 59,93 % |
| 3 | 0,4131 | 62,10 % | 0,4077 | 62,10 % |
| 4 | 0,3812 | 64,08 % | 0,4015 | 63,17 % |
| 5 | 0,3545 | 65,47 % | 0,3929 | 62,40 % |
| 6 | 0,3349 | 66,30 % | 0,3956 | 63,68 % |
| 7 | 0,3146 | 68,19 % | 0,3858 | 64,36 % |
| 8 | 0,2889 | 69,62 % | 0,3910 | 66,18 % |
| 9 | 0,2695 | 71,36 % | **0,3842** | **65,76 % — modèle sauvegardé** |
| 10 | 0,2453 | 73,37 % | 0,3920 | 67,03 % |
| 11 | 0,2220 | 75,37 % | 0,4010 | 67,33 % |
| 12 | 0,2026 | 77,35 % | 0,4438 | 67,77 % |
| 13 | 0,1790 | 79,82 % | 0,4258 | **68,83 % — maximum observé** |

L'arrêt anticipé intervient à l'époque 13 et restaure l'époque 9.
Résultat final communiqué : **65,76 % de précision, perte focale 0,3842**.

### 6.2. Le problème et ce qui devait changer

**Le critère par défaut répondait mal à l'objectif de précision globale.**
Les époques 10 à 13 ne battent pas la perte de l'époque 9 ; la patience 4 se
déclenche donc correctement. Pourtant, la précision continue de progresser.
Le programme optimise et restaure ce qui lui a été demandé, mais le défaut
`val_loss` choisi dans la première révision ne retient pas le meilleur taux de
bonnes réponses demandé ici.

Sur les seules époques observées, sélectionner la précision aurait retenu
l'époque 13 à 68,83 % au lieu de l'époque 9 à 65,76 %, soit environ **3,07 points**
d'après les valeurs affichées. Le fichier actuel contient les poids de
l'époque 9 ; changer le critère ne reconstruit pas les poids de l'époque 13.
Un nouvel entraînement est nécessaire.

**La régression ne se résume pas à la sélection du modèle.** Le premier journal
atteignait 76,60 %. Le modèle sauvegardé du deuxième lancement perd donc
10,84 points, et son maximum observé perd encore 7,77 points. Les dérivées
corrigées restent justifiées par les tests numériques, mais les réglages de
la première révision n'ont pas amélioré la précision affichée sur ces lancements.
Les échantillons et partitions du tout premier lancement ne sont pas archivés ;
ils ne sont pas identiques par construction à la partition stratifiée actuelle.
La régularisation, les gradients et l'échantillonnage ont aussi changé : on ne
peut pas attribuer toute cette différence à un paramètre isolé.

**La perte pondérée poursuivait un autre compromis.** L'échantillonnage fournit
déjà 12 000 positions par classe, mais la perte multiplie encore les Check par
1,2 et les Checkmate par 2. Cette priorité tactique peut se défendre si le rappel
des mats est le but principal ; elle n'est pas le même objectif que maximiser
la proportion globale de bonnes réponses, où chaque position compte une fois.
La Focal Loss ajoute également un mécanisme de priorité aux exemples difficiles.
Une CCE sans pondération sert ici de nouveau point de comparaison plus simple.

**Le bilan masquait les différences entre classes et entre époques.** Il fallait
afficher séparément le meilleur score observé, le minimum de perte et l'époque
réellement sauvegardée, puis calculer les métriques de chaque classe sur ce
modèle sauvegardé.

### 6.3. Essai contrôlé avant de modifier les valeurs par défaut

Trois variantes ont été comparées sur les mêmes 2 000 positions : 400 par classe,
échantillonnées dans `train.txt` avec la graine 42. La partition commune contient
1 700 positions d'entraînement et 300 de validation, soit 60 par classe en
validation. L'initialisation et les tirages d'entraînement utilisent la graine 42.
Le jeu de test indépendant n'est pas utilisé.

L'architecture 773 → 512 → 256 → 128 → 5, BatchNorm, les trois Dropout 0,25,
Adam à 0,001, la L2 0,0005, les lots de 64 et CosineAnnealingLR avec une limite
de 20 époques sont communs aux trois variantes. Chacune restaure le meilleur
réseau selon son critère.

| Variante | Critère / patience | Époques exécutées | Époque retenue | Précision validation | Bonnes réponses / 300 |
| --- | --- | ---: | ---: | ---: | ---: |
| Focal Loss pondérée, réglages reçus | val_loss / 4 | 8 | 4 | 54,67 % | 164 |
| Focal Loss pondérée, sélection sur précision | val_accuracy / 6 | 11 | 5 | 55,67 % | 167 |
| CCE sans pondération, sélection sur précision | val_accuracy / 6 | 9 | 3 | **58,67 %** | **176** |

Sur ce pilote, la nouvelle configuration gagne 12 bonnes réponses, soit
**4,00 points** par rapport aux réglages reçus, et 9 bonnes réponses, soit
3,00 points, par rapport à la Focal Loss sélectionnée sur la précision.
Changer le critère et la patience apporte déjà 1,00 point dans ce pilote ;
ce résultat ne sépare pas leurs contributions individuelles.

Le compromis est visible dans les rappels par classe :

| Classe | Focal pondérée / perte | Focal pondérée / précision | CCE uniforme / précision |
| --- | ---: | ---: | ---: |
| Nothing | 20,00 % | 20,00 % | 26,67 % |
| Check White | 33,33 % | 46,67 % | 51,67 % |
| Check Black | 56,67 % | 61,67 % | 66,67 % |
| Checkmate White | 85,00 % | 71,67 % | 73,33 % |
| Checkmate Black | 78,33 % | 78,33 % | 75,00 % |

La CCE augmente la précision globale de ce pilote mais réduit certains rappels
de mats par rapport à la variante sélectionnée sur la perte. Le réglage
`--boost-mate` reste disponible lorsque la priorité est tactique.
Il s'agit d'un seul petit échantillon et d'une seule graine : **le gain de
4 points n'est ni une garantie sur 60 000 positions ni une preuve de supériorité
statistique**. Les pertes CCE et focales ne sont pas comparées numériquement
entre familles. Aucun score final du prochain lancement complet n'est inventé.

### 6.4. Modifications exactes dans le code

Les changements de valeurs par défaut concernent
[`src/training/train.py`](../src/training/train.py). Le point d'entrée
`src/my_torch.py` récupère déjà ces valeurs dans le parseur partagé.

| Paramètre ou comportement | Avant ce deuxième correctif | Après ce deuxième correctif | Motif |
| --- | --- | --- | --- |
| `--monitor` | `val_loss` | **`val_accuracy`** | Retenir le meilleur taux de bonnes réponses observé |
| `--patience` | **4** | **6** | Tolérer deux époques supplémentaires sans amélioration du critère choisi |
| `--loss` | **`focal`** | **`cce`** | Nouveau point de comparaison pour la précision globale |
| Surpondération tactique par défaut | **Activée** | **Désactivée** | Chaque exemple contribue sans priorité de classe ajoutée |
| Poids Nothing | 1,0 | 1,0 | Inchangé |
| Poids Check White | **1,2** | **1,0** | Pondération uniforme |
| Poids Check Black | **1,2** | **1,0** | Pondération uniforme |
| Poids Checkmate White | **2,0** | **1,0** | Pondération uniforme |
| Poids Checkmate Black | **2,0** | **1,0** | Pondération uniforme |
| Gamma focal | 2,0 utilisé par défaut | Inactif pour la CCE ; toujours 2,0 avec `--loss focal` | Le paramètre n'est pas modifié artificiellement |
| Dernier résumé | Précision et perte du réseau restauré | Maximum de précision, minimum de perte, époques correspondantes et critère du réseau retenu | Éviter de confondre pic observé et modèle sauvegardé |
| Diagnostics de classes | Absents de l'affichage standard | Précision, rappel, F1, support et matrice de confusion | Voir quelles classes progressent ou régressent |
| Métadonnées `.nn` | Historique, critère et époque retenue | Champs précédents + meilleurs scores/époques et métriques par classe | Conserver un bilan vérifiable |

La pondération uniforme est représentée dans le code par `class_weights=None`,
équivalent à un facteur 1 pour chaque classe. `--boost-mate` rétablit les
pondérations [1 ; 1,2 ; 1,2 ; 2 ; 2] en cinq classes ; `--no-boost-mate` les
désactive explicitement.

Les valeurs suivantes restent identiques à celles du deuxième lancement reçu :

| Paramètre conservé | Avant | Après |
| --- | ---: | ---: |
| Taux initial `--lr` | 0,001 | 0,001 |
| Dropout, trois couches cachées | 0,25 | 0,25 |
| Régularisation L2 `--weight-decay` | 0,0005 | 0,0005 |
| Maximum d'époques | 20 | 20 |
| Taille de lot | 64 | 64 |
| Limite par classe | 12 000 | 12 000 |
| Graine | 42 | 42 |
| Partition entraînement / validation | 85 % / 15 % | 85 % / 15 % |
| Largeurs cachées | 512 / 256 / 128 | 512 / 256 / 128 |

L'échantillonnage en réservoir, la stratification, les gradients corrigés et
la restauration des statistiques BatchNorm sont conservés. Cette fois, les
autres réglages numériques sont maintenus afin de limiter les facteurs modifiés.

Dans [`model.py`](../src/neural_network/model.py), `evaluate()` peut maintenant
renvoyer les prédictions avec `return_predictions=True`, en plus de la perte et
de la précision. Le comportement par défaut reste le tuple à deux valeurs.
Cela permet de calculer les diagnostics sans refaire une propagation avant.
[`evaluate.py`](../src/training/evaluate.py) cumule les matrices de confusion
par lot et affiche aussi les métriques par classe lors de `-eval`.

Les matrices ont pour lignes les classes attendues et pour colonnes les classes
prédites. Une classe absente ou jamais prédite reçoit des métriques nulles plutôt
qu'une division par zéro ; son support est affiché pour interpréter ce cas.
La macro-F1 fait la moyenne sur toutes les classes déclarées du modèle.
Le tableau final d'entraînement porte sur le réseau restauré, et non sur les
poids d'une autre époque figurant dans l'historique.

### 6.5. Vérifications et prochain lancement

La suite comporte désormais **51 tests réussis**. Les trois ajouts vérifient les
nouveaux défauts et l'accès explicite aux anciens réglages, la sélection de
poids d'une époque plus précise malgré une perte plus élevée, et les métriques
des classes absentes ou jamais prédites. Les tests existants vérifient toujours
les gradients numériques, la restauration BatchNorm, les modes CLI et la
sauvegarde/relecture des modèles.

La commande courte communiquée par l'utilisateur utilise désormais les nouveaux
défauts. Pour expliciter les paramètres de la prochaine comparaison et conserver
le fichier `model.nn` actuel :

```bash
python3 src/my_torch.py -train \
  --dataset dataset/cleaned/train.txt --output model_next.nn --classes 5 \
  --loss cce --no-boost-mate --monitor val_accuracy --patience 6 \
  --epochs 20 --batch-size 64 --max-per-class 12000 --seed 42 \
  --lr 0.001 --dropout 0.25 --weight-decay 0.0005
```

Pour isoler l'effet de la sélection tout en conservant la Focal Loss pondérée :

```bash
python3 src/my_torch.py -train \
  --dataset dataset/cleaned/train.txt --output model_focal_accuracy.nn --classes 5 \
  --loss focal --boost-mate --monitor val_accuracy --patience 6 --seed 42
```

Les sorties `model_*.nn` sont ignorées par Git. Le lancement complet de la
nouvelle configuration n'a pas été exécuté ici ; le `model.nn` fourni est
conservé. Le prochain résultat devra être ajouté à ce rapport avec les mêmes
informations de lancement et les métriques par classe. Le score sur le fichier
de test indépendant reste réservé à l'évaluation finale des réglages retenus.

## 7. Troisième lancement complet : 71,10 % et limites restantes

Cette section prolonge l'historique sans modifier les journaux précédents.
Elle conserve le troisième lancement fourni, analyse ses erreurs et distingue
les modifications retenues des essais qui n'ont pas amélioré la précision.

### 7.1. Configuration et résultats communiqués

```bash
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn --classes 5
```

Configuration : 60 000 positions, 12 000 par classe, graine 42, partition
stratifiée 51 000 / 9 000 ; architecture 773 → 512 → 256 → 128 → 5, BatchNorm,
trois Dropout 0,25 ; CCE sans pondération, Adam à 0,001, L2 0,0005, lots de 64,
CosineAnnealingLR sur 20 époques, sélection sur `val_accuracy`, patience 6 et
restauration des meilleurs poids. Les métadonnées du fichier local `model.nn`
confirment la sélection de l'époque 17 et la précision 0,711.

| Époque | Perte entraînement | Précision entraînement | Perte validation | Précision validation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,0110 | 51,75 % | 0,8289 | 60,84 % |
| 2 | 0,8135 | 62,49 % | 0,7778 | 64,40 % |
| 3 | 0,7518 | 65,90 % | 0,7544 | 64,99 % |
| 4 | 0,7001 | 68,55 % | 0,7389 | 66,26 % |
| 5 | 0,6545 | 70,89 % | 0,7195 | 66,99 % |
| 6 | 0,6109 | 73,45 % | **0,7123** | 67,46 % |
| 7 | 0,5558 | 76,35 % | 0,7209 | 68,43 % |
| 8 | 0,5066 | 78,78 % | 0,7315 | 68,43 % |
| 9 | 0,4508 | 81,46 % | 0,7402 | 69,24 % |
| 10 | 0,3961 | 84,01 % | 0,7637 | 69,40 % |
| 11 | 0,3400 | 86,63 % | 0,8149 | 69,37 % |
| 12 | 0,2927 | 88,69 % | 0,8691 | 69,53 % |
| 13 | 0,2496 | 90,45 % | 0,9022 | 69,94 % |
| 14 | 0,2081 | 92,20 % | 0,9534 | 69,92 % |
| 15 | 0,1767 | 93,46 % | 0,9766 | 70,38 % |
| 16 | 0,1493 | 94,56 % | 1,0175 | 70,78 % |
| 17 | 0,1276 | 95,32 % | **1,0486 — modèle sauvegardé** | **71,10 %** |
| 18 | 0,1186 | 95,70 % | 1,0864 | 70,79 % |
| 19 | 0,1071 | 96,17 % | 1,0859 | 70,96 % |
| 20 | 0,1027 | 96,29 % | 1,0818 | 70,89 % |

Le réseau sauvegardé est celui de l'époque 17, pas celui de l'époque 20.
Son score est **6 399 bonnes réponses sur 9 000 : 71,10 %**, perte CCE 1,0486,
macro-F1 0,7098. La patience 6 n'a pas été atteinte après l'époque 17 puisque
seules trois époques restaient avant la limite.

Comparaison avec les journaux précédents :

- Gain de **5,34 points** sur le modèle sauvegardé précédent à 65,76 %.
- Gain de **2,27 points** sur le maximum de 68,83 % observé dans ce précédent
  lancement, même s'il n'avait pas été sauvegardé.
- Le premier journal à 76,60 % reste supérieur de **5,50 points**. Ses indices
  d'échantillonnage et sa partition n'étant pas archivés, il n'est pas une
  comparaison contrôlée avec les lancements stratifiés à graine 42.

### 7.2. Résultats par classe et matrice de confusion

| Classe | Précision | Rappel | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Nothing | 73,17 % | **56,06 %** | 63,48 % | 1 800 |
| Check White | 64,48 % | 70,28 % | 67,25 % | 1 800 |
| Check Black | 64,69 % | 73,50 % | 68,82 % | 1 800 |
| Checkmate White | 77,66 % | 77,44 % | 77,55 % | 1 800 |
| Checkmate Black | 77,41 % | 78,22 % | 77,81 % | 1 800 |

Lignes : classe attendue. Colonnes : classe prédite.

| Attendu / prédit | Nothing | Check White | Check Black | Checkmate White | Checkmate Black |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nothing | 1 009 | 328 | 344 | 60 | 59 |
| Check White | 194 | 1 265 | 0 | 341 | 0 |
| Check Black | 125 | 0 | 1 323 | 0 | 352 |
| Checkmate White | 37 | 369 | 0 | 1 394 | 0 |
| Checkmate Black | 14 | 0 | 378 | 0 | 1 408 |

La matrice contient **2 601 erreurs**, réparties ainsi :

| Groupe d'erreurs | Nombre | Part des erreurs |
| --- | ---: | ---: |
| Check ↔ Checkmate de même couleur | **1 440** | **55,36 %** |
| Nothing → Check | 672 | 25,84 % |
| Nothing → Checkmate | 119 | 4,58 % |
| Check ou Checkmate → Nothing | 370 | 14,23 % |

Les pourcentages sont arrondis. Sur les quatre classes colorées, aucune erreur
ne va vers une classe de la couleur opposée dans cette validation. La principale
difficulté observée est donc la distinction de l'état tactique, surtout entre
échec et mat, et le rappel de Nothing ; une nouvelle priorité arbitraire à une
couleur ne répondrait pas à ces erreurs.

### 7.3. Le problème restant et les pistes testées

**Le choix du checkpoint fonctionne désormais pour la précision.** Restaurer
l'époque 17 retient bien le maximum observé. Revenir au minimum de perte
aurait retenu l'époque 6 à 67,46 %, soit 3,64 points de moins. Il ne faut pas
confondre ce compromis avec une panne de la restauration.

**Le surapprentissage reste marqué.** La perte de validation passe de 0,7123
à l'époque 6 à 1,0486 à l'époque 17 : environ **47,21 % d'augmentation**, alors
que la précision gagne seulement 3,64 points. À l'époque 17, l'écart entre les
précisions affichées atteint **24,22 points** : 95,32 % contre 71,10 %. Les
métriques d'entraînement sont accumulées pendant les mises à jour avec Dropout
actif, tandis que la validation est en mode inférence ; cet écart reste donc
indicatif. Les tendances combinées montrent néanmoins une mémorisation croissante.
La remontée de CCE est compatible avec des erreurs plus confiantes, sans
constituer à elle seule une mesure complète de calibration.

Trois pistes ont été testées avant de modifier les défauts : lisser légèrement
les cibles CCE, réduire la capacité du réseau, ou augmenter le Dropout. Le pilote
utilise les mêmes 2 000 positions, la même partition 1 700 / 300, les mêmes
graines 42 et les mêmes réglages communs que le pilote précédent. La validation
contient 60 positions par classe. Les époques sont limitées à 20, avec sélection
sur précision et patience 6 ; la règle de départage décrite ci-dessous est
commune aux variantes. Le fichier de test indépendant n'a pas été utilisé.

### 7.4. Résultats des essais : variantes non retenues par défaut

| Variante du pilote | Valeur changée | Époque retenue / exécutées | Bonnes réponses / 300 | Précision | Perte CCE de validation |
| --- | --- | --- | ---: | ---: | ---: |
| Référence conservée | Aucun lissage, réseau large, Dropout 0,25 | 3 / 9 | **176** | **58,67 %** | 1,0201 |
| Lissage léger | epsilon **0 → 0,05** | 3 / 9 | 171 | 57,00 % | 1,0339 |
| Lissage plus fort | epsilon **0 → 0,10** | 3 / 9 | 173 | 57,67 % | 1,0487 |
| Réseau compact | Couches **512/256/128 → 256/128/64** | 9 / 15 | 164 | 54,67 % | 1,0627 |
| Dropout renforcé | **0,25 → 0,35** sur les trois couches cachées | 7 / 13 | 168 | 56,00 % | 0,9892 |

Les autres réglages sont maintenus : Adam 0,001, L2 0,0005, lots de 64,
CCE sans pondération, même planification. Les variantes de capacité utilisent
BatchNorm et sont initialisées avec la graine 42 ; leurs dimensions différentes
ne permettent évidemment pas de conserver les mêmes matrices de poids.

Le compact passe de **562 949 à 240 517 paramètres entraînables**, soit environ
57,28 % de moins, mais perd 12 bonnes réponses dans ce pilote. Il n'est pas
adopté. Le Dropout 0,35 réduit la perte de validation sur ce pilote mais perd
8 bonnes réponses : il ne satisfait pas ici l'objectif prioritaire de précision.
Les deux lissages perdent respectivement 5 et 3 bonnes réponses.

**Aucune de ces quatre variantes n'est donc activée par défaut.** Ces résultats
restent exploratoires : un petit échantillon et une graine ne prouvent pas ce
qui se produirait sur 60 000 positions. Plusieurs essais sur cette même
validation peuvent aussi favoriser des choix optimistes ; ils ne remplacent
pas une évaluation finale indépendante. Aucun gain sur les 71,10 % n'est annoncé.

### 7.5. Modifications réellement retenues dans le code

Les valeurs testées dans la section précédente ne doivent pas être confondues
avec les changements appliqués par défaut :

| Paramètre ou comportement | Avant | Après | Pourquoi |
| --- | --- | --- | --- |
| Départage à précision exactement égale | Première époque atteignant ce score | **Perte de validation la plus faible parmi ces époques** | Préférer la meilleure CCE sans diminuer la précision maximale |
| Patience lors d'une égalité de précision | Pas de remise à zéro | **Toujours pas de remise à zéro** | Une amélioration de perte seule ne prolonge pas une attente basée sur la précision |
| Époque annoncée pour la meilleure précision | Premier maximum de précision | Maximum de précision, puis minimum de perte à égalité, puis première occurrence si les deux sont égaux | Cohérence avec le modèle retenu |
| Contrôle des métriques | Contrôle du critère principal fini | Critère principal **et perte de validation finis** | Ne pas utiliser une perte invalide pour départager |
| Option `--label-smoothing` | Absente, epsilon implicitement 0 | Option CCE disponible, **défaut 0** | Permettre une expérience explicite sans imposer une variante moins bonne au pilote |
| Cibles d'entraînement avec `--label-smoothing 0.05` | One-hot : 1 pour la cible et 0 ailleurs | En cinq classes : **0,96** pour la cible et **0,01** pour chacune des autres | Tester un apprentissage moins incité à produire des probabilités extrêmes |
| Cibles de validation et de test | One-hot ou indices de classe | **Inchangées** | Garder une CCE de validation comparable à la référence |
| Métadonnées `.nn` et affichage de lancement | Pas de champ de lissage ni de départage explicite | Valeur effective `label_smoothing` et `checkpoint_tie_breaker` enregistrées et affichées | Conserver les conditions exactes de l'expérience |

Dans [`model.py`](../src/neural_network/model.py), le départage utilise la
précision non arrondie : deux lignes affichant le même pourcentage arrondi ne
sont pas nécessairement égales. Le snapshot comprend toujours les statistiques
BatchNorm. Si la précision et la perte sont toutes deux identiques, la première
époque est conservée. **Cette règle n'aurait pas changé le choix de l'époque 17
du journal reçu, dont le maximum est unique.**

Dans [`train.py`](../src/training/train.py), le lissage s'applique seulement aux
cibles d'entraînement : `y_lisse = (1 - epsilon) * y_one_hot + epsilon / C`.
La valeur doit être finie et comprise entre 0 inclus et 1 exclu. Un lissage
non nul avec `--loss focal` est refusé pour éviter un changement silencieux de
sa définition. La CCE sauvegardée et utilisée par `-eval` reste la CCE calculée
sur les cibles de validation/test non lissées. La perte d'entraînement lissée
n'est pas directement comparable à la perte d'entraînement à cibles dures du
journal précédent.

Le point d'entrée [`my_torch.py`](../src/my_torch.py) transmet la nouvelle option
et continue de la réserver à `-train`. Le résumé des meilleurs scores utilise
le même départage que le modèle, et les deux champs sont enregistrés dans les
métadonnées d'entraînement.

Voici les paramètres numériques conservés par défaut après ces essais :

| Paramètre | Avant | Après |
| --- | --- | --- |
| Lissage des cibles CCE | 0 | **0** |
| Largeurs cachées | 512 / 256 / 128 | **512 / 256 / 128** |
| Dropout | 0,25 | **0,25** |
| Taux initial | 0,001 | **0,001** |
| L2 | 0,0005 | **0,0005** |
| Perte | CCE sans pondération | **CCE sans pondération** |
| Critère principal | val_accuracy | **val_accuracy** |
| Patience | 6 | **6** |
| Maximum d'époques | 20 | **20** |
| Lot / limite par classe / graine | 64 / 12 000 / 42 | **64 / 12 000 / 42** |

### 7.6. Vérifications et suite de l'expérience

La suite compte **54 tests réussis**. Les trois nouveaux tests couvrent le
lissage des seules cibles d'entraînement et sa sauvegarde, le rejet des valeurs
invalides ou d'une perte incompatible, et le départage d'égalités de précision
sans remise à zéro de la patience. Ils s'ajoutent aux vérifications de gradients,
de BatchNorm, des métriques, du format `.nn` et des commandes existantes.

Le rapport précédent est conservé intégralement et le fichier `model.nn` à
71,10 % n'est pas remplacé par un modèle issu du pilote. Le lancement complet
d'une variante de cette section n'a pas été effectué. **Relancer la même
commande avec les mêmes données et le même environnement ne doit donc pas
être présenté comme une promesse de progression : les réglages numériques
restent identiques.**

Pour explorer explicitement le lissage sur le corpus complet, avec un nouveau
fichier et en gardant à l'esprit son résultat inférieur sur le petit pilote :

```bash
python3 src/my_torch.py -train \
  --dataset dataset/cleaned/train.txt --output model_smoothing005.nn --classes 5 \
  --loss cce --no-boost-mate --monitor val_accuracy --patience 6 \
  --epochs 20 --batch-size 64 --max-per-class 12000 --seed 42 \
  --lr 0.001 --dropout 0.25 --weight-decay 0.0005 --label-smoothing 0.05
```

`--label-smoothing 0` désactive l'option et correspond aux cibles de la référence
à 71,10 %. Les fichiers `model_*.nn` sont ignorés par Git. Le prochain bilan
devra comparer en validation les mêmes indices et graines, la précision, la
perte CCE sur cibles dures et les rappels de Nothing, Check et Checkmate. Le
fichier de test indépendant reste réservé à la configuration finalement retenue.

## 8. Répétition du lancement : résultat reproduit à 71,10 %

Le journal suivant, communiqué après les modifications de la section 7, utilise
la même commande :

```bash
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn --classes 5
```

Les 20 époques et la matrice de confusion reproduisent les valeurs affichées
en section 7. Il s'agit d'une répétition de la référence, pas d'une nouvelle
variante de réglages. Le tableau des époques et la matrice ne sont donc pas
dupliqués ici.

| Mesure | Lancement précédent | Répétition |
| --- | ---: | ---: |
| Positions entraînement / validation | 51 000 / 9 000 | 51 000 / 9 000 |
| Graine | 42 | 42 |
| Lissage des cibles | 0 | 0 |
| Époque retenue | 17 | 17 |
| Précision du modèle retenu | 71,10 % | 71,10 % |
| Perte CCE du modèle retenu | 1,0486 | 1,0486 |
| Meilleure perte observée | 0,7123, époque 6 | 0,7123, époque 6 |
| Bonnes réponses / validation | 6 399 / 9 000 | 6 399 / 9 000 |
| Macro-F1 | 0,7098 | 0,7098 |

Les métadonnées du fichier `model.nn` confirment `label_smoothing=0.0`,
`checkpoint_tie_breaker=val_loss`, la graine 42 et les métriques reproduites.

### Pourquoi le résultat reste identique

Les paramètres numériques par défaut avaient été conservés après les résultats
inférieurs des quatre variantes du pilote. Le lissage est une option disponible,
mais il reste désactivé sans argument explicite. La graine fixe les tirages de
l'échantillonnage, de la partition, de l'initialisation et de l'entraînement.
Enfin, le départage par perte ne modifie pas le choix du maximum unique de
précision à l'époque 17.

Cette répétition vérifie donc la reproductibilité observée, à la précision des
valeurs du journal, dans l'environnement utilisé. Elle ne mesure aucun gain
supplémentaire et ne fournit pas une évaluation sur le fichier de test indépendant.
Les constats de surapprentissage et de confusion Check/Checkmate de la section 7
restent valables. Aucun nouveau changement de code ou de paramètre n'est appliqué
sur la seule base de ce journal identique ; la prochaine comparaison devra
correspondre à une variante explicitement identifiée.

## 9. Amélioration du code : augmentation par symétrie de couleurs

### 9.1. Problème visé et modification de l'apprentissage

La référence conserve 71,10 % en validation, avec 55,36 % des erreurs entre
Check et Checkmate de même couleur. Les essais précédents de lissage, de
capacité et de Dropout n'avaient pas amélioré le petit pilote. Une nouvelle
modification porte cette fois sur les exemples utilisés pour apprendre :
exploiter la symétrie entre positions jouées par les Blancs et les Noirs.

Pour une position d'entraînement, la nouvelle fonction
[`augment_color_symmetry`](../src/data/augmentation.py) ajoute une variante :

1. Les rangées sont inversées : a1 ↔ a8, b1 ↔ b8, etc. Les colonnes restent
   identiques ; il ne s'agit pas d'un échange des côtés roi et dame.
2. Les six canaux des pièces blanches et ceux des pièces noires sont échangés.
3. Le joueur au trait est inversé et les droits de roque K ↔ k et Q ↔ q sont
   échangés.
4. Les classes Check White ↔ Check Black et Checkmate White ↔ Checkmate Black
   sont échangées. Nothing reste Nothing. Pour un modèle trois classes, les
   étiquettes Nothing/Check/Checkmate restent inchangées.

La dimension reste **773** et la bibliothèque neuronale conserve la même
architecture. Les annotations supplémentaires sont obtenues par transformation
des annotations existantes, sans appel au moteur pour produire un verdict.
La transformation peut être appliquée deux fois pour retrouver l'entrée initiale.

### 9.2. Séparation de la validation

L'augmentation intervient **après** la partition stratifiée. Elle est appliquée
uniquement à l'entraînement. Une variante dont les 773 valeurs correspondent
exactement à une entrée de validation est rejetée, indépendamment de son
annotation. Les caractéristiques binaires sont compactées en clés de 97 octets
pour limiter la mémoire du contrôle de présence.

Ce contrôle protège contre l'introduction de copies de validation par la
nouvelle augmentation. Il ne remplace pas un audit des doublons déjà présents
dans les exemples originaux, des positions équivalentes ou des parties communes.
L'encodage actuel ne contient ni les compteurs de coups ni la case en passant :
le contrôle porte sur les caractéristiques effectivement vues par le réseau.

Une préparation complète du corpus, sans réentraînement du grand modèle, a
mesuré les effectifs suivants :

| Mesure | Sans augmentation | Avec symétrie |
| --- | ---: | ---: |
| Positions sélectionnées dans le fichier source | 60 000 | 60 000 |
| Positions originales d'entraînement | 51 000 | 51 000 |
| Variantes candidates | 0 | 51 000 |
| Variantes reproduisant une entrée de validation | 0 | **5 exclues** |
| Variantes effectivement ajoutées | 0 | **50 995** |
| Total d'exemples d'entraînement | 51 000 | **101 995** |
| Positions de validation | 9 000 | **9 000, inchangées** |

### 9.3. Premier pilote : contrôle de la répétition des exemples

Le premier essai reprend 2 000 positions sources, 400 par classe, avec graine 42
et partition commune de 1 700 exemples d'entraînement et 300 de validation.
L'architecture, les poids initiaux, la CCE uniforme, Adam 0,001, la L2 0,0005,
les Dropout 0,25, les lots de 64 et l'horizon de planification de 20 époques sont
identiques. Le meilleur modèle est restauré sur la précision, avec patience 6.

Une variante de contrôle répète simplement les positions originales afin
d'obtenir autant d'exemples par époque que la variante symétrique. Aucune
variante symétrique n'a été exclue dans ce pilote.

| Variante | Exemples par époque | Époques exécutées | Époque retenue | Bonnes réponses / 300 | Précision | Perte CCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Référence | 1 700 | 9 | 3 | 176 | 58,67 % | 1,0201 |
| Répétition des originaux | 3 400 | 13 | 7 | 173 | 57,67 % | 1,3871 |
| Symétrie de couleurs | 3 400 | 11 | 5 | **179** | **59,67 %** | **0,9296** |

La symétrie gagne trois bonnes réponses sur la référence, soit **1,00 point**,
et six sur la répétition, soit **2,00 points**. Le contrôle ne soutient donc pas
l'idée que répéter simplement les exemples suffit à reproduire ce gain.
L'arrêt anticipé entraîne toutefois des nombres d'époques et de mises à jour
différents : ce pilote ne sépare pas parfaitement tous les effets d'optimisation.

### 9.4. Vérification sur un pilote plus grand

Un second essai utilise 5 000 positions sources, 1 000 par classe : partition
commune de 4 250 exemples d'entraînement et 750 de validation, graine 42.
Les autres réglages restent ceux de la référence ; seules les variantes
symétriques sont ajoutées à l'entraînement. Aucune exclusion de variante n'est
nécessaire dans ce deuxième pilote.

| Variante | Exemples par époque | Époques exécutées | Époque retenue | Bonnes réponses / 750 | Précision | Perte CCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Référence | 4 250 | 20 | 16 | 420 | 56,00 % | 1,5629 |
| Symétrie de couleurs | 8 500 | 11 | 5 | **458** | **61,07 %** | **0,9589** |

Le gain est de **38 bonnes réponses, soit 5,07 points**. Les rappels par classe
sur les mêmes 150 positions de validation par classe sont :

| Classe | Référence | Symétrie de couleurs |
| --- | ---: | ---: |
| Nothing | 42,00 % | 44,00 % |
| Check White | 56,67 % | 62,67 % |
| Check Black | 55,33 % | 62,67 % |
| Checkmate White | 58,00 % | 65,33 % |
| Checkmate Black | 68,00 % | 70,67 % |

Les deux pilotes donnent un résultat favorable, mais utilisent une seule
graine et des tailles de corpus réduites. Le nombre d'exemples par époque
augmente et la trajectoire d'optimisation change. Les scores de ces pilotes
ne sont pas directement comparables au 71,10 % obtenu sur 60 000 positions.
**Le gain sur le lancement complet reste à mesurer ; aucun nouveau score
supérieur à 71,10 % n'est annoncé pour ce lancement.** Le fichier de test
indépendant n'a pas été utilisé.

### 9.5. Paramètres et comportements effectivement modifiés

La symétrie est retenue comme nouveau défaut à la suite de ces deux essais.

| Paramètre ou comportement | Avant | Après |
| --- | --- | --- |
| Augmentation de couleurs | Désactivée / absente du lancement standard | **Activée par défaut** |
| Exemples d'entraînement sur le corpus actuel | **51 000** | **101 995** |
| Validation sur le corpus actuel | 9 000 | **9 000, inchangées** |
| Lots par époque, lot de 64 | 797 | **1 594** |
| Nombre d'entrées / de sorties | 773 / 5 | **773 / 5** |
| Largeurs cachées | 512 / 256 / 128 | **512 / 256 / 128** |
| Perte et pondération | CCE uniforme | **CCE uniforme** |
| Taux initial / L2 / Dropout | 0,001 / 0,0005 / 0,25 | **0,001 / 0,0005 / 0,25** |
| Lissage / patience / limite d'époques | 0 / 6 / 20 | **0 / 6 / 20** |
| Critère / départage / graine | val_accuracy / val_loss / 42 | **val_accuracy / val_loss / 42** |
| Retour à la référence | Commande standard précédente | **`--no-augment-colors`** |

Le nouveau défaut change donc réellement les données d'apprentissage de la
commande standard. `--augment-colors` permet de l'indiquer explicitement ;
les deux options d'activation et de désactivation sont mutuellement exclusives
et réservées au mode entraînement.

L'augmentation n'effectue aucun tirage aléatoire supplémentaire. La partition
et l'initialisation restent reproductibles ; le mélange d'un ensemble plus
grand modifie ensuite la trajectoire d'entraînement. Les effectifs avant/après,
le booléen d'activation et le nombre d'exclusions sont enregistrés sous
`training.augmentation` dans les métadonnées du `.nn`, avec
`train_samples_before_augmentation` et le total `train_samples`.

Le coût maximal par époque est proche du double et la matrice d'entraînement
est presque deux fois plus grande. Le code libère le corpus encodé complet
après avoir construit les partitions indépendantes, puis remplit la partie
augmentée par blocs pour éviter une copie filtrée complète supplémentaire.
L'arrêt anticipé peut modifier le coût total ; les temps des pilotes ne
garantissent pas une durée particulière sur une autre machine.

### 9.6. Vérifications et lancement de comparaison

La suite comprend **59 tests réussis**. Les nouveaux tests vérifient la
transformation contre une FEN symétrique encodée indépendamment, les pièces,
le trait et les roques, le retour à l'entrée initiale après deux transformations,
les permutations de classes en trois et cinq sorties, l'exclusion d'une
variante présente en validation, l'absence de mutation des entrées, le rejet
de données invalides et les effectifs/métadonnées d'un entraînement augmenté.

Les modèles livrés gardent le format `.nn` et ne demandent pas de transformation
particulière lors de la prédiction : l'augmentation est uniquement un traitement
d'entraînement. Le fichier `model.nn` à 71,10 % est conservé pendant ces essais.

Pour le nouveau lancement complet, avec une sortie distincte :

```bash
python3 src/my_torch.py -train \
  --dataset dataset/cleaned/train.txt --output model_colors.nn --classes 5 \
  --augment-colors --seed 42
```

La commande habituelle sans nouvelle option active également cette augmentation.
Ajouter `--no-augment-colors` permet de reproduire les réglages sans symétrie
de la référence à 71,10 %. Les sorties `model_*.nn` restent ignorées par Git.

## 10. Lancement complet avec symétrie : 75,27 % en validation

Le lancement complet communiqué après l'ajout de l'augmentation confirme un
gain sur la partition de validation utilisée pour la référence à 71,10 %.
Il ne s'agit plus d'une estimation issue d'un petit pilote. Les métadonnées
du `model.nn` local confirment les effectifs, la graine 42, le choix de l'époque
19 et la précision exacte 0,7526666667, arrondie à **75,27 %**.

### 10.1. Configuration et journal des 20 époques

```bash
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn --classes 5
```

Le fichier source fournit toujours 60 000 positions, 12 000 par classe. La
partition stratifiée reste 51 000 / 9 000 avant augmentation. Les 50 995 variantes
acceptées portent l'entraînement à **101 995 exemples** ; 5 variantes sont
exclues car elles reproduisent des entrées de validation. Les **9 000 positions
de validation sont inchangées**.

Le réseau reste 773 → 512 → 256 → 128 → 5 avec BatchNorm et Dropout 0,25.
Les paramètres sont identiques à la référence hors augmentation : CCE uniforme,
Adam à 0,001, L2 0,0005, lissage 0, lots de 64, graine 42, CosineAnnealingLR sur
20 époques, sélection sur `val_accuracy`, départage par `val_loss` et patience 6.

| Époque | Perte entraînement | Précision entraînement | Perte validation | Précision validation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,9282 | 55,83 % | 0,7839 | 63,40 % |
| 2 | 0,7792 | 64,08 % | 0,7323 | 66,29 % |
| 3 | 0,7304 | 66,65 % | 0,6999 | 67,79 % |
| 4 | 0,6941 | 68,67 % | 0,6835 | 68,92 % |
| 5 | 0,6601 | 70,53 % | 0,6515 | 70,77 % |
| 6 | 0,6311 | 72,15 % | 0,6417 | 70,68 % |
| 7 | 0,6010 | 73,68 % | 0,6321 | 72,00 % |
| 8 | 0,5702 | 75,23 % | 0,6203 | 72,39 % |
| 9 | 0,5350 | 77,14 % | **0,6099** | 73,27 % |
| 10 | 0,4973 | 79,01 % | 0,6183 | 73,48 % |
| 11 | 0,4656 | 80,63 % | 0,6157 | 73,89 % |
| 12 | 0,4191 | 82,83 % | 0,6117 | 74,33 % |
| 13 | 0,3746 | 84,92 % | 0,6328 | 74,90 % |
| 14 | 0,3309 | 86,81 % | 0,6470 | 74,59 % |
| 15 | 0,2914 | 88,53 % | 0,6665 | 74,57 % |
| 16 | 0,2547 | 90,14 % | 0,7065 | 74,97 % |
| 17 | 0,2280 | 91,34 % | 0,7165 | 74,80 % |
| 18 | 0,2027 | 92,40 % | 0,7370 | 75,26 % |
| 19 | 0,1880 | 92,90 % | **0,7587 — modèle retenu** | **75,27 %** |
| 20 | 0,1769 | 93,39 % | 0,7699 | 75,10 % |

Les meilleurs poids sont restaurés à l'époque 19. La meilleure perte reste
celle de l'époque 9 : 0,6099, avec 73,27 % de précision. Le critère de précision
retient donc 2,00 points de plus, au prix d'une perte plus élevée. La patience
ne se déclenche pas : une seule époque suit le dernier maximum avant la limite.

### 10.2. Comparaison complète avant/après augmentation

| Mesure | Sans symétrie, section 7 | Avec symétrie | Évolution |
| --- | ---: | ---: | --- |
| Exemples d'entraînement par époque | 51 000 | 101 995 | +50 995 |
| Positions de validation | 9 000 | 9 000 | Inchangées |
| Époque retenue | 17 | 19 | Critère identique |
| Bonnes réponses | 6 399 | **6 774** | **+375** |
| Précision validation | 71,10 % | **75,27 %** | **+4,17 points** |
| Erreurs | 2 601 | **2 226** | **−375, soit −14,42 %** |
| Perte CCE du modèle retenu | 1,0486 | **0,7587** | Environ −27,65 % |
| Meilleure perte observée | 0,7123 | **0,6099** | Minimum à l'époque 9 au lieu de 6 |
| Macro-F1 | 0,7098 | **0,7518** | +0,0420 sur les valeurs affichées |

Le changement améliore ici à la fois la précision, la macro-F1 et la perte CCE
du modèle retenu sur la même validation. Le traitement des données et le nombre
de mises à jour par époque ont changé ; ce gain ne doit pas être attribué à
une augmentation de capacité ou à un nouveau réglage de taux d'apprentissage.
Le contrôle par répétition de la section 9 apportait un premier élément pour
distinguer la symétrie de la seule répétition, mais n'a pas été réexécuté sur
le corpus complet.

Le premier journal historique à 76,60 % reste supérieur de 1,33 point au score
arrondi actuel. Sa partition non archivée diffère de la validation stratifiée
actuelle ; ce n'est donc pas le même protocole de comparaison que le passage
contrôlé de 71,10 % à 75,27 %.

### 10.3. Résultats par classe et erreurs restantes

| Classe | Précision | Rappel | F1 | Support | Rappel sans symétrie |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nothing | 77,82 % | **65,50 %** | 71,13 % | 1 800 | 56,06 % |
| Check White | 68,73 % | 72,78 % | 70,70 % | 1 800 | 70,28 % |
| Check Black | 71,26 % | 72,33 % | 71,79 % | 1 800 | 73,50 % |
| Checkmate White | 79,01 % | **83,00 %** | 80,95 % | 1 800 | 77,44 % |
| Checkmate Black | 80,01 % | **82,72 %** | 81,34 % | 1 800 | 78,22 % |

Le rappel de Nothing gagne environ 9,44 points. Les gains de bonnes réponses
par classe sont respectivement **+170, +45, −21, +100 et +81**, soit +375 au
total. Le rappel de Check Black recule légèrement ; sa précision et sa F1
augmentent. Le résultat ne constitue donc pas une amélioration de chaque
métrique de chaque classe.

La nouvelle matrice, lignes attendues et colonnes prédites, est :

| Attendu / prédit | Nothing | Check White | Check Black | Checkmate White | Checkmate Black |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nothing | 1 179 | 303 | 226 | 50 | 42 |
| Check White | 143 | 1 310 | 0 | 347 | 0 |
| Check Black | 168 | 0 | 1 302 | 0 | 330 |
| Checkmate White | 13 | 293 | 0 | 1 494 | 0 |
| Checkmate Black | 12 | 0 | 299 | 0 | 1 489 |

Les confusions Check ↔ Checkmate de même couleur passent de **1 440 à 1 269**,
soit 171 erreurs de moins. Elles représentent néanmoins **57,01 % des erreurs
restantes** : leur part augmente parce que d'autres erreurs diminuent plus vite.
Nothing reste la classe au rappel le plus faible, avec 621 positions manquées
sur 1 800. Aucune confusion de couleur opposée entre les quatre classes
colorées n'apparaît dans cette matrice.

### 10.4. Diagnostic et décision après ce résultat

L'augmentation est maintenue comme défaut : contrairement aux variantes
rejetées précédemment, son bénéfice a maintenant été observé sur le lancement
complet fourni. **Aucun autre paramètre numérique n'est modifié à partir de
ce seul journal.** L'architecture, le taux initial, le Dropout, la L2, la perte,
la patience et le lissage restent aux valeurs détaillées en section 9.

Le surapprentissage n'a pas disparu : après l'époque 9, la perte de validation
remonte jusqu'à 0,7587 sur le modèle retenu, tandis que la perte d'entraînement
continue de baisser. À l'époque 19, l'écart entre les précisions affichées est
d'environ **17,63 points** : 92,90 % contre 75,27 %, au lieu de 24,22 points sur
le modèle de référence. Cet écart est indicatif, car l'entraînement est mesuré
avec Dropout et porte maintenant sur un corpus augmenté.

Il s'agit toujours d'une validation équilibrée, sur une graine et une partition.
Ce résultat ne garantit pas le même gain sur des positions indépendantes.
Le journal ne contient pas d'évaluation sur `dataset/cleaned/test.txt` ; cette
étape reste distincte, après vérification des recouvrements entre les corpus.

L'utilisateur a choisi `--output model.nn` : le fichier local contient désormais
le modèle **à 75,27 %**, ce que confirment ses métadonnées. Le précédent modèle
à 71,10 % n'est plus le contenu de ce chemin. Aucun modèle n'a été réentraîné
ou remplacé par l'assistant pour rédiger cette section ; seuls les résultats
et les métadonnées existants ont été vérifiés.

## 11. Encodage tactique optionnel : modifications et pilote

### 11.1. Problème et objectif

Le dernier entraînement complet reste celui de la section 10 : **75,27 %**
sur 9 000 positions, avec 1 269 confusions Check ↔ Checkmate de même couleur.
Les 773 entrées historiques décrivent les pièces, le trait et les roques,
mais ne représentent pas explicitement les relations d'attaque. La case de
prise en passant est aussi absente. Le réseau doit donc apprendre ces relations
à partir des seules positions des pièces, alors que le surapprentissage est
déjà visible dans les résultats.

Le changement consiste à tester une représentation tactique plus explicite.
La classification reste apprise par le réseau supervisé ; les entrées ne
contiennent ni annotation attendue, ni verdict de classification par règles,
ni nombre de coups légaux.

### 11.2. Changements exacts

| Élément | Avant | Après |
| --- | --- | --- |
| Encodage | Unique, 773 entrées | `--encoding basic` : 773, défaut ; `--encoding tactical` : **1 333** |
| Première couche | `773 → 512` | `1333 → 512` en tactique : **286 720 poids supplémentaires** |
| Prise en passant | Ignorée | 64 indicateurs de case cible, tous nuls si absente |
| Cartes d'attaque | Absentes | 128 indicateurs : 64 cases × 2 couleurs |
| Lignes autour des rois | Absentes | 304 indicateurs : 2 rois × 8 directions × (12 pièces + 7 distances) |
| Voisinage des rois | Absent | 64 indicateurs : case valide, pièce amie, pièce adverse, attaque adverse après retrait du roi |
| Symétrie | Transformation des 773 entrées | Transformation de toutes les entrées tactiques, y compris la prise en passant |
| Encodage à l'inférence | Toujours 773 | Lu dans les métadonnées du modèle ; les anciens modèles gardent 773 |
| Validation | Séparation stratifiée 85/15 | Identique par défaut ; `--validation-dataset` ajoute une validation externe fixe |
| Graine du réseau | Identique à celle du corpus | Identique par défaut ; `--init-seed` permet de changer seulement l'aléatoire d'entraînement |
| Traçabilité | Effectifs | Effectifs et empreintes des partitions communes aux deux encodages |

Les cartes représentent des attaques géométriques, y compris celles des pièces
clouées et les cases occupées défendues. Les rayons s'arrêtent après la première
pièce rencontrée. Le voisinage des rois ne représente pas tous les coups légaux.
Les colonnes exactes et les directions sont décrites dans
[le guide du réseau](neural_network.md#encodage-tactique-optionnel).

Les nouveaux modèles enregistrent l'encodage et sa version dans leurs
métadonnées. L'évaluation, l'analyseur en prédiction/réentraînement et le bonus
jouable sélectionnent automatiquement cet encodage. Une dimension incompatible
ou une version inconnue est rejetée. Les anciens poids ne sont pas convertis :
un modèle tactique doit être entraîné avec ses nouvelles entrées.

Les autres paramètres restent identiques : **20 époques maximum, batch 64,
LR 0,001, Dropout 0,25, L2 0,0005, lissage 0, CCE uniforme, patience 6,
sélection par val_accuracy puis val_loss en cas d'égalité**, BatchNorm,
couches cachées 512/256/128 et symétrie de couleurs activées. Le changement
augmente également la capacité de la première couche ; aucun contrôle à nombre
de paramètres égal n'a été exécuté.

### 11.3. Comparaison effectivement observée avant interruption

Commande lancée :

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m src.experiments.compare_encodings \
  --dataset dataset/cleaned/train.txt --output-dir /tmp/torch_tactical_pilot_v1 \
  --max-per-class 1000 --epochs 20 --init-seeds 42 43 44
```

Les variantes utilisent les mêmes **5 000 positions**, 1 000 par classe,
sélectionnées et séparées avec la graine 42 : 4 250 positions originales
d'entraînement, 4 250 miroirs et **750 positions de validation**, soit 150 par
classe. Aucun miroir de validation n'est ajouté. L'audit a trouvé 5 000 identités
de plateau distinctes et aucun recouvrement entre entraînement original et
validation. L'identité conservative utilise les 773 colonnes historiques,
en ignorant les compteurs et la prise en passant, pour conserver les mêmes
exclusions entre représentations.

La graine d'initialisation change uniquement le réseau et son entraînement.
Le comparateur vérifie les empreintes des partitions et les effectifs
d'augmentation. Les deux paires entièrement observées donnent :

| Initialisation | Basic | Tactical | Gain | Époque retenue basic / tactical | Époques exécutées basic / tactical |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 61,07 % | **80,93 %** | **+19,87 points** | 5 / 9 | 11 / 15 |
| 43 | 60,93 % | **82,67 %** | **+21,73 points** | 8 / 13 | 14 / 19 |

Sur ces deux paires, la moyenne passe de **61,00 % à 81,80 %**, soit
**+20,80 points**. Il s'agit de deux initialisations sur une seule partition,
pas de deux corpus de validation indépendants.

Pour l'initialisation 42, les bonnes réponses passent de **458/750 à 607/750**,
la CCE de **0,9589 à 0,7148** et la macro-F1 de **0,6078 à 0,8069**.
Le rechargement des deux modèles par `src/my_torch.py -eval` sur les 750 FEN
originales, avec des lots de 64, a reproduit ces précisions et ces pertes.

La troisième référence, initialisation 44, a terminé à **60,00 %**, avec
l'époque 16 retenue sur 20. L'exécution a été interrompue pendant son essai
tactique : aucun résultat final n'a été relevé pour cette paire. Elle n'est
donc pas incluse dans la moyenne. Les fichiers temporaires des essais ne sont
plus disponibles à la reprise ; les valeurs ci-dessus sont celles relevées
pendant leur exécution. Les historiques complets n'ont pas été archivés dans
le dépôt. Le comparateur permet de les recréer dans un nouveau dossier.

### 11.4. Vérifications et utilisation

Avant l'interruption, **67 tests ont passé** avec :

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make tests_run
```

Ils couvrent les attaques comparées à une implémentation indépendante, les
bloqueurs et distances, les lignes ouvertes après retrait du roi, la prise en
passant, la symétrie sur des FEN et des plateaux aléatoires, l'exclusion des
miroirs de validation, la validation externe fixe, les erreurs d'annotation,
la sauvegarde, le rechargement, la prédiction et le réentraînement. Le
comparateur est aussi testé sur les deux représentations et refuse d'écraser
un dossier de résultats existant.

L'encodage tactique reste **optionnel**. Ses matrices d'entrées comportent
environ 72 % de colonnes supplémentaires, avec un coût accru en mémoire,
en encodage et dans la première couche.

**Ce pilote ne mesure pas le score sur les 9 000 positions du résultat à
75,27 %.** Aucun entraînement tactique complet sur les 60 000 positions n'a
été exécuté pendant cette modification. Pour le mesurer :

```sh
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt \
  --output model_tactical.nn --classes 5 --encoding tactical
```

Le modèle local `model.nn` à 75,27 % n'a pas été remplacé pendant ces essais ;
son empreinte avait été vérifiée après les cinq premiers entraînements.
Pour augmenter ensuite le corpus, utiliser `--validation-dataset` avec un
fichier fixe et réentraîner aussi la référence basic sous ce même protocole.
Les plateaux communs sont retirés de l'entraînement avant augmentation.
Le fichier de test reste réservé à la mesure finale de généralisation.

## 12. Complément : configurations et activation des nouvelles entrées

### 12.1. Problème rencontré lors de l'utilisation

Les modifications tactiques et leurs résultats pilotes sont détaillés en
section 11. Le lancement suivant a cependant conservé les résultats historiques :

```sh
./my_torch_generator full_network.conf 1
python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn --classes 5
```

La première commande crée `full_network_1.nn` avec des poids aléatoires.
La seconde construit un nouveau réseau à partir de `src/training/train.py` :
elle ne charge ni le `.conf`, ni le `.nn` généré. Comme `--encoding` vaut
`basic` par défaut, le journal affiche toujours **773 entrées**. La sélection
des positions, la séparation et l'entraînement utilisant la même graine 42,
retrouver les mêmes valeurs d'époque est cohérent avec ce protocole.

Il faut donc choisir explicitement la représentation tactique pour mesurer
l'effet des nouvelles informations. La création d'un fichier de configuration
ou d'un réseau vierge ne modifie pas le défaut du programme principal.

### 12.2. Fichiers de configuration ajoutés

Avant cette modification, seul `basic_network.conf` était disponible.
Trois configurations supplémentaires ont été ajoutées à la racine :

| Fichier | Couches denses | Sorties | BatchNorm | Dropout |
| --- | --- | ---: | --- | --- |
| `basic_network.conf`, existant | 773 → 128 → 64 → 3 | 3 | Non | 0,10 après la première couche cachée |
| `medium_network.conf`, nouveau | 773 → 512 → 256 → 128 → 3 | 3 | Après chaque couche cachée | 0,15 puis 0,10 ; aucun après la dernière |
| `full_network.conf`, nouveau | 773 → 512 → 256 → 128 → 5 | 5 | Après chaque couche cachée | 0,15 puis 0,10 ; aucun après la dernière |
| `tactical_network.conf`, nouveau | 1333 → 512 → 256 → 128 → 5 | 5 | Après chaque couche cachée | 0,25 après chacune des trois couches cachées |

Les configurations medium et full décrivent les couches des modèles
préentraînés livrés, avec leurs réglages historiques. La configuration tactique
reprend l'architecture actuelle du programme principal avec `--encoding tactical`.
Les couches cachées utilisent ReLU, puis la sortie utilise Softmax.
L'initialisation des couches denses cachées est He ; celle de la sortie est Xavier.

Un `.conf` décrit les couches et leur initialisation. Un `.nn` contient aussi
les valeurs des poids et des statistiques BatchNorm sauvegardées. Générer un
réseau depuis le `.conf` ne reproduit donc pas les poids d'un modèle entraîné.
Ces fichiers ne règlent pas le nombre d'époques, la séparation de validation,
la perte ou le taux d'apprentissage du programme principal.

### 12.3. Adaptation du générateur et compatibilité

Le générateur accepte désormais une directive optionnelle :

```text
encoding basic
```

ou :

```text
encoding tactical
```

Elle enregistre l'encodage et la version 1 dans les métadonnées du réseau,
vérifie que la première couche reçoit respectivement **773 ou 1 333 entrées**
et sauvegarde les classes usuelles pour les réseaux à trois ou cinq sorties.
Une directive répétée, mal formée, un encodage inconnu ou une dimension
incompatible est rejeté. Les anciennes configurations sans cette directive
restent acceptées, y compris les configurations de réseaux génériques.

L'analyseur et l'évaluation retrouvent ainsi la représentation correcte à
partir du `.nn`. La symétrie de couleurs et les 560 entrées supplémentaires
restent celles décrites en section 11 ; l'ajout des `.conf` ne les modifie pas.

### 12.4. Commandes et différence entre les deux parcours

Pour entraîner la variante tactique avec le protocole de validation du rapport :

```sh
python3 src/my_torch.py -train \
  --dataset dataset/cleaned/train.txt \
  --output model_tactical.nn \
  --classes 5 \
  --encoding tactical
```

Le journal doit afficher `feature dimension: 1333`. Le programme conserve
20 époques maximum, batch 64, LR 0,001, Dropout 0,25 après chaque couche cachée,
L2 0,0005, validation stratifiée 85/15 par défaut, symétrie de couleurs et
restauration du meilleur modèle. **`--classes 5` choisit les sorties ;
`--encoding tactical` choisit les entrées.** Ce sont deux réglages distincts.

Pour utiliser précisément l'architecture d'un `.conf` :

```sh
./my_torch_generator tactical_network.conf 1
./my_torch_analyzer --train --save model_tactical_custom.nn \
  tactical_network_1.nn dataset/cleaned/train.txt
```

Ce second parcours charge le réseau généré. L'analyseur l'entraîne actuellement
pendant **15 époques, avec des lots de 32**, sur le fichier fourni, sans la
partition de validation ni l'augmentation du programme principal. Ces deux
parcours ne constituent donc pas une comparaison contrôlée des architectures.
Pour reproduire les mesures du rapport, utiliser le premier parcours.

### 12.5. Vérifications et état des résultats

Lors de l'ajout des configurations, **70 tests ont passé**, dont la génération,
la sauvegarde, le rechargement et la prédiction avec les quatre configurations.
Les couches générées pour medium et full ont été comparées à celles des modèles
livrés. Les tests vérifient également le rejet des directives invalides et la
compatibilité des configurations génériques sans encodage explicite.

L'ajout des `.conf` n'apporte pas à lui seul un nouveau score mesuré. Les deux
paires pilotes complètes de la section 11 donnent toujours **61,00 % en moyenne
en basic contre 81,80 % en tactical**, sur la même validation de 750 positions.
Le résultat complet à 75,27 % porte sur une autre partition de 9 000 positions.
Aucun nouveau résultat d'entraînement tactique complet n'a été fourni pour ce
complément ; le gain sur ces 9 000 positions reste à mesurer.


## 13. Entraînement tactique complet : 94,26 % de validation

### 13.1. Commande, configuration et vérification des données

Commande exécutée par l'utilisateur :

```sh
python3 src/my_torch.py -train \
  --dataset dataset/cleaned/train.txt \
  --output model_tactical.nn \
  --classes 5 \
  --encoding tactical
```

Ce lancement active effectivement les **1 333 entrées tactiques**, contre
773 pour le modèle de référence. Les résultats ci-dessous proviennent du
journal fourni et sont confirmés par les métadonnées de `model_tactical.nn`.
Il s'agit maintenant d'un entraînement complet, distinct du pilote sur
5 000 positions décrit en section 11.

| Paramètre | Référence basic | Modèle tactical |
| --- | --- | --- |
| Entrées | 773 | **1 333** |
| Couches denses | 773 → 512 → 256 → 128 → 5 | 1333 → 512 → 256 → 128 → 5 |
| Positions sélectionnées | 60 000, soit 12 000 par classe | Identiques |
| Sélection / séparation / initialisation | Graine 42 | Graine 42 |
| Entraînement original / validation | 51 000 / 9 000 | Identiques |
| Miroirs ajoutés / exclus car présents en validation | 50 995 / 5 | Identiques |
| Exemples d'entraînement après augmentation | 101 995 | Identiques |
| Époques maximum / batch | 20 / 64 | Identiques |
| LR initial / ordonnanceur | 0,001 / CosineAnnealingLR | Identiques |
| Dropout / L2 | 0,25 / 0,0005 | Identiques |
| Perte / poids de classes / lissage | CCE / uniformes / 0 | Identiques |
| BatchNorm / patience | Activée / 6 | Identiques |
| Sélection | val_accuracy, puis val_loss en cas d'égalité | Identique |

Les métadonnées des deux fichiers confirment **les mêmes empreintes de
partitions avant augmentation**, calculées sur les entrées historiques et
les annotations dans leur ordre :

- Entraînement : `490ca55f399a7c266dfe7067f05d719cf3a301061479effb431dac163ab1260c`.
- Validation : `0f62aae7c8876d64eeb9f566dc6b975c59c4e868dada731bce9bb185982edec8`.

Les effectifs et exclusions d'augmentation correspondent aussi. La comparaison
avec les 75,27 % de la section 10 porte donc sur les mêmes positions de validation
et le même protocole de sélection du modèle. La variante change la représentation
et augmente la capacité de la première couche de 286 720 poids ; ces deux effets
ne sont pas isolés par un contrôle à nombre de paramètres égal.

### 13.2. Historique des 20 époques

| Époque | Perte entraînement | Précision entraînement | Perte validation | Précision validation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,5406 | 77,45 % | 0,3393 | 86,09 % |
| 2 | 0,3535 | 85,59 % | 0,3121 | 86,97 % |
| 3 | 0,3219 | 87,04 % | 0,2837 | 88,27 % |
| 4 | 0,3028 | 87,83 % | 0,2723 | 88,72 % |
| 5 | 0,2814 | 88,79 % | 0,2625 | 89,34 % |
| 6 | 0,2587 | 89,89 % | 0,2424 | 90,28 % |
| 7 | 0,2362 | 90,82 % | 0,2157 | 91,33 % |
| 8 | 0,2178 | 91,61 % | 0,2063 | 91,93 % |
| 9 | 0,1998 | 92,38 % | 0,2038 | 91,96 % |
| 10 | 0,1806 | 93,16 % | 0,1975 | 92,41 % |
| 11 | 0,1623 | 93,93 % | 0,1815 | 93,13 % |
| 12 | 0,1432 | 94,62 % | 0,1793 | 93,42 % |
| 13 | 0,1264 | 95,27 % | 0,1725 — minimum de perte | 93,73 % |
| 14 | 0,1091 | 96,04 % | 0,1730 | 93,61 % |
| 15 | 0,0916 | 96,70 % | 0,1821 | 93,92 % |
| 16 | 0,0746 | 97,32 % | 0,1781 | 94,14 % |
| 17 | 0,0630 | 97,72 % | 0,1851 | 93,80 % |
| 18 | 0,0517 | 98,21 % | 0,1921 — modèle retenu | 94,26 % |
| 19 | 0,0455 | 98,47 % | 0,1951 | 94,26 % |
| 20 | 0,0407 | 98,60 % | 0,1964 | 94,23 % |

Les meilleurs poids sont restaurés à **l'époque 18**, avec **94,26 %** de
précision et une CCE de **0,1921**. Les époques 18 et 19 donnent exactement
le même nombre de bonnes réponses : 8 483 sur 9 000. L'époque 18 est retenue
car sa perte est plus faible : **0,1921 contre 0,1951**. L'époque 20 termine
à 94,23 %, mais ses poids ne constituent pas le modèle sauvegardé.

La perte minimale est atteinte à **l'époque 13 : 0,1725**, avec 93,73 % de
précision. La sélection par précision retient donc 47 bonnes réponses de plus,
soit environ **0,52 point**, au prix d'une CCE plus élevée. L'arrêt anticipé
ne se déclenche pas avant la limite de 20 époques.

### 13.3. Comparaison avec le modèle basic à 75,27 %

| Mesure | Basic, section 10 | Tactical | Évolution |
| --- | ---: | ---: | --- |
| Époque retenue | 19 | 18 | Même critère de sélection |
| Bonnes réponses / 9 000 | 6 774 | **8 483** | **+1 709** |
| Précision validation | 75,27 % | **94,26 %** | **+18,99 points** |
| Erreurs | 2 226 | **517** | **−1 709, soit −76,77 %** |
| CCE du modèle retenu | 0,7587 | **0,1921** | Environ **−74,68 %** |
| Meilleure CCE observée | 0,6099, époque 9 | **0,1725, époque 13** | Minimum plus faible |
| Macro-F1 | 0,7518 | **0,9424** | +0,1906 sur les valeurs affichées |
| Écart entraînement / validation à l'époque retenue | 17,63 points | **3,95 points** | Écart réduit |

Les différences sont calculées à partir des valeurs exactes des métadonnées,
puis arrondies pour l'affichage. Le gain apparaît également dès la première
époque tactique : **86,09 %**, contre 75,27 % pour le meilleur modèle basic.
L'encodage tactique améliore ici la précision, la perte et la macro-F1 sur
la validation complète ; son bénéfice ne repose plus uniquement sur le pilote.

### 13.4. Métriques par classe et matrice de confusion

| Classe | Précision | Rappel | F1 | Support | Rappel basic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nothing | 95,97 % | 96,56 % | 96,26 % | 1 800 | 65,50 % |
| Check White | 94,22 % | 91,50 % | 92,84 % | 1 800 | 72,78 % |
| Check Black | 96,00 % | 90,56 % | 93,20 % | 1 800 | 72,33 % |
| Checkmate White | 92,93 % | 95,61 % | 94,25 % | 1 800 | 83,00 % |
| Checkmate Black | 92,38 % | 97,06 % | 94,66 % | 1 800 | 82,72 % |

Le rappel augmente pour les cinq classes. Les gains de bonnes réponses par
classe, dans l'ordre du tableau, sont **+559, +337, +328, +227 et +258**,
soit +1 709 au total. Le rappel de Nothing passe de 65,50 % à **96,56 %** :
les positions de cette classe manquées passent de 621 à **62**.

Matrice de confusion, lignes attendues et colonnes prédites :

| Attendu / prédit | Nothing | Check White | Check Black | Checkmate White | Checkmate Black |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nothing | 1 738 | 28 | 21 | 9 | 4 |
| Check White | 31 | 1 647 | 0 | 122 | 0 |
| Check Black | 30 | 0 | 1 630 | 0 | 140 |
| Checkmate White | 6 | 73 | 0 | 1 721 | 0 |
| Checkmate Black | 6 | 0 | 47 | 0 | 1 747 |

Les confusions Check ↔ Checkmate de même couleur passent de **1 269 à 382**,
soit **887 erreurs de moins**. Elles représentent désormais **73,89 % des
517 erreurs restantes** : elles diminuent en nombre, mais concentrent une plus
grande part des erreurs résiduelles. Aucune confusion entre couleurs opposées
n'apparaît entre les quatre classes colorées dans cette matrice.

Le rappel le plus faible est désormais celui de **Check Black : 90,56 %**,
avec 170 erreurs sur 1 800 positions, dont 140 prédites Checkmate Black.
Pour Check White, 122 positions sont prédites Checkmate White. La distinction
entre un échec et un mat reste donc le principal point à examiner.

### 13.5. Diagnostic, portée du résultat et modèle sauvegardé

La progression de la précision de validation se poursuit jusqu'à l'époque 18.
Un surapprentissage reste visible après l'époque 13 : la CCE de validation
remonte de 0,1725 à 0,1964 à l'époque 20, tandis que la perte d'entraînement
continue à diminuer. Pour le modèle retenu, l'entraînement affiche **98,21 %**
contre **94,26 %** en validation, soit un écart d'environ **3,95 points**.
Cet écart est indicatif : l'entraînement est mesuré avec Dropout sur le corpus
augmenté, contrairement à la validation.

Il s'agit d'un résultat sur une seule partition et une seule initialisation
pour ce lancement complet. Il ne constitue ni une garantie de 100 % sur de
nouvelles positions, ni une mesure sur le fichier de test indépendant.
Le journal fourni ne contient pas d'évaluation sur `dataset/cleaned/test.txt`.
Le maintien d'un test distinct et l'examen des erreurs restantes permettront
de préciser la généralisation.

Le fichier **`model_tactical.nn` contient le modèle de l'époque 18**, son
encodage tactique et ses métriques de validation. Le chemin `model.nn`
correspond au modèle basic de référence à 75,27 % lors de la vérification.
Cette section consigne le nouvel entraînement fourni : aucun hyperparamètre,
poids de modèle ou code d'entraînement n'a été modifié pour sa rédaction.
