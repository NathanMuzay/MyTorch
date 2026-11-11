# Construire et défendre son premier réseau de neurones

Cours progressif pour MY_TORCH. Public : une personne qui sait écrire des fonctions, des boucles et lire un fichier, mais n'a jamais construit de système neuronal.

Le but est double : savoir réaliser un classifieur d'échiquiers sans bibliothèque neuronale, et savoir expliquer ses choix en reliant **besoin, solution, limite, mesure et correction**. Ce cours couvre les fondations nécessaires et les principales alternatives pertinentes pour le projet. Il ne prétend pas couvrir tout le domaine de l'intelligence artificielle.

Les chiffres de configuration proposés sont des points de départ, pas des résultats expérimentaux. Les petits exemples numériques sont pédagogiques. Une performance sur XOR ne prédit pas une performance aux échecs.

## Sommaire

1. [Ce que demande le projet](#1-ce-que-demande-le-projet)
2. [Le vocabulaire et le mécanisme général](#2-le-vocabulaire-et-le-mécanisme-général)
3. [Les mathématiques indispensables](#3-les-mathématiques-indispensables)
4. [Le neurone, les couches et les architectures](#4-le-neurone-les-couches-et-les-architectures)
5. [Choisir les sorties](#5-choisir-les-sorties)
6. [Mesurer l'erreur](#6-mesurer-lerreur)
7. [Calculer les gradients](#7-calculer-les-gradients)
8. [Mettre à jour les paramètres](#8-mettre-à-jour-les-paramètres)
9. [Construire le premier moteur](#9-construire-le-premier-moteur)
10. [Transformer une FEN en entrée](#10-transformer-une-fen-en-entrée)
11. [Construire des données fiables](#11-construire-des-données-fiables)
12. [Éviter le surapprentissage](#12-éviter-le-surapprentissage)
13. [Mesurer les résultats](#13-mesurer-les-résultats)
14. [Optimiser sans tâtonner](#14-optimiser-sans-tâtonner)
15. [Livrer une bibliothèque et un analyseur](#15-livrer-une-bibliothèque-et-un-analyseur)
16. [Diagnostiquer les problèmes](#16-diagnostiquer-les-problèmes)
17. [Parcours de réalisation et critères de réussite](#17-parcours-de-réalisation-et-critères-de-réussite)
18. [Justifier les choix à la soutenance](#18-justifier-les-choix-à-la-soutenance)
19. [Exercices et questions de contrôle](#19-exercices-et-questions-de-contrôle)
20. [Références et glossaire](#20-références-et-glossaire)

## 1. Ce que demande le projet

Le [sujet MY_TORCH](../my_torch-project.pdf), version 2.6.0 présente deux composants :

| Composant | Responsabilité |
|---|---|
| Bibliothèque neuronale générique | Créer un réseau, l'entraîner, sauvegarder et charger ses paramètres |
| `my_torch_analyzer` | Lire des positions FEN ; entraîner un modèle ou produire ses prédictions |

Les points à respecter sont notamment :

- solution fondée sur le machine learning et entraînement supervisé ;
- bibliothèques qui prennent en charge les réseaux de neurones interdites, dont PyTorch et TensorFlow ;
- classes de base : `Nothing`, `Check`, `Checkmate` ; traitement du pat en bonus ;
- en prédiction, une sortie par position, dans l'ordre du fichier ;
- réseau préentraîné livré, dont le nom commence par `my_torch_network` ;
- scripts d'entraînement conservés ; datasets conservés et présentables à la soutenance, sans obligation de les pousser dans le dépôt ;
- gestion du surapprentissage ; démonstration de méthodes d'analyse et d'optimisation ;
- README obligatoire ; benchmarks et documentation explicative recommandés, avec preuves visuelles ;
- erreurs sur stderr, code de sortie 84 ; succès avec code 0 ; règles Makefile demandées si compilation nécessaire.

Le sujet laisse le format de sauvegarde et l'architecture au choix. Son générateur en ligne de commande est un exemple de solution, pas l'unique forme imposée. Il faut vérifier le contrat exact de l'évaluation pour les sorties avec couleur : les exemples montrent `Check White` et `Checkmate Black`, mais il ne faut pas inventer leur sens.

**État du dépôt au moment de la rédaction :** `src/my_torch.py` contient essentiellement des messages d'affichage. Le README décrit davantage de fonctionnalités ; ces descriptions ne sont pas une preuve qu'elles sont implémentées ou mesurées. Les benchmarks du cours sont à réaliser.

### Ce que signifie « sans bibliothèque qui fait tout »

Tu dois écrire la propagation avant, les dérivées, la rétropropagation, les mises à jour, la boucle d'entraînement et la sérialisation.

| Approche | Intérêt | Limite |
|---|---|---|
| Python et bibliothèque standard | Toutes les opérations sont visibles ; installation simple | Boucles lentes sur les grands tableaux |
| Python et NumPy pour les tableaux | Produits matriciels rapides ; calcul des gradients toujours écrit par toi | Dépendance supplémentaire ; autorisation à vérifier dans les règles applicables |
| C/C++ et tableaux | Contrôle mémoire et performances | Plus de travail de gestion mémoire et de débogage |
| Framework neuronal ou moteur d'autodiff | Beaucoup de calculs pris en charge | Contredit l'objectif de cette implémentation et les interdictions du sujet |

Le compagnon de ce cours utilise seulement la bibliothèque standard Python. Tu peux commencer ainsi puis vectoriser les mêmes équations si les dépendances retenues sont autorisées. Un outil d'échecs servant à générer des labels ne doit pas devenir le mécanisme de prédiction à la place du réseau.

## 2. Le vocabulaire et le mécanisme général

Un réseau est une fonction réglable `f(x; θ)` : `x` est une entrée numérique ; `θ` regroupe les poids et biais. L'entraînement cherche des paramètres qui produisent les bonnes réponses sur des exemples.

**Apprentissage supervisé :** tu fournis des couples `(entrée, réponse correcte)`. Le réseau ne découvre pas seul ce que les mots « échec » ou « mat » signifient. Il reçoit un signal d'erreur calculé à partir de tes labels.

```text
FEN -> validation -> encodage x -> réseau -> scores -> classe prédite
                                     |
                 label attendu -> perte L
                                     |
                             gradients de L
                                     |
                           mise à jour des poids
```

| Terme | Sens concret |
|---|---|
| Paramètre | Poids ou biais appris pendant l'entraînement |
| Hyperparamètre | Réglage choisi : nombre de couches, learning rate, batch size... |
| Inférence | Utilisation des paramètres appris, sans les modifier |
| Batch | Groupe d'exemples utilisé pour une mise à jour |
| Epoch | Passage complet sur le jeu d'entraînement |
| Loss / perte | Nombre qui mesure l'erreur optimisée |
| Gradient | Dérivées indiquant la sensibilité de la perte aux paramètres |
| Généralisation | Réussite sur des exemples nouveaux |

La loss et le taux de réussite sont différents. Une prédiction correcte à 51 % et une autre à 99 % comptent toutes deux comme correctes ; elles n'ont pas la même entropie croisée.

**À savoir expliquer :** un réseau n'applique pas automatiquement les règles exactes des échecs. Il apprend une approximation à partir des données. Une grande confiance affichée ne garantit pas une réponse juste.

## 3. Les mathématiques indispensables

### 3.1 Scalaires, vecteurs et matrices

Un scalaire est un nombre. Un vecteur est une liste de nombres. Une matrice est un tableau de nombres. Pour le cours, les exemples d'un batch sont des **lignes**.

Un produit scalaire multiplie les coordonnées correspondantes puis additionne :

```text
x = [1, 2] ; w = [0.5, -0.2]
x · w = 1 × 0.5 + 2 × (-0.2) = 0.1
```

Pour une couche dense :

```text
X : B × D       B exemples, D entrées
W : D × H       H neurones de sortie
b : H           un biais par neurone
Z = XW + b      forme B × H
```

Le biais est ajouté à chaque ligne. Le produit matriciel n'est pas une multiplication coordonnée par coordonnée. `A ⊙ B` désigne cette dernière ; `AB` désigne le produit matriciel. `Xᵀ` est la transposée : on échange lignes et colonnes.

Exemple avec une entrée et deux neurones :

```text
X = [[1, 2]]
W = [[0.5, 0.3], [-0.2, 0.4]]
b = [0.1, -0.1]
Z = [[0.2, 1.0]]
```

**Exercice :** calcule le résultat pour `X = [[0, 1]]`. Réponse : `[[-0.1, 0.3]]`. Si tu ne retrouves pas ces résultats, travaille les produits matriciels avant de coder l'entraînement.

### 3.2 Dérivée et gradient

La dérivée mesure comment une sortie change quand on change légèrement une entrée. Pour `f(w)=w²`, `f'(w)=2w`. À `w=3`, une petite augmentation de `w` augmente `f` d'environ six fois cette variation.

Une dérivée partielle change une seule variable en maintenant les autres fixes. Le gradient regroupe toutes ces dérivées. Il pointe localement vers la plus forte augmentation ; on utilise son opposé pour réduire la perte.

### 3.3 Règle de la chaîne

Si `z = wx + b`, `a = f(z)` et `L = g(a)`, alors :

```text
∂L/∂w = (∂L/∂a) × (∂a/∂z) × (∂z/∂w)
       = (∂L/∂a) × f'(z) × x
```

Ce calcul traverse les opérations en sens inverse. C'est le principe de la rétropropagation. On ne « remonte » pas la classe prédite ; on propage des sensibilités numériques.

### 3.4 Logarithme et exponentielle

`exp(z)` est toujours positif. `log(p)` pour `0 < p ≤ 1` est négatif ou nul. `-log(p)` devient grand quand la probabilité de la bonne classe est petite. Ces fonctions vont transformer des scores en probabilités puis en erreur.

## 4. Le neurone, les couches et les architectures

### 4.1 Un neurone

Un neurone calcule `z = somme(x_i × w_i) + b`, puis `a = activation(z)`.

Le poids règle l'influence d'une entrée. Le biais décale le seuil. L'activation introduit une non-linéarité. Plusieurs couches linéaires sans activation se réduisent à une seule transformation affine : empiler des matrices seules n'ajoute pas la capacité recherchée.

### 4.2 Choisir une activation

| Activation | Formule et dérivée | Avantage | Défaut et réponse possible |
|---|---|---|---|
| ReLU | `max(0,z)` ; dérivée 1 si `z>0`, 0 si `z<0` | Très simple, pas de saturation du côté positif | Neurones toujours inactifs ; surveiller l'activité, réduire le pas ou essayer LeakyReLU |
| LeakyReLU | `z` si positif, `αz` sinon | Gradient non nul côté négatif | `α` à fixer ; ne garantit pas de meilleurs résultats |
| Sigmoid | `1/(1+exp(-z))` ; `a(1-a)` | Une probabilité pour une sortie binaire | Saturation dans les couches cachées ; petits gradients |
| Tanh | `tanh(z)` ; `1-a²` | Valeurs centrées entre -1 et 1 | Saturation pour de grandes valeurs absolues |
| Identité | `z` ; dérivée 1 | Sortie réelle pour régression ; scores avant softmax | Ne crée aucune non-linéarité |

À zéro, ReLU n'est pas dérivable ; on choisit une convention, généralement 0. La vérification numérique des gradients doit éviter ce point. Le compagnon utilise tanh pour disposer d'une activation lisse sur ce petit exercice.

**Choix de départ pour les échiquiers :** ReLU dans les couches cachées. Son calcul et sa dérivée sont faciles à contrôler. Si beaucoup de neurones sont toujours à zéro, examiner le learning rate, l'initialisation et les données avant d'ajouter des couches. Le [cours de Stanford sur les activations](https://cs231n.github.io/neural-networks-1/) compare notamment saturation et neurones ReLU inactifs.

### 4.3 Les architectures possibles

| Architecture | Capacité et intérêt | Coût ou difficulté | Place dans ce projet |
|---|---|---|---|
| Classifieur linéaire + softmax | Baseline simple et rapide | Peu d'interactions entre pièces | Mesurer ce qu'apportent les couches cachées |
| MLP : couches denses + activations | Apprend des interactions non linéaires | Ne possède pas de notion native de voisinage ou de déplacements | Premier véritable réseau conseillé |
| CNN : convolutions sur plans 8 × 8 | Partage des motifs spatiaux et des poids | Convolution et backward à écrire ; attaques longues à représenter | Expérience ultérieure si MLP validé |
| Réseau récurrent | Traite une séquence | La tâche utilise un état, pas nécessairement une histoire | Complexité peu justifiée au départ |
| Attention / Transformer | Met en relation des éléments éloignés | Davantage d'opérations et de réglages | Extension exigeante, pas une première étape |

Un échiquier ressemble à une grille, mais cela ne prouve pas qu'un CNN sera meilleur. La géométrie des règles et la distribution des données comptent. Une architecture doit gagner une comparaison mesurée pour justifier son coût.

### 4.4 Nombre de paramètres

Une couche `D -> H` contient `D × H + H` paramètres. Exemple de MLP :

```text
773 -> 128 -> 64 -> 3
773×128+128 + 128×64+64 + 64×3+3 = 107 523 paramètres
```

Avec cinq sorties : `107 653`. Ce sont des configurations de départ, pas des architectures optimales établies. Plus de paramètres peut aider l'ajustement, mais augmente calcul, mémoire et risque de mémorisation. Commencer avec `entrée -> 64 -> 3` permet une première comparaison moins coûteuse.

## 5. Choisir les sorties

### 5.1 Une sortie par classe avec softmax

Si une position appartient à exactement une catégorie, utiliser `K` scores, appelés **logits**, puis :

```text
p_k = exp(z_k) / somme_j exp(z_j)
```

Toutes les valeurs sont positives et leur somme vaut 1. `argmax(p)` donne l'indice de la plus grande. Il donne aussi le même indice que `argmax(z)` ; softmax sert notamment à l'entraînement et à l'interprétation probabiliste.

```text
Ordre : [Nothing, Check, Checkmate]
Sortie : [0.03, 0.91, 0.06] -> Check
Label de Check : indice 1, ou vecteur [0, 1, 0]
```

Le vecteur cible est appelé **one-hot**. Les poids appris ne sont pas limités à 0 et 1. Ce sont les labels qui indiquent une classe unique.

Pour inclure la couleur, une possibilité est cinq classes : `Nothing`, `Check Black`, `Check White`, `Checkmate Black`, `Checkmate White`. Définir le sens de la couleur et conserver le même ordre dans le dataset, la sauvegarde et l'affichage.

**Convention illustrative de ce cours : couleur = camp qui attaque**, donc, pour une position légale où le joueur au trait est en échec, couleur opposée au joueur au trait. Confirmer cette convention avec le contrat attendu avant de l'appliquer à l'analyseur.

Sur des positions légales et ce contrat, on peut également apprendre trois classes et déduire la couleur du joueur au trait. Cela réduit la sortie, mais repose sur une convention et des données valides. Évaluer ces hypothèses, surtout avec des positions artificielles.

### 5.2 Pourquoi ton code `00010`, `00100`, etc. est plus difficile

Ton idée encode plusieurs propriétés dans cinq bits : rien, échec ou mat, et éventuellement couleur. Le dernier bit est toujours 0 dans les exemples : il ne transporte aucune information. Les codes `Check` sans couleur et `Check White/Black` introduisent aussi des niveaux de précision différents dans les labels.

Avec cinq sigmoids et un seuil par bit, le réseau peut produire un état inconnu, deux couleurs, ou échec et mat simultanément. Le seuil ajoute un réglage et deux codes différents peuvent donner la même erreur binaire sans la même conséquence métier.

Ce codage n'est pas mathématiquement impossible. On pourrait apprendre les bits puis choisir le code valide le plus compatible avec leurs probabilités. Mais on ajoute un décodeur, et l'objectif par bit ne correspond pas directement à l'erreur de classe.

| Sortie | Quand elle convient | Ce qu'il faut gérer |
|---|---|---|
| Softmax à 3 ou 5 classes | Une catégorie exclusive par position | Ordre des classes ; classes rares |
| Plusieurs sigmoids | Plusieurs propriétés peuvent être vraies ensemble | Seuils ; cohérence entre propriétés |
| Deux têtes : état + couleur | Deux tâches liées ; couleur applicable selon l'état | Masquer la perte couleur pour `Nothing` ; arbitrer les deux pertes |
| Code binaire compact | Grand nombre de classes, intérêt réel au codage | Décodeur et combinaisons invalides |

Pour `Nothing / Check / Checkmate`, choisir une softmax est facile à défendre : objectif et décodage correspondent à une classe exclusive. Attention : « être en échec » et « être mat » sont liés dans les règles ; les **étiquettes** restent exclusives grâce à la priorité `mat`, puis `échec`, puis `rien`.

### 5.3 Softmax stable

Calculer directement `exp(1000)` déborde. Retrancher le plus grand score ne change pas les probabilités :

```python
from math import exp

def softmax(logits):
    maximum = max(logits)
    values = [exp(z - maximum) for z in logits]
    total = sum(values)
    return [value / total for value in values]
```

Exercice : softmax de `[1000,1000,1000]` doit donner trois valeurs proches de `1/3`, sans dépassement. Rejeter ou diagnostiquer les valeurs non finies en amont ; cette stabilisation ne répare pas des poids déjà `NaN`.

## 6. Mesurer l'erreur

### 6.1 Entropie croisée multiclasses

Pour le vrai label `y` et la probabilité `p_y` : `L = -log(p_y)`. Sur `B` exemples, utiliser la moyenne :

```text
L = -(1/B) × somme_n log(p[n, y_n])
```

Exemples : bonne classe à `0.9` donne une perte d'environ `0.105` ; à `0.1`, environ `2.303`. La prédiction très fausse est davantage pénalisée.

Pour une implémentation stable à partir des logits :

```text
m = max(z)
L = log(somme_k exp(z_k - m)) - (z_y - m)
```

Cela évite `log(0)` même si une probabilité très petite est arrondie à zéro. Pour un batch, calculer cette expression sur chaque ligne puis moyenner.

### 6.2 Autres pertes

| Perte | Usage naturel | Limite pour notre classification |
|---|---|---|
| Entropie croisée catégorielle | Une classe parmi K, avec softmax | Sensible aux labels incorrects ; peut rendre le réseau trop confiant |
| Entropie croisée binaire | Une cible 0/1 avec sigmoid ; plusieurs labels indépendants | Ne force pas des catégories mutuellement exclusives |
| MSE : moyenne des erreurs au carré | Régression de valeurs réelles | Possible pour des probabilités, mais gradients différents et souvent moins adaptés à cet objectif |
| Entropie croisée pondérée | Donner plus de poids à des classes rares | Modifie l'objectif et peut déplacer la calibration ; normalisation à préciser |

Ne pas combiner une formule de backward de softmax + entropie croisée avec une loss MSE. Un gradient doit être celui de la fonction effectivement affichée.

### 6.3 Une première référence numérique

Sur un réseau donnant des probabilités uniformes, `L = log(K)` : environ `1.099` pour trois classes et `1.609` pour cinq. Ce n'est pas une obligation pour chaque initialisation aléatoire : des logits non égaux donnent d'autres probabilités.

Une loss qui baisse montre que l'objectif d'entraînement est réduit ; elle ne prouve pas la généralisation. Il faut une validation indépendante.

## 7. Calculer les gradients

### 7.1 Exemple complet avec un neurone

Prenons une régression pédagogique : `x=2`, cible `y=1`, `w=0.2`, `b=0.1`, activation identité, perte `L=(a-y)²/2`.

```text
z = wx+b = 0.5 ; a = 0.5 ; L = 0.125
∂L/∂a = a-y = -0.5
∂L/∂w = -0.5 × x = -1
∂L/∂b = -0.5
```

Avec un pas `η=0.1`, `w <- 0.2 - 0.1×(-1) = 0.3`, `b <- 0.15`. La nouvelle sortie est `0.75` et la perte `0.03125`. Tu viens de réaliser une mise à jour d'apprentissage.

Le signe négatif explique qu'ici on augmente les paramètres. On ne décide pas manuellement « augmenter les poids » : le gradient donne la direction locale.

### 7.2 Couche dense sur un batch

Si `Z=XW+b` et si `G=∂L/∂Z` a la forme `B×H`, alors :

```text
dW = XᵀG                  D × H
db = somme des lignes G   H
dX = GWᵀ                  B × D
```

`dX` transmet le gradient à la couche précédente. Les gradients de poids et biais sont stockés pour l'optimiseur.

### 7.3 Activation

Si `A=ReLU(Z)` et `U=∂L/∂A`, alors `G=U ⊙ (Z>0)`. Pour tanh : `G=U ⊙ (1-A²)`. Conserver `Z` ou `A` lors du forward pour faire ce calcul.

### 7.4 Softmax et entropie croisée ensemble

Pour les probabilités `P` et les cibles one-hot `Y`, la dérivée par rapport aux **logits** est :

```text
G = (P-Y)/B
```

Avec `P=[0.2,0.7,0.1]` et classe attendue 1, sur un seul exemple : `G=[0.2,-0.3,0.1]`. Pour les logits considérés directement, la descente diminue les scores des mauvaises classes et augmente celui de la bonne.

La simplification inclut déjà la dérivée de softmax. **Ne pas refaire un backward softmax après `P-Y`.** Diviser par le batch une seule fois ; si chaque couche redivise, l'apprentissage dépend incorrectement de sa profondeur.

### 7.5 Ordre des opérations

```text
forward couches 1, 2, 3
loss et gradient des logits
backward couche 3, puis 2, puis 1
mise à jour de tous les paramètres
```

Tous les gradients d'une étape doivent correspondre aux poids utilisés pendant son forward. Modifier `W` avant de calculer `dX` mélangerait deux versions du réseau.

### 7.6 Vérification numérique : la preuve du backward

Sur un petit réseau et un paramètre `θ_i` :

```text
g_num = [L(θ_i+ε)-L(θ_i-ε)]/(2ε)
```

Comparer à la dérivée analytique. Avec calculs double précision, `ε=1e-5` est un premier essai raisonnable. Ce n'est pas un réglage universel : trop petit amplifie l'arrondi ; trop grand donne une approximation grossière.

Utiliser erreur absolue et relative, par exemple `|g-g_num| / max(1e-8, |g|+|g_num|)`. Près de zéro, une petite erreur absolue peut donner une grande erreur relative : regarder les deux. Désactiver dropout et aléatoire, utiliser la même loss, éviter les cassures de ReLU et restaurer le paramètre après les perturbations.

Le compagnon fait ce contrôle pour tous les paramètres de son petit MLP. À grande échelle, vérifier un échantillon de paramètres est plus abordable.

### 7.7 D'où vient exactement `P-Y` ?

Pour un exemple, écrivons `S = somme_k exp(z_k)`. La probabilité de la vraie classe est `p_y = exp(z_y)/S`. Donc :

```text
L = -log(p_y) = -z_y + log(S)
∂L/∂z_j = -1 si j=y, sinon 0, plus exp(z_j)/S
         = p_j - Y_j
```

Pour la moyenne d'un batch, chaque contribution est divisée par `B`. La version stable avec maximum soustrait représente la même fonction ; elle ne change pas la dérivée mathématique.

Pourquoi `dW=XᵀG` ? Le score `Z[n,j]` contient `X[n,i]×W[i,j]`. Sa dérivée par rapport à `W[i,j]` est donc `X[n,i]`. Ce même poids est utilisé pour tous les exemples, alors on additionne `X[n,i]×G[n,j]` sur `n`. C'est précisément l'entrée `[i,j]` de `XᵀG`.

Pourquoi additionner les gradients de biais ? Le même biais `b[j]` est ajouté à chaque exemple et sa dérivée vaut 1. Ses contributions sont donc la somme de `G[n,j]`.

### 7.8 Une mise à jour multiclasses calculable à la main

Prenons une seule couche dense avec trois sorties et aucun neurone caché : `x=[1,2]`, tous les poids et biais à zéro, label `Check`, indice 1. Cette initialisation est acceptable pour cet exercice de classifieur linéaire ; elle n'est pas un conseil pour les couches cachées d'un MLP.

```text
logits = [0,0,0]
P = [1/3,1/3,1/3]
G = [1/3,-2/3,1/3]
dW = [[1/3,-2/3,1/3], [2/3,-4/3,2/3]]
db = [1/3,-2/3,1/3]
```

Après une mise à jour SGD de pas `0.1` :

```text
W = [[-1/30,1/15,-1/30], [-1/15,2/15,-1/15]]
b = [-1/30,1/15,-1/30]
nouveaux logits sur x = [-0.2,0.4,-0.2]
P ≈ [0.2616,0.4767,0.2616]
loss ≈ 0.7408, contre log(3) ≈ 1.0986 avant la mise à jour
```

La probabilité de la bonne classe augmente. Dans un vrai réseau, cette modification doit aussi servir d'autres exemples ; réduire l'erreur d'un seul exemple ne garantit pas de réduire celle de tous les autres.

## 8. Mettre à jour les paramètres

### 8.1 SGD : la référence à comprendre d'abord

Pour chaque paramètre : `θ <- θ - ηg`, où `η` est le learning rate et `g` son gradient. Avec un mini-batch, on parle couramment de SGD même si le gradient est calculé sur plusieurs exemples.

| Type de batch | Avantage | Limite |
|---|---|---|
| Un exemple | Mises à jour fréquentes, peu de mémoire | Gradient bruyant ; peu efficace en calcul matriciel |
| Tout le dataset | Gradient exact de la perte d'entraînement actuelle | Mise à jour coûteuse, stockage ou accumulation nécessaire |
| Mini-batch | Compromis vitesse, mémoire et bruit | Taille à ajuster ; inclure le dernier batch incomplet |

Mélanger l'ordre des exemples à chaque epoch. Sinon, un fichier regroupé par classe peut conduire le modèle à se spécialiser successivement sur chaque classe. Mélanger les exemples n'autorise pas à mélanger validation et entraînement.

Le learning rate n'est pas la précision demandée au réseau. C'est la taille du déplacement dans l'espace des paramètres. Trop grand : oscillations, divergence ou gradients non finis. Trop petit : progression lente ou stagnation pratique.

### 8.2 Momentum

Convention possible : `v <- μv + g`, puis `θ <- θ - ηv`. Le coefficient `μ` conserve une partie des gradients précédents ; l'optimiseur accumule une direction.

Avantage : peut réduire les zigzags et accélérer les directions persistantes. Défaut : état supplémentaire par paramètre et risque de dépasser une bonne région avec un pas excessif. Il existe d'autres conventions mettant `η` ou `(1-μ)` dans `v` : choisir une convention cohérente et la documenter.

### 8.3 RMSprop et Adam

RMSprop suit une moyenne mobile des gradients au carré et adapte le pas par coordonnée. Adam ajoute une moyenne mobile du gradient, avec correction des biais initiaux :

```text
m <- β1 m + (1-β1)g
v <- β2 v + (1-β2)g²
m_hat = m/(1-β1^t)
v_hat = v/(1-β2^t)
θ <- θ - η m_hat/(sqrt(v_hat)+ε)
```

`g²`, racine et division sont coordonnée par coordonnée. `t` compte les mises à jour, pas les epochs. Les états commencent à zéro et persistent d'un batch au suivant. Les valeurs `β1=0.9`, `β2=0.999`, `ε=1e-8`, `η=0.001` sont des essais initiaux usuels, pas une garantie pour ton dataset. Voir [l'article original d'Adam](https://arxiv.org/abs/1412.6980).

Adam peut accélérer les premières expériences, mais coûte deux tableaux d'état supplémentaires et ne garantit pas une meilleure généralisation que SGD. Commencer par SGD pour valider les gradients, puis comparer Adam à budget égal.

### 8.4 Initialiser les poids

Des poids tous égaux maintiennent la symétrie entre neurones cachés : ils apprennent les mêmes choses. Les biais peuvent être à zéro ; les poids doivent généralement être aléatoires.

| Initialisation | Écart-type normal proposé | Usage et limite |
|---|---|---|
| Petite normale arbitraire | Exemple `0.01` | Simple, mais ignore la largeur des couches |
| Xavier / Glorot normal | `sqrt(2/(fan_in+fan_out))` | Bon point de départ pour tanh ; pas spécifiquement adapté à ReLU |
| He normal | `sqrt(2/fan_in)` | Adapté aux rectifications ReLU ; n'évite pas à lui seul les neurones inactifs |
| LeCun normal | `sqrt(1/fan_in)` | Autre adaptation à la taille d'entrée ; dépend de l'activation retenue |

`fan_in` est le nombre d'entrées, `fan_out` le nombre de sorties. Une normale de moyenne 0 et d'écart-type `σ` s'obtient avec `random.gauss(0, σ)`. Pour une uniforme `[-a,a]`, la variance vaut `a²/3` : ne pas réutiliser directement l'écart-type normal comme borne uniforme.

Les travaux de [Glorot et Bengio](https://proceedings.mlr.press/v9/glorot10a.html) et de [He et ses coauteurs](https://arxiv.org/abs/1502.01852) motivent ces initialisations par la propagation des signaux. Garde une seed pour reproduire un essai, puis varie les seeds pour évaluer sa stabilité.

## 9. Construire le premier moteur

### 9.1 Commencer petit

L'ordre de réalisation recommandé est :

1. Un neurone et la mise à jour numérique du chapitre 7.
2. Une couche dense avec plusieurs entrées/sorties.
3. Activation et dérivée.
4. Softmax stable et entropie croisée calculée sur les logits.
5. Backward et vérification numérique.
6. Boucle SGD qui apprend XOR.
7. Sauvegarde/relecture sans changement de prédiction.
8. Seulement ensuite : données d'échecs et optimisations.

XOR vaut 1 si les deux bits sont différents :

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Un classifieur linéaire ne peut pas séparer ces points par une droite. Une couche cachée non linéaire permet de les apprendre. Ce petit exercice vérifie un mécanisme ; il ne teste pas la généralisation, puisqu'on entraîne ici sur les quatre entrées possibles.

### 9.2 Contrats de classes

Une couche peut exposer :

```text
forward(X, training) -> A
backward(dA) -> dX
parameters() -> poids, biais et gradients associés
```

Une loss peut exposer une opération fusionnée `loss_and_grad(logits, labels)` retournant le scalaire moyen et le gradient des logits. Un optimiseur reçoit les paramètres et les gradients ; il n'interprète pas les labels.

Un conteneur `Sequential` appelle les couches dans l'ordre au forward et dans l'ordre inverse au backward. La bibliothèque ne doit pas connaître FEN : cela appartient au composant échiquier. Cette séparation permet de réutiliser le même moteur sur XOR ou sur un autre dataset.

### 9.3 Boucle d'entraînement

```text
initialiser le réseau et l'optimiseur
pour chaque epoch :
    mélanger les indices d'entraînement
    pour chaque mini-batch :
        réinitialiser les accumulateurs de gradients si nécessaire
        logits = réseau.forward(batch_X, training=True)
        loss, dlogits = loss_and_grad(logits, batch_y)
        réseau.backward(dlogits)
        optimiseur.step(paramètres)
    évaluer train et validation en mode inférence
    enregistrer métriques et éventuellement meilleur modèle
restaurer le meilleur modèle choisi sur validation
```

Avec dropout, distinguer la loss obtenue pendant les mises à jour de la loss train réévaluée sans dropout ; préciser laquelle figure dans le graphe. La loss d'une epoch doit être pondérée par le nombre d'exemples de chaque batch, sinon le petit dernier batch compte trop.

### 9.4 Le compagnon exécutable

Lire [xor_sans_lib.py](exemples/xor_sans_lib.py) en parallèle des chapitres 3 à 9, puis lancer :

```sh
python3 doc/exemples/xor_sans_lib.py
```

Il utilise une architecture `2 -> 8 -> 2`, tanh, softmax et SGD sur les quatre exemples à chaque étape. Les gradients de la perte moyenne sont écrits avec des boucles. Il vérifie aussi les grandes valeurs de softmax, le gradient et la relecture du modèle. La sauvegarde de démonstration est temporaire pour ne pas ajouter de faux modèle d'échecs dans le dépôt.

Résultat attendu pour cette seed et cette configuration : quatre bonnes classes après entraînement, une perte nettement inférieure à la perte initiale et des probabilités conservées après sauvegarde. Lire les valeurs réellement imprimées ; aucune mesure d'échecs n'est déduite de cet exercice.

Contrôle réellement exécuté lors de la rédaction, avec Python 3 et la seed 7 :

```text
Gradients vérifiés : 42 paramètres
Erreur maximale absolue=6.623e-12, relative=2.543e-09
Loss initiale : 0.705144
Epoch 5000 : loss=0.000852
XOR : 4/4 ; sauvegarde/relecture : sorties identiques.
```

Les derniers chiffres peuvent varier légèrement selon l'environnement numérique. Ce relevé confirme l'exercice fourni, pas l'existence d'un réseau d'échecs préentraîné.

## 10. Transformer une FEN en entrée

### 10.1 Les six champs

Exemple :

```text
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

| Champ | Signification |
|---|---|
| Placement des pièces | Huit rangées séparées par `/`, de la rangée 8 vers 1 ; cases de a vers h |
| `w` ou `b` | Camp au trait, c'est-à-dire celui qui doit jouer |
| `KQkq` ou `-` | Droits de roque |
| Case ou `-` | Case cible de prise en passant |
| Compteur de demi-coups | Nombre depuis un coup de pion ou une capture |
| Numéro du coup | Compteur de coups complets |

Majuscules : pièces blanches. Minuscules : noires. `P/N/B/R/Q/K` signifient pion/cavalier/fou/tour/dame/roi. Un chiffre entre 1 et 8 indique ce nombre de cases vides. La [spécification PGN, section 16.1](https://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm#c16.1) définit FEN.

Chaque rangée doit représenter exactement huit cases. Valider séparément syntaxe et légalité : une FEN bien formée peut décrire une position illégale. Tous les fichiers historiques ou artificiels ne respectent pas les mêmes contraintes ; expliciter les positions acceptées et le contrat attendu.

### 10.2 Les encodages possibles

| Encodage | Avantage | Défaut et traitement |
|---|---|---|
| Un identifiant numérique par case | 64 valeurs seulement | Introduit une distance arbitraire entre types de pièces ; comparer à one-hot |
| 12 canaux binaires de pièces | Pas d'ordre artificiel ; identité et couleur distinctes | 768 valeurs, souvent nulles ; coût plus grand en dense |
| 13 catégories par case, vide inclus | Représentation explicite de toutes les cases | 832 valeurs ; canal vide redondant si les 12 autres sont fiables |
| Features de règles : attaques, mobilité... | Aide à représenter les interactions | Coût du moteur et risque de donner la réponse directement |
| Encodage relatif au joueur au trait | Peut faciliter le partage entre couleurs | Transformations et droits spéciaux à traiter sans erreur |

Proposition de base : 12 plans × 64 cases = 768 nombres binaires, plus joueur au trait, quatre droits de roque et un encodage de la prise en passant.

Pour chaque case, un seul des 12 indicateurs vaut 1 si elle est occupée ; tous valent 0 si elle est vide. Choisir un ordre constant des pièces et des cases. Exemple d'ordre : `PNBRQKpnbrqk`, cases `a8,b8,...,h1`.

**Attention aux 773 entrées mentionnées dans le README :** `768+1+4` omet la prise en passant. Cela peut suffire pour certaines données, mais ne représente pas toutes les informations qui influencent les coups légaux. La prise en passant peut être un coup de défense contre un échec et donc affecter le diagnostic de mat.

Deux représentations supplémentaires possibles :

- 65 catégories one-hot : aucune cible ou chacune des 64 cases ; total `768+1+4+65 = 838` ; simple, mais nombreuses entrées ;
- huit indicateurs de colonne plus un indicateur de présence ; total `782` ; rangée déduite du joueur au trait pour une FEN légale ; hypothèse à documenter et valider.

Ce sont des options, pas un ordre d'optimalité. Ajouter l'information suffit à rendre certains états distinguables ; cela ne garantit pas que le réseau apprendra à l'utiliser.

Les compteurs ne déterminent pas directement `Check / Checkmate` selon cette définition locale. Les parser demeure nécessaire. Leur omission du vecteur peut être justifiée pour ces trois classes ; réévaluer ce choix si on traite les nulles et leur historique.

### 10.3 Pourquoi les features doivent suffire

Si deux positions ont le même vecteur d'entrée mais des labels différents, aucun réseau déterministe ne peut les distinguer. Augmenter sa largeur ne répare pas une information absente. Avant de chercher un modèle plus grand, rechercher ces collisions d'encodage.

Ne pas donner `is_checkmate` en entrée : ce serait fournir la cible. Un réseau recevant des features intermédiaires d'un moteur reste une conception différente d'un réseau apprenant depuis les pièces ; décrire honnêtement le travail déjà fait par l'encodage.

### 10.4 Normalisation

Les indicateurs binaires ont déjà une échelle maîtrisée. Si on ajoute des valeurs numériques, calculer moyenne et écart-type sur le train seulement : `x'=(x-μ)/σ`. Gérer `σ=0`, sauvegarder ces statistiques et appliquer exactement la même transformation en prédiction.

Ne pas normaliser des identifiants de pièces en espérant résoudre leur ordre arbitraire : changer l'échelle ne change pas leur sens.

## 11. Construire des données fiables

### 11.1 Définir la vérité terrain

Pour une position légale :

```text
si le roi du joueur au trait est attaqué ET aucun coup légal : Checkmate
sinon si ce roi est attaqué : Check
sinon : Nothing
```

Le pat est « aucun coup légal sans être en échec ». Dans le classifieur à trois classes, il est inclus dans `Nothing` si ce contrat est retenu ; créer une classe distincte est une extension.

La [FIDE, articles 1.4, 3 et 5](https://handbook.fide.com/chapter/E012023) définit échec, coups légaux, mat et pat. Une case attaquée ne se confond pas avec la possibilité pour une pièce de jouer légalement : une pièce clouée peut encore attaquer une case au sens des règles. Traiter aussi les directions des pions et l'adjacence des rois.

### 11.2 Générer ou récupérer les exemples

| Source | Intérêt | Biais à surveiller |
|---|---|---|
| Parties réellement jouées | Positions légales et structures plausibles | Beaucoup de `Nothing`, peu de mats effectivement présents |
| Positions composées / puzzles | Cas tactiques et mats variés | Distribution différente des parties et des positions simples |
| Génération de parties légales | Reproductible, contrôle du processus | Biais de la politique de jeu utilisée |
| Placement aléatoire de pièces | Contrôle du nombre de pièces | Beaucoup de positions impossibles ; raccourcis artificiels |
| Petits cas construits à la main | Vérification du parser et des règles | Trop peu variés pour constituer un entraînement sérieux |

Une partie peut s'arrêter par abandon avant le mat : ne pas transformer automatiquement le résultat de la partie en label de la dernière position. Le résultat global n'est pas l'état local de l'échiquier.

Si tu écris le moteur de labels, tester d'abord coups des pièces, obstacles, roi en échec, coups qui laissent le roi attaqué, roque et prise en passant. Un coup pseudo-légal respecte le déplacement d'une pièce ; un coup légal respecte aussi la sécurité du roi.

Vérifier un corpus de labels avec une référence indépendante si disponible et autorisée. Un réseau qui reproduit des erreurs de labels peut avoir une bonne accuracy sans résoudre la tâche réelle.

### 11.3 Train, validation et test

- **Train** : calcule les gradients.
- **Validation** : choisit hyperparamètres, epoch d'arrêt et meilleur modèle.
- **Test** : estime le résultat du choix final ; ne sert pas à décider les réglages.

Une répartition illustrative est 70/15/15, à ajuster à la taille des données et au nombre d'exemples de chaque classe. Il faut surtout suffisamment de cas rares dans chaque ensemble.

Dédupliquer avant la séparation. Regrouper les positions d'une même partie ou les variantes très proches d'un même problème dans un seul ensemble, autant que possible. Sinon, des quasi-doublons en train et test rendent l'évaluation trop optimiste. Faire la séparation par groupes, puis chercher une répartition de classes raisonnable ; un simple tirage stratifié ligne par ligne ne suffit pas.

Une même position peut avoir plusieurs FEN avec des compteurs différents. La clé de déduplication doit refléter les champs pertinents pour la tâche. Ne pas supprimer joueur au trait ou droits spéciaux qui peuvent modifier le label.

### 11.4 Déséquilibre des classes

Sur un dataset ayant 90 % de `Nothing`, répondre toujours `Nothing` donne 90 % d'accuracy. C'est une référence obligatoire à comparer, pas une solution satisfaisante.

Possibilités : collecter plus de cas rares, échantillonner les classes pendant l'entraînement, pondérer la perte, ou utiliser un mélange contrôlé. Répéter exactement les mêmes rares positions peut accroître leur mémorisation. Enrichir leur variété est préférable à seulement multiplier leurs copies.

Conserver un test représentant l'usage attendu et éventuellement un second test équilibré pour voir chaque classe. Un score sur le test équilibré ne représente pas automatiquement la fréquence réelle des erreurs.

### 11.5 Augmentation de données

Une transformation doit préserver ou adapter correctement le label et tous les champs FEN. Exemple à étudier : réflexion des rangées et échange des couleurs pour conserver le sens des pions, avec joueur au trait, droits de roque et prise en passant transformés.

Une rotation ou un miroir quelconque n'est pas automatiquement une symétrie des échecs, notamment à cause des pions et du roque. Vérifier chaque transformation contre le moteur de règles. Les augmentations d'un exemple restent dans le même ensemble que l'original.

## 12. Éviter le surapprentissage

### 12.1 Reconnaître le problème

Surapprentissage : la loss train baisse mais la performance validation cesse de progresser ou se dégrade. Sous-apprentissage : le modèle échoue aussi sur le train. Une mauvaise généralisation peut également venir d'un changement de distribution ou de labels incohérents : ce n'est pas toujours un problème de taille du réseau.

### 12.2 Les outils

| Méthode | Mécanisme | Avantage | Limite et vérification |
|---|---|---|---|
| Early stopping | Arrêter après une absence de progrès validation | Simple ; limite la poursuite de la mémorisation | Validation bruitée ; garder et restaurer le meilleur modèle |
| Réduire le modèle | Moins de couches ou neurones | Moins de paramètres et calcul | Peut sous-apprendre ; comparer les deux courbes |
| Plus de données variées | Plus de situations représentées | Traite la couverture du problème | Qualité et coût des labels |
| L2 | Ajouter `λ/2 × somme(W²)` à la perte | Décourage les poids très grands | `λ` à régler ; ajouter `λW` au gradient correspondant |
| Dropout | Masquer aléatoirement des activations | Réduit certaines coadaptations | Apprentissage plus bruité ; modes train/inférence distincts |
| Augmentation valide | Ajouter des variantes compatibles avec les règles | Peut développer des invariances utiles | Transformation erronée = labels corrompus |

Early stopping : par exemple surveiller la loss validation, patience 10 epochs, amélioration minimale explicite. Sauvegarder une **copie** des meilleurs poids, pas une référence au tableau qui continue d'être modifié. La patience proposée est un essai, pas une exigence du sujet.

### 12.3 Dropout et backward

Pour une probabilité de suppression `p`, on garde chaque activation avec probabilité `1-p`. Version « inverted dropout » :

```text
M_i = 1 si gardé, 0 sinon
A_train = A ⊙ M / (1-p)
dA = dA_train ⊙ M / (1-p)
A_inférence = A
```

Le même masque est utilisé au forward et au backward de cette étape. Le facteur maintient l'espérance de l'activation, pas la prédiction complète exacte du réseau. Vérifier `0 ≤ p < 1`. Commencer sans dropout pour vérifier le moteur, puis essayer par exemple `p=0.1` ou `0.2` sur les couches cachées. L'[article original sur le dropout](https://jmlr.org/papers/v15/srivastava14a.html) présente cette méthode de régularisation.

**MC dropout** garde le dropout actif pour plusieurs inférences et agrège les prédictions. Il sert à explorer la variabilité et une forme d'incertitude, avec un coût de plusieurs passages. Ce n'est ni une obligation pour réussir le projet, ni une garantie de détection de toutes les erreurs. L'inférence standard doit rester déterministe si ce mode n'est pas explicitement retenu.

### 12.4 Ne pas confondre L2 et AdamW

L2 dans la loss donne un gradient `g+λW`. Avec SGD simple, cela correspond à une forme de décroissance des poids. Avec des pas adaptatifs, cette pénalité n'est pas identique à une décroissance découplée de type AdamW. Ne pas employer les deux noms pour le même code sans vérifier les équations.

## 13. Mesurer les résultats

### 13.1 Plusieurs métriques nécessaires

```text
Accuracy = nombre de bonnes prédictions / nombre d'exemples
Précision d'une classe = TP / (TP+FP)
Rappel d'une classe = TP / (TP+FN)
F1 = 2 × précision × rappel / (précision+rappel)
```

`TP` : exemples correctement reconnus dans cette classe ; `FP` : autres classes prédites comme celle-ci ; `FN` : exemples de cette classe manqués. Préciser quoi faire si un dénominateur est nul ; par exemple rapporter 0 avec la taille de classe, plutôt que cacher la classe.

**Exemple inventé pour comprendre :** 20 mats réels, 15 mats reconnus, 5 mats manqués, 3 autres positions annoncées mat. Précision `15/18≈83.3 %`, rappel `15/20=75 %`, F1 `≈78.9 %`. Une bonne précision n'implique pas que tous les mats sont détectés.

Macro-F1 = moyenne des F1 des classes, chaque classe comptant autant. F1 pondéré = moyenne pondérée par leur fréquence. Le second peut masquer les difficultés d'une classe rare.

Une matrice de confusion montre, ligne par ligne, la vraie classe et, colonne par colonne, la prédiction. Écrire explicitement l'ordre des classes et le sens des axes. Pour les sorties avec couleur, comparer aussi l'état sans couleur afin de distinguer erreur d'état et erreur de camp.

### 13.2 Graphes et preuves

Conserver un CSV par expérience avec epoch, loss train, loss validation, accuracy et macro-F1. Tracer les courbes train/validation sur les mêmes axes, la matrice de confusion et, pour les comparaisons, résultats et temps.

Ces graphiques peuvent être produits avec un outil de tracé ou un tableur si autorisés ; ils ne réalisent pas l'apprentissage. Sans bibliothèque externe, un export CSV et un tracé SVG écrit par ton script restent possibles.

Rapporter taille et origine des ensembles, distribution des classes, protocole de séparation, machine, seeds et nombre d'essais. Évaluer l'inférence sans inclure le temps de chargement si c'est ce qui est annoncé ; mesurer ce chargement séparément.

### 13.3 Quels résultats viser ?

Le sujet ne donne pas dans les pages décrivant la tâche un seuil universel de précision à atteindre. Ne pas inventer « 95 % obligatoire ». Les objectifs utiles sont :

1. Gradient contrôlé et aucun nombre non fini.
2. XOR appris correctement pour la configuration fournie.
3. Petit sous-ensemble d'échecs volontairement surajusté pour vérifier la capacité du moteur.
4. Sur un ensemble nouveau, dépasser la baseline de classe majoritaire et comparer au classifieur linéaire, notamment en macro-F1 et rappel des mats.
5. Résultats stables à travers plusieurs seeds et documentés avec leurs limites.
6. Sauvegarde/relecture fidèle ; contrat CLI et ordre des prédictions respectés.

Le point 3 est un diagnostic, pas le modèle final : réussir à mémoriser quelques dizaines d'exemples ne prouve aucune généralisation. Les scores du README ne remplacent pas tes mesures reproductibles sur les données finales.

### 13.4 Confiance et calibration

Si un modèle annonce environ 90 % de probabilité pour beaucoup de prédictions, vérifier si environ 90 % sont correctes sur un ensemble indépendant. Softmax impose une somme de 1 ; elle ne rend pas automatiquement les probabilités calibrées.

Rapporter également les erreurs très confiantes. Une classe `Nothing` n'est pas une absence de réponse ou une faible confiance : c'est une vraie catégorie du problème. Un éventuel mécanisme d'abstention doit être défini séparément et compatible avec le contrat CLI.

## 14. Optimiser sans tâtonner

### 14.1 Référence expérimentale

Proposition de départ à tester : encodage complet retenu, une couche cachée de 64 neurones ReLU, softmax à trois classes, initialisation He sur la couche cachée, SGD mini-batch ou Adam après validation du moteur. Batches de 32 ou 64 ; learning rate exploré sur une échelle logarithmique.

Ce choix vise la simplicité de débogage. Il ne garantit pas de capturer toutes les interactions d'échecs. Augmenter profondeur ou largeur seulement si les mesures justifient ce besoin.

### 14.2 Plan de comparaison

| Expérience | Variable étudiée | Question |
|---|---|---|
| Linéaire contre MLP | Présence d'une couche non linéaire | Les interactions apprises améliorent-elles le test de validation ? |
| 64 contre 128 neurones | Largeur | Gain réel ou mémorisation supplémentaire ? |
| Une contre deux couches cachées | Profondeur | Gain suffisant pour le coût supplémentaire ? |
| SGD contre Adam | Optimiseur | Meilleure convergence à temps ou mises à jour comparables ? |
| Avec/sans dropout | Régularisation | Le gain validation compense-t-il la difficulté d'apprentissage ? |
| Avec/sans prise en passant | Encodage | Quels cas échouent et quelles collisions disparaissent ? |

Conserver le même découpage de données et le même protocole. Pour une comparaison simple, changer une variable à la fois ; ensuite étudier les interactions, car le meilleur learning rate dépend de l'optimiseur et du batch. Comparer SGD mal réglé à Adam bien réglé ne mesure pas seulement l'algorithme.

Répéter plusieurs seeds ; publier moyenne, dispersion et résultats individuels. Si le budget ne permet qu'un essai, l'annoncer au lieu d'attribuer chaque différence à l'architecture.

### 14.3 Recherche des hyperparamètres

| Méthode | Avantage | Limite |
|---|---|---|
| Réglage manuel raisonné | Facile pour peu d'essais, utile au diagnostic | Difficile à reproduire sans journal |
| Grid search | Tableau systématique | Coût multiplicatif : 4 pas × 3 tailles × 3 dropouts = 36 essais |
| Random search | Explore plusieurs dimensions avec un budget fixé | Peut manquer une région utile ; seeds et distributions à conserver |
| Optimisation bayésienne | Utilise les observations pour proposer des essais | Moteur supplémentaire et coût d'implémentation |
| Métaheuristiques / évolution | Explore des choix hétérogènes | Nombreuses évaluations ; pas nécessairement plus efficace |

Exemple d'essais : learning rate dans `{0.0003, 0.001, 0.003}` pour Adam ; largeur dans `{64,128}` ; dropout dans `{0,0.1,0.2}`. Il s'agit d'une plage de départ à réviser selon les courbes. Utiliser la validation, conserver le test fermé jusqu'au choix final.

### 14.4 Vitesse et stabilité

Priorités : mesurer les goulots d'étranglement, éviter de parser les mêmes FEN à chaque epoch, organiser les batches et, si autorisé, vectoriser les matrices. Multithreading ou GPU arrivent après la correction du moteur, pas avant.

Un scheduler diminue le learning rate selon une règle documentée ; par exemple après un plateau validation. Le gradient clipping limite la norme du gradient si elle dépasse un seuil, mais ne répare pas une dérivée fausse. Adam ne supprime pas les points selles, minima locaux ou problèmes de données. « Il existe un minimum local » n'est pas un diagnostic suffisant d'une loss qui stagne.

La [note Stanford sur l'entraînement](https://cs231n.github.io/neural-networks-3/) donne des pistes de contrôle, notamment la vérification des gradients et le suivi de l'apprentissage. Utilise ces contrôles pour décider de la prochaine expérience, pas pour multiplier des réglages au hasard.

## 15. Livrer une bibliothèque et un analyseur

### 15.1 Organisation possible

```text
src/
    neural_network/  dense, activations, pertes, optimiseurs, modèle
    chess_engine/    parseur FEN, encodage, règles
    data/            nettoyage, génération et annotations communes
    training/        entraînement et évaluation
    experiments/     benchmarks et recherche des hyperparamètres
    cli/             arguments, predict, train, gestion des erreurs
    my_torch.py      point d'entrée clean/train/eval
dataset/             corpus sources, nettoyés et exemples
tests/               contrôles numériques, FEN et contrat CLI
tools/               outils de maintenance
doc/                 cours, protocole, décisions et graphes
```

Cette organisation correspond au projet. Chaque module doit servir une responsabilité claire.

### 15.2 Que sauvegarder ?

Pour l'inférence : version du format, architecture et tailles, activations et leurs réglages, poids et biais, ordre des labels, version et paramètres de l'encodage, éventuelle normalisation. Le modèle doit être compréhensible sans deviner quel script l'a créé.

Pour reprendre exactement un entraînement : ajouter états de l'optimiseur, compteur des mises à jour, état du générateur aléatoire, scheduler, état d'early stopping et position dans les données si nécessaire. Les poids seuls permettent de redémarrer un entraînement, mais pas de reproduire sa continuation exacte avec Adam.

| Format | Avantage | Limite |
|---|---|---|
| JSON | Lisible, inspectable, sans dépendance en Python | Fichiers volumineux, chargement plus lent |
| Binaire documenté | Compact et rapide | Versions, types et dimensions à vérifier rigoureusement |
| Format sérialisant du code Python | Facile pour des objets arbitraires | Chargement de fichiers non fiables risqué ; forte dépendance à l'implémentation |

Un JSON minimal est un bon départ. Vérifier dimensions, nombre de classes et nombres finis au chargement. L'extension `.nn` ne définit pas à elle seule le contenu.

### 15.3 Flux de prédiction

```text
charger modèle et métadonnées
pour chaque ligne d'entrée :
    extraire les 6 champs FEN et l'éventuel label
    valider et encoder selon la version du modèle
    forward(training=False)
    argmax puis conversion en texte attendu
    écrire une prédiction
```

Une FEN contient déjà des espaces. Ne pas traiter son premier mot comme toute la position ou son dernier champ comme le label. Les sorties colorées peuvent aussi contenir un espace : définir une grammaire explicite. En `--predict`, un label fourni sert au plus à mesurer le résultat ; il ne doit pas influencer la prédiction.

Les logs de chargement et d'entraînement ne doivent pas polluer le flux des prédictions. Mettre les diagnostics sur stderr. Vérifier comportement sur fichier absent, modèle incompatible, argument manquant, ligne invalide et mode `--save` hors entraînement.

Le sujet donne ce contrat :

```sh
./my_torch_analyzer --predict LOADFILE CHESSFILE
./my_torch_analyzer --train --save SAVEFILE LOADFILE CHESSFILE
```

Sans `--save` en entraînement, le sujet indique une sauvegarde dans `LOADFILE`. Le modèle préentraîné livré doit avoir un nom commençant par `my_torch_network`. Une sauvegarde écrite dans un fichier temporaire puis remplacée réduit les risques de fichier incomplet.

### 15.4 Reproductibilité

Conserver commande exacte, configuration, version du code, seed, dataset et découpage, encodage, origine des labels et environnement. Un hash des fichiers permet d'identifier les données utilisées. Mettre ces informations dans un journal d'expérience.

Exemple de ligne de journal à **remplir après une expérience réelle** :

| ID | Encodage | Architecture | Optimiseur/pas | Seed | Validation macro-F1 | Temps | Décision |
|---|---|---|---|---|---|---|---|
| E01 | à préciser | à préciser | à préciser | à préciser | à mesurer | à mesurer | à justifier |

## 16. Diagnostiquer les problèmes

| Symptôme | Causes plausibles | Prochain contrôle utile |
|---|---|---|
| Loss `NaN` / infinie | Exponentielle instable, pas trop grand, entrée non finie | Localiser la première opération non finie ; vérifier softmax et logits |
| Loss constante | Poids non modifiés, cache erroné, gradients nuls, pas trop faible | Vérifier changements de paramètres et gradient numérique sur petit réseau |
| XOR échoue | Absence de non-linéarité, erreur de backward, configuration inadéquate | Contrôler les dérivées avant d'agrandir le réseau |
| Train et validation médiocres | Sous-capacité, mauvaise optimisation, information absente, labels erronés | Essayer de mémoriser un petit sous-ensemble et rechercher collisions d'encodage |
| Train excellent, validation faible | Surapprentissage, fuite inverse de distribution, dataset train peu varié | Courbes, séparation par groupes, diversité, régularisation |
| Accuracy bonne, mats manqués | Déséquilibre des classes | Matrice de confusion, rappel mat et baseline majoritaire |
| Scores différents après chargement | Ordre des pièces/classes différent, poids incomplets, dropout actif | Comparer entrées, logits et métadonnées avant/après |
| Échec surtout en fin de fichier | Dernier batch ignoré, sortie réordonnée, parsing | Contrôle du nombre et de l'ordre des lignes |
| Trop de neurones ReLU à zéro | Initialisation ou pas excessif, distribution d'entrée | Mesurer leur activité et essayer un pas réduit / LeakyReLU |
| Bon test, mauvaises positions nouvelles | Test peu représentatif ou contaminé | Test provenant d'une autre source et analyse par type de position |

Ne pas empiler simultanément dropout, Adam, couches supplémentaires et nouveaux encodages pour corriger un bug. Isoler un contrôle dont le résultat permet de choisir la suite.

## 17. Parcours de réalisation et critères de réussite

| Palier | Travail à réaliser | Preuve attendue pour passer au suivant |
|---|---|---|
| A : opérations | Produit matriciel, dense, activations | Petits exemples manuels retrouvés ; formes cohérentes |
| B : entraînement | Loss, backward, SGD | Gradients numériques proches des gradients analytiques |
| C : XOR | Petit MLP non linéaire | 4/4 avec la configuration contrôlée ; loss qui baisse |
| D : bibliothèque | Modèle générique et sérialisation | Même entrée donne les mêmes sorties avant/après chargement |
| E : échiquier | Parser et encodage | Cases, trait et droits spéciaux vérifiés avec positions connues |
| F : datasets | Labels, déduplication, séparation | Distribution publiée ; provenance et absence de fuite contrôlées |
| G : premier modèle | Apprentissage et validation | Baselines comparées ; métriques par classe et courbes |
| H : optimisation | Quelques expériences ciblées | Une décision fondée sur gain, coût et dispersion |
| I : livraison | CLI, modèle préentraîné, README, scripts | Prédictions ordonnées ; erreurs 84 ; procédure reproductible |

Le palier C est entièrement illustré dans le compagnon ; les autres ne sont pas présentés comme déjà réalisés dans le dépôt. Ton niveau de maîtrise se juge à ce que tu peux expliquer et reproduire, pas au nombre de fonctionnalités mentionnées dans le README.

## 18. Justifier les choix à la soutenance

### 18.1 La structure d'une justification solide

Pour chaque décision, répondre à six questions :

1. Quel besoin précis cherchais-je à satisfaire ?
2. Quelles alternatives ai-je réellement considérées ?
3. Pourquoi l'option retenue répond-elle à ce besoin ?
4. Quels défauts ou hypothèses accepte-t-elle ?
5. Qu'ai-je fait pour réduire ces défauts ?
6. Quelle mesure confirme le choix, et dans quelles limites ?

Tu peux avoir une justification théorique avant les expériences. Après entraînement, dire « ce choix est meilleur sur mes essais » nécessite un protocole et des mesures. Ne pas transformer une intuition raisonnable en résultat prouvé.

### 18.2 Exemples à adapter à tes résultats

**Sortie :** « J'ai choisi une softmax à trois classes parce qu'une position reçoit une catégorie unique. Elle produit une distribution normalisée et l'argmax donne une réponse sans seuil par bit. Mon codage binaire initial permettait des combinaisons sans signification. La softmax ne garantit cependant pas une confiance fiable : je contrôle les erreurs très confiantes et les métriques par classe. »

**Entrée :** « J'ai choisi douze plans binaires parce qu'ils séparent type et couleur sans imposer une distance arbitraire entre pièces. Un identifiant par case était plus compact mais portait un ordre numérique artificiel. L'encodage retenu est plus volumineux ; je limite donc d'abord la largeur de la couche dense. J'ai ajouté les informations de trait et de droits spéciaux pour éviter de rendre certains états indiscernables. »

**Architecture :** « J'ai commencé par un MLP pour écrire et vérifier facilement son backward. Il peut apprendre des interactions non linéaires, contrairement à ma baseline linéaire. Il ne possède pas de géométrie native d'échiquier ; j'analyse donc ses erreurs selon les attaques à longue distance et compare plusieurs tailles. Un CNN serait une autre expérience, pas une amélioration que je peux affirmer sans test. »

**Activation :** « J'ai choisi ReLU pour son calcul simple et son gradient non saturé du côté positif. La sigmoid peut saturer dans les couches cachées. ReLU peut rendre des neurones inactifs ; j'ai choisi une initialisation adaptée, surveillé leur activité et testé un pas plus faible. Si LeakyReLU améliore mes mesures, je peux justifier ce changement avec le même protocole. »

**Optimiseur :** « J'ai validé SGD avant d'ajouter Adam afin de vérifier une mise à jour simple. Adam adapte le pas par paramètre et accumule les gradients, mais ajoute des états et ne garantit pas une meilleure généralisation. Je le retiens seulement si mes comparaisons à budget annoncé montrent un gain utile. »

**Régularisation :** « J'ai observé une baisse de loss train accompagnée d'une hausse de loss validation. J'ai choisi early stopping pour garder le modèle de meilleure validation, puis testé dropout. Le dropout ajoute du bruit et ralentit parfois l'ajustement ; je compare donc les courbes et l'utilise seulement si la validation en bénéficie. »

**Métriques :** « L'accuracy seule était insuffisante parce que `Nothing` dominait le dataset. J'ai ajouté macro-F1, rappel des mats et matrice de confusion. Le macro-F1 ne reflète pas les fréquences réelles d'usage ; je publie aussi ces fréquences et l'accuracy. »

Ces paragraphes ne décrivent pas des résultats déjà obtenus. Remplacer les formulations conditionnelles par tes mesures uniquement après les expériences.

### 18.3 Fiche de décision à recopier

```text
Décision :
Besoin / contrainte :
Alternative A et limite :
Alternative B et limite :
Option retenue et bénéfice attendu :
Défaut accepté / hypothèse :
Mesure de réduction de ce défaut :
Protocole de comparaison :
Résultats réellement obtenus, seeds et coût :
Limites de la conclusion :
Condition qui me ferait changer de choix :
```

Une phrase comme « Adam est mieux » est trop vague. « À découpage identique, après réglage du pas pour chaque optimiseur et sur trois seeds, Adam atteint telle macro-F1 en tant de secondes ; son gain compense ici sa mémoire supplémentaire » est défendable si tu possèdes les mesures.

## 19. Exercices et questions de contrôle

### Exercices progressifs

1. Retrouver les sorties des deux exemples matriciels du chapitre 3.
2. Refaire la mise à jour du neurone du chapitre 7 avec `η=0.05`. Réponse : `w=0.25`, `b=0.125`, sortie `0.625`, perte `0.0703125`.
3. Expliquer pourquoi deux couches affines sans activation restent affines. Indice : développer `(XW1+b1)W2+b2`.
4. Calculer `P-Y` pour `P=[0.8,0.1,0.1]` et label `Checkmate`. Réponse pour un exemple : `[0.8,0.1,-0.9]`.
5. Dans le compagnon, remplacer tanh par l'identité **et adapter la dérivée**. Expliquer pourquoi l'objectif XOR ne peut plus être satisfait parfaitement.
6. Modifier volontairement le signe d'un gradient ; constater l'échec de la vérification numérique, puis corriger.
7. Construire un test de parser avec une pièce en a8 et une en h1. Vérifier indices, canaux et couleur.
8. Chercher deux FEN dont une information omise change les coups légaux. Vérifier leurs labels avec un moteur fiable avant de conclure qu'elles ont des états différents.
9. Produire un jeu déséquilibré et mesurer le prédicteur toujours `Nothing`. Expliquer accuracy et rappel des autres classes.
10. Rédiger une fiche de décision softmax/code binaire, puis une fiche MLP/linéaire. Distinguer bénéfices théoriques et résultats mesurés.

### Questions de soutenance

- Qu'est-ce qui est appris et qu'est-ce qui est fixé à la main ?
- Pourquoi le réseau a-t-il besoin d'une non-linéarité ?
- Que représente chaque dimension de ton entrée ?
- Peux-tu dérouler le forward et le backward d'une couche avec leurs formes ?
- Pourquoi le gradient des logits vaut-il `(P-Y)/B` avec cette loss ?
- Comment sais-tu que ton backward est correct ?
- Pourquoi la classe majoritaire ne suffit-elle pas ?
- Comment sais-tu que tes données de test n'ont pas servi aux choix ?
- Que se passe-t-il pour un pat, une FEN illégale ou un label coloré ?
- Comment traites-tu roque, prise en passant et joueur au trait ?
- Qu'est-ce qui est sauvegardé pour prédire ? Et pour reprendre Adam ?
- Quel résultat te ferait préférer une architecture plus petite ?
- Quelle est l'erreur la plus fréquente et que suggère-t-elle ?
- Quels défauts as-tu réduits, et lesquels demeurent ?

Tu es prêt lorsque tu peux répondre avec un petit exemple, un morceau de code ou une mesure, plutôt qu'avec seulement le nom d'une technique.

## 20. Références et glossaire

### Lectures primaires pour approfondir

Les explications et exercices du cours sont construits pour ce projet. Les références suivantes permettent d'approfondir les méthodes et les règles, sans déléguer leur implémentation :

- [Sujet local MY_TORCH](../my_torch-project.pdf) : exigences, interfaces et livraison.
- [Stanford CS231n : neurones et architectures](https://cs231n.github.io/neural-networks-1/) : comparaison des activations.
- [Stanford CS231n : préparation des données et initialisation](https://cs231n.github.io/neural-networks-2/) : compléments pour construire un premier modèle.
- [Stanford CS231n : apprentissage et contrôle](https://cs231n.github.io/neural-networks-3/) : vérification et suivi des calculs.
- [Kingma et Ba : Adam](https://arxiv.org/abs/1412.6980) : description originale de l'optimiseur.
- [Glorot et Bengio : initialisation](https://proceedings.mlr.press/v9/glorot10a.html) : propagation des signaux et initialisation Xavier.
- [He et al. : rectifications et initialisation](https://arxiv.org/abs/1502.01852) : initialisation adaptée aux rectifications.
- [Srivastava et al. : dropout](https://jmlr.org/papers/v15/srivastava14a.html) : méthode de régularisation.
- [Spécification PGN/FEN, section 16.1](https://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm#c16.1) : représentation des positions.
- [Règles FIDE](https://handbook.fide.com/chapter/E012023) : définitions et légalité des coups.

### Glossaire à garder sous la main

| Mot | Traduction utile |
|---|---|
| Dense / fully connected | Chaque sortie reçoit toutes les entrées de la couche |
| Forward | Calcul des sorties dans l'ordre des couches |
| Backward / backpropagation | Calcul des gradients dans l'ordre inverse |
| Logit | Score réel avant softmax |
| One-hot | Un indicateur à 1, tous les autres à 0 |
| Learning rate | Pas d'une mise à jour |
| Batch size | Nombre d'exemples pour cette mise à jour |
| Overfitting | Surapprentissage, ajustement qui ne se transfère pas bien |
| Underfitting | Sous-apprentissage, ajustement insuffisant |
| Baseline | Solution de référence à dépasser ou comparer |
| Ablation | Retirer une composante pour mesurer son apport |
| Data leakage | Information de validation/test qui influence indûment l'apprentissage ou la sélection |
| Checkpoint | État sauvegardé du modèle, éventuellement de l'entraînement |
| Calibration | Accord entre probabilités annoncées et fréquences de réussite |

La bonne justification ne consiste pas à réciter tous ces mots. Elle relie un défaut observé à une correction et à une preuve de son effet.
