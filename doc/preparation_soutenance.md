# Préparer la soutenance de my_torch : maths et choix techniques

Fiche vérifiée le 10 novembre 2026 contre le code, les métadonnées des modèles
locaux et le sujet. Les scores ci-dessous sont ceux enregistrés lors des
entraînements ; ces entraînements complets n'ont pas été relancés pour cette fiche.

## 1. Ton fil conducteur à l'oral

> « Nous avons développé une bibliothèque de réseaux de neurones en Python et
> NumPy, sans framework de deep learning. Nous l'appliquons à une classification
> supervisée de positions d'échecs : Nothing, Check ou Checkmate, avec une variante
> qui distingue la couleur qui donne l'échec. Une FEN est encodée en nombres, le
> réseau calcule des probabilités, puis l'entraînement ajuste ses poids grâce au
> gradient de la perte. Nous avons comparé plusieurs représentations et réglages
> en validation, en surveillant le surapprentissage. »

Le projet classe une **position actuelle**. Il ne prédit pas, à lui seul, le
vainqueur futur d'une partie et ne choisit pas directement le meilleur coup.
La recherche de coups appartient au bonus de jeu.

Pour répondre à un « pourquoi », suis toujours ce chemin : **problème → choix →
effet attendu → preuve disponible → limite**. Ne transforme pas une raison
théorique en résultat expérimental que tu n'as pas mesuré.

## 2. Comprendre le calcul d'une prédiction

### Du neurone à la couche Dense

Un neurone calcule une somme pondérée, ajoute un biais puis applique une activation :

\[
z = \sum_i x_i w_i + b, \qquad h = f(z).
\]

- Les entrées décrivent la position.
- Les poids déterminent l'influence des entrées ; ils sont appris.
- Le biais permet de décaler le seuil de réponse.
- L'activation apporte une non-linéarité.

Exemple pédagogique : avec \(x=(1,2)\), \(w=(0,2;-0,1)\) et \(b=0,1\),
on obtient \(z=1\times0,2+2\times(-0,1)+0,1=0,1\).
Après ReLU, la sortie vaut toujours \(0,1\).

Pour un lot de B exemples, toutes les sommes se calculent ensemble :

\[
Z=XW+b.
\]

Avec la première couche basic et B = 64 :

| Objet | Dimensions | Sens |
| --- | --- | --- |
| X | 64 × 773 | Un échiquier encodé par ligne |
| W | 773 × 512 | Poids entre entrées et neurones |
| b | 1 × 512 | Un biais par neurone, diffusé sur les lignes |
| Z | 64 × 512 | Sorties de la couche avant activation |

Fichier à montrer : [layers.py](../src/neural_network/layers.py), `Dense.forward`.

### Pourquoi une activation ?

\[
\operatorname{ReLU}(z)=\max(0,z).
\]

Sans activation non linéaire, empiler des couches affines revient à une seule
couche affine : \((XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2)\).
La profondeur n'apporte alors pas la capacité d'expression recherchée.

Réponse orale : « ReLU permet d'apprendre des interactions non linéaires entre
les caractéristiques. Elle est simple et sa dérivée vaut 1 du côté positif.
Elle peut toutefois laisser des neurones inactifs ; LeakyReLU est une alternative. »

### Pourquoi Softmax en sortie ?

La dernière Dense produit des scores, appelés logits. Pour K classes :

\[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}.
\]

Les sorties sont positives et leur somme vaut 1. `argmax` choisit la classe au
score maximal. Ces probabilités ne constituent pas une garantie de calibration.

Le code soustrait le maximum de chaque ligne avant l'exponentielle. Cela ne change
pas le résultat mathématique, car le facteur commun s'annule au numérateur et
au dénominateur, et évite l'overflow de l'exponentielle pour des logits finis.

Réponse orale : « Les classes sont exclusives. Softmax représente cette compétition
entre classes ; plusieurs sigmoïdes indépendantes conviennent plutôt à des labels
qui peuvent coexister. »

Fichier : [activations.py](../src/neural_network/activations.py).

## 3. Comprendre comment le réseau apprend

### La perte mesure une erreur continue

Pour une cible one-hot y et une prédiction p :

\[
L=-\sum_k y_k\log(p_k)=-\log(p_{\text{vraie classe}}).
\]

Si la vraie classe est Check et que le modèle lui donne 0,2, la perte vaut
\(-\log(0,2)\approx1,609\). S'il lui donne 0,8, elle vaut environ 0,223.
L'objectif est de diminuer la perte moyenne sur les exemples d'entraînement.

Réponse orale : « La cross-entropy tient compte de la probabilité de la bonne
classe. L'accuracy ne compte que les bonnes réponses après argmax et ne fournit
pas un signal différentiable utile pour entraîner le réseau. »

### Le gradient indique la sensibilité de la perte

\(\partial L/\partial w\) décrit comment la perte varie pour une petite
modification d'un poids. Un gradient positif signifie qu'augmenter légèrement
ce poids augmente localement la perte.

La descente de gradient élémentaire fait donc :

\[
w\leftarrow w-\eta\frac{\partial L}{\partial w}.
\]

Avec w = 0,5, gradient = 0,2 et learning rate = 0,1, le nouveau poids vaut 0,48.
Ce pas vise une diminution locale ; un pas trop grand peut faire augmenter la perte.

### La rétropropagation applique la règle de la chaîne

Si \(L\) dépend de h et que h dépend de z :

\[
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial h}\frac{\partial h}{\partial z}.
\]

On commence à la sortie, puis chaque couche transmet le gradient à la précédente.
Il faut conserver les valeurs du forward utiles au backward : par exemple X
pour une Dense et le masque pour un Dropout.

Pour \(Z=XW+b\), en notant \(G=\partial L/\partial Z\) :

\[
\frac{\partial L}{\partial W}=X^TG,\qquad
\frac{\partial L}{\partial b}=\sum_{i=1}^{B}G_i,\qquad
\frac{\partial L}{\partial X}=GW^T.
\]

Vérification des dimensions : \((773\times64)(64\times512)=773\times512\)
pour les gradients de W. Chaque poids reçoit donc une correction de sa propre taille.

Pour ReLU, on transmet le gradient là où z > 0 et on le bloque là où z < 0.
La convention utilisée au point z = 0 est un gradient nul.

### Softmax + cross-entropy : le raccourci important

Pour une moyenne sur B exemples, avec cibles dont chaque ligne somme à 1,
sans pondération de classes :

\[
\frac{\partial L}{\partial z}=\frac{p-y}{B}.
\]

Intuition avec un seul exemple :

```text
Classes :         Nothing   Check   Checkmate
Prédiction p :       0,7     0,2       0,1
Cible y :           0       1         0
Gradient p - y :    0,7    -0,8       0,1
```

Pour les logits considérés directement, soustraire le gradient diminue les
scores des mauvaises classes et augmente celui de Check. Pour les poids,
la règle de la chaîne transforme ensuite ce signal.

Dans `Sequential.fit`, `backward_softmax()` fournit déjà le gradient par rapport
aux logits. Le backward saute alors la dernière Softmax. La dériver une deuxième
fois donnerait un gradient incorrect. La division par B est effectuée dans la
perte, pas à nouveau dans chaque Dense.

Ce raccourci doit être adapté pour une CCE pondérée ou une Focal Loss ; le code
implémente leurs dérivées spécifiques.

Fichiers : [losses.py](../src/neural_network/losses.py),
[model.py](../src/neural_network/model.py),
[tests numériques](../tests/test_training.py).

### Un lot, une époque, une mise à jour

1. Mélanger les exemples pour l'époque.
2. Prendre un mini-batch.
3. Calculer les probabilités : forward.
4. Calculer la perte.
5. Calculer les gradients : backward.
6. L'optimiseur met à jour les paramètres.
7. Recommencer sur les autres lots, puis évaluer en validation.

Une époque est un passage sur tout le jeu d'entraînement. Un batch est une
fraction des exemples. Pour 101 995 exemples et des lots de 64, une époque
comporte 1 594 mises à jour, dont un dernier lot de 43 exemples.

## 4. Défendre les principaux choix

| Question | Réponse courte | Limite à connaître |
| --- | --- | --- |
| Pourquoi NumPy ? | Les opérations matricielles sont vectorisées ; nous écrivons les couches, les gradients et l'entraînement. | « From scratch » ne signifie pas réimplémenter la multiplication matricielle. |
| Pourquoi un MLP ? | C'est un modèle générique, simple à implémenter et compatible avec notre vecteur de caractéristiques. Les couches denses relient directement toutes les entrées. | Un CNN est aussi envisageable. Nous n'avons pas prouvé que le MLP est optimal. |
| Pourquoi le one-hot ? | Il distingue les types de pièces sans imposer une distance numérique arbitraire entre eux. | La représentation est volumineuse et ne rend pas les relations tactiques explicites. |
| Pourquoi plusieurs couches ? | Elles permettent de composer des transformations non linéaires. | Davantage de capacité augmente le coût et peut favoriser le surapprentissage. |
| Pourquoi 512, 256, 128 neurones ? | C'est une architecture expérimentée pour combiner capacité et réduction progressive de dimension. | Aucun théorème ne désigne ces tailles comme les meilleures. Ne pas prétendre avoir isolé chaque effet. |
| Pourquoi ReLU ? | Calcul simple et gradient non saturant du côté positif. | Neurones potentiellement inactifs ; comparer à LeakyReLU est possible. |
| Pourquoi He ? | L'échelle initiale est adaptée aux couches suivies de ReLU. | Elle aide à contrôler les amplitudes, sans garantir la convergence. |
| Pourquoi Adam ? | Il lisse les gradients et adapte les pas par paramètre à partir de leurs moments. | Il n'est pas universellement meilleur ou plus rapide que SGD. |
| Pourquoi les mini-batches ? | Compromis entre mémoire, calcul vectorisé et fréquence des mises à jour. | La meilleure taille dépend du matériel et de l'apprentissage. |
| Pourquoi CCE ? | Elle convient à une classification à classes exclusives et fournit un gradient exploitable. | La Focal Loss existe dans la bibliothèque mais n'est pas la perte des deux essais à 75,27 % et 94,26 %. |
| Pourquoi un scheduler cosinus ? | Le learning rate diminue progressivement pour affiner les mises à jour en fin d'entraînement. | Il ne garantit pas un minimum global. |
| Pourquoi trois ou cinq classes ? | Trois répondent à la classification d'état ; cinq ajoutent la couleur qui donne l'échec. | Un score à trois classes ne se compare pas directement à un score à cinq classes. |

### Initialisation : pourquoi ne pas mettre tous les poids à zéro ?

Des neurones cachés identiques recevraient les mêmes gradients et resteraient
identiques. L'aléatoire casse cette symétrie. Le biais peut commencer à zéro
alors que les poids diffèrent.

Pour He normal : \(\sigma=\sqrt{2/n_{in}}\). Pour Xavier normal :
\(\sigma=\sqrt{2/(n_{in}+n_{out})}\). Le projet utilise He pour les Dense
cachées suivies de ReLU et Xavier pour la Dense de sortie.

### Adam : ce qu'il faut savoir expliquer

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

m mémorise une moyenne mobile des gradients ; v mémorise une moyenne mobile
de leurs carrés. v n'est pas directement une variance centrée.

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta\leftarrow\theta-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

La correction compense l'initialisation à zéro des moyennes mobiles. Epsilon
stabilise le dénominateur. Le code compte une itération par mise à jour du lot,
et non une par couche. Valeurs par défaut : beta1 = 0,9, beta2 = 0,999.

Dans ce projet, `weight_decay` ajoute lambda × W au gradient des poids avant
les moments d'Adam : c'est une régularisation L2 couplée, pas AdamW.

## 5. Surapprentissage : quatre outils, quatre rôles

Le surapprentissage apparaît lorsque le modèle continue à améliorer ses
résultats d'entraînement tandis que la généralisation se dégrade.

| Outil | Mécanisme | À l'oral |
| --- | --- | --- |
| Dropout | Met aléatoirement des activations à zéro pendant l'entraînement. | Réduit la dépendance à certaines activations. |
| L2 | Ajoute un terme proportionnel aux poids dans leur gradient. | Pénalise les poids trop grands. |
| Augmentation | Ajoute des positions obtenues par une symétrie valide. | Expose le réseau à des exemples de couleurs inversées. |
| Early stopping et restauration | Arrête après une période sans amélioration et restaure l'état sélectionné. | Conserve le modèle retenu en validation, qui peut précéder la dernière époque. |

Pour le Dropout inversé : \(\tilde h=mh/(1-p)\), avec m valant 1 avec
probabilité 1-p. Ainsi \(E[\tilde h]=h\). En inférence ordinaire, il est désactivé.

BatchNorm normalise les valeurs d'une caractéristique dans le lot :

\[
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad
y=\gamma\hat x+\beta.
\]

Gamma et beta sont appris. À l'inférence, on utilise les statistiques glissantes
enregistrées. Elles doivent donc être sauvegardées et restaurées avec les poids.
BatchNorm peut faciliter l'optimisation ; ne pas lui attribuer à elle seule une
amélioration dont plusieurs réglages ont changé simultanément.

L'augmentation inverse les rangées et les couleurs, change le joueur au trait
et adapte le roque et les labels. Elle est appliquée **après** la séparation.
Les miroirs reproduisant une entrée de validation sont exclus. Cette protection
spécifique ne remplace pas un audit de tous les doublons entre corpus.

## 6. Expliquer la représentation des échiquiers

Une FEN contient six champs : placement des pièces, joueur au trait, roques,
prise en passant, compteur de demi-coups et numéro de coup.

L'encodage basic utilise :

\[
64\times12+1+4=773.
\]

Il contient 12 indicateurs de pièces par case, le trait et quatre droits de
roque. Une case vide a ses 12 indicateurs à zéro. Le basic ne représente pas
la prise en passant : des états différents peuvent donc avoir le même vecteur.

L'encodage tactical conserve ces 773 valeurs et ajoute :

| Ajout | Taille |
| --- | ---: |
| Case de prise en passant | 64 |
| Cartes d'attaque des deux couleurs | 128 |
| Première pièce et distance dans huit directions autour de chaque roi | 2 × 8 × 19 = 304 |
| Contexte des cases voisines des deux rois | 2 × 8 × 4 = 64 |
| Total supplémentaire | 560 |
| Total tactical | 1 333 |

Réponse au jury : « Le basic donne surtout la position des pièces. Le tactical
rend explicites des relations utiles pour distinguer échec et mat. Ces calculs
géométriques n'utilisent pas le label attendu et ne remplacent pas la décision
du réseau. Par exemple, les cartes d'attaque incluent aussi des pièces clouées :
ce ne sont pas des listes de coups légaux. »

La couleur du label est celle **qui donne l'échec** : `Checkmate Black` signifie
que les Noirs donnent le mat, pas que leur roi est mat.

Le moteur de règles sert à des évaluations déterministes et à la génération de
données. `analyzer.run_predict` encode les FEN puis appelle `model.predict` ; il
ne remplace pas la prédiction par `evaluate_game_state`.

## 7. Les chiffres que tu peux défendre

### Comparaison contrôlée basic / tactical

Métadonnées vérifiées dans `model.nn` et `model_tactical.nn` : mêmes empreintes
de partitions avant augmentation, cinq classes, 60 000 positions sélectionnées,
51 000 pour apprendre et 9 000 pour valider. Après ajout de 50 995 miroirs,
l'entraînement comporte 101 995 exemples. Cinq miroirs de validation sont exclus.

| Mesure | model.nn | model_tactical.nn |
| --- | ---: | ---: |
| Entrées | 773 | 1 333 |
| Couches Dense | 773 → 512 → 256 → 128 → 5 | 1333 → 512 → 256 → 128 → 5 |
| Paramètres apprenables, BatchNorm comprise | 562 949 | 849 669 |
| Précision de validation | 75,27 % | 94,26 % |
| Bonnes réponses sur 9 000 | 6 774 | 8 483 |
| Macro-F1 | 0,7518 | 0,9424 |
| Époque sélectionnée | 19 | 18 |

Formulation exacte : « Sur notre partition commune de validation, la variante
tactique gagne 18,99 points de précision. Elle augmente aussi la première couche
de 286 720 poids ; nous n'avons pas isolé l'effet de la représentation à nombre
de paramètres égal. »

Le score de 94,26 % est une **validation**, sur une partition et une graine.
Il ne faut pas l'annoncer comme un score de test indépendant.

À l'époque 18, la précision tactique vaut 94,26 % et la CCE 0,1921. À l'époque
13, la CCE était plus basse, 0,1725, avec une précision inférieure de 93,73 %.
C'est possible : argmax et probabilité de la vraie classe mesurent deux choses
différentes. Le critère retenu était la précision, puis la perte en cas d'égalité.

### Modèles livrés : d'autres expériences

| Fichier | Architecture Dense réelle | Score enregistré |
| --- | --- | --- |
| my_torch_network_basic.nn | 773 → 128 → 64 → 3 | Pas de score de validation dans ses métadonnées |
| my_torch_network_medium.nn | 773 → 512 → 256 → 128 → 3 | 86,61 % en validation externe, 71 554 positions |
| my_torch_network_full.nn | 773 → 512 → 256 → 128 → 5 | 86,57 % en validation externe, 71 554 positions |

Ces deux derniers scores ne sont pas la comparaison à 9 000 positions et ne
permettent pas de classer directement les trois expériences. Le modèle local
tactique s'appelle `model_tactical.nn`, alors que le sujet exige pour le modèle
préentraîné remis un nom commençant par `my_torch_network`.

Accuracy = bonnes prédictions / nombre d'exemples. Pour une classe : précision =
VP / (VP + FP), rappel = VP / (VP + FN), F1 = 2PR / (P + R). La macro-F1 donne
le même poids à chaque classe. Dans la matrice du projet, les lignes sont les
classes attendues et les colonnes les classes prédites.

## 8. Points actuels à connaître avant de présenter les bonus

- Les anciennes fiches `defense_guide.md` et `defense_recap.md` contiennent des
  architectures et affirmations contradictoires. Ne pas réciter « Adam 3,4 fois
  plus rapide », « MLP optimal » ou « minimum global garanti ».
- Le code actuel de `build_enhanced_model(wide=True)` met par défaut un Dropout
  0,25 après la première couche cachée et 0,10 après la deuxième, sans Dropout
  après la troisième. Les deux modèles historiques à 75,27 % et 94,26 % ont,
  eux, 0,25 après chacune des trois couches cachées. Relancer la commande
  historique avec le code actuel ne reproduit donc pas exactement ces couches.
- Le parcours `analyzer --train` entraîne le modèle chargé pendant 15 époques,
  lots de 32, sans validation dédiée. Le parcours `src/my_torch.py -train` construit
  son architecture et réalise la sélection en validation. Ils ne suivent pas
  le même protocole expérimental.
- Le script génétique est dans `src/experiments/hyperparameter_tuning.py`.
  Actuellement, il passe `early_stopping_patience=0`, refusé par `Sequential.fit`,
  puis capture l'exception en retournant un score nul. Il faut corriger et
  revalider ce parcours avant de le présenter comme une optimisation réussie.
- `parse_fen` contrôle notamment la grille, les symboles et la présence des
  deux rois, mais accepte des champs omis. Le nettoyage des données applique
  des contrôles plus stricts. Ne pas annoncer une validation complète de toute
  la légalité historique des positions.
- Le moteur de règles a des simplifications ; éviter de le présenter comme un
  arbitre FIDE exhaustif. Le pat détecté par les règles n'est pas une sortie
  supplémentaire des réseaux actuels à trois ou cinq classes.

## 9. Démonstration courte

Les commandes d'aide, d'erreur, de prédiction ci-dessous, la génération en dossier
temporaire et le bonus `--auto 2 --depth 1` ont été vérifiés pendant la préparation.
Les 70 tests de `make tests_run` passent. Leurs affichages d'évaluation internes
portent sur de petits jeux de test logiciels, pas sur les performances des
modèles livrés.

Depuis la racine :

```sh
make
./my_torch_analyzer --help
./my_torch_analyzer --predict my_torch_network_medium.nn dataset/examples/sample_chessboards.txt
./my_torch_analyzer --predict my_torch_network_full.nn dataset/examples/sample_chessboards.txt
./my_torch_analyzer --predict model_tactical.nn dataset/examples/sample_chessboards.txt
```

Pour l'erreur, lancer les deux lignes consécutivement afin de lire le bon code :

```sh
./my_torch_analyzer --invalid-flag
echo $?
```

Résultat attendu : message sur stderr et code 84.

Pour générer un réseau vierge, utiliser un dossier de démonstration jetable et y
copier `basic_network.conf`, puis lancer le générateur par son chemin absolu :

```sh
/home/nathan-muzay/A_tek/S5-6/Torch/my_torch_generator basic_network.conf 1
```

Expliquer : « Il a une architecture et des poids initialisés, mais il n'a pas
encore appris. La sauvegarde permet de le transmettre à l'analyseur. »

Bonus facultatif, depuis la racine :

```sh
python3 bonus/play_ai.py --auto 2 --depth 1
```

Ne pas lancer un entraînement complet pendant l'oral. Ne pas présenter cinq
positions d'exemple comme un benchmark : commenter les erreurs éventuelles.

## 10. Répétition active : 90 minutes

| Temps | Exercice | Tu es prêt si… |
| --- | --- | --- |
| 0–20 min | Réexpliquer neurone, Dense, ReLU, Softmax et CCE sans lire. | Tu calcules la perte pour une bonne classe de probabilité 0,2. |
| 20–40 min | Refaire le backward Dense avec les dimensions et le signal p-y. | Tu expliques pourquoi on soustrait le gradient et pourquoi on saute le dernier backward Softmax. |
| 40–60 min | Répondre aux « pourquoi » du tableau, en 30 secondes chacun. | Chaque réponse comporte un bénéfice et une limite concrète. |
| 60–75 min | Présenter basic / tactical et l'overfitting. | Tu distingues validation et test, et gain observé et preuve causale. |
| 75–90 min | Faire la démo puis répondre à cinq questions au hasard. | Tu peux montrer les fichiers concernés sans chercher longtemps. |

Questions de jury à tirer au hasard :

1. Pourquoi deux Dense sans activation ne suffisent-elles pas ?
2. Quelle différence entre un poids, un gradient et un hyperparamètre ?
3. Que signifie un gradient positif ? Que se passe-t-il si le learning rate est trop grand ?
4. Pourquoi la cross-entropy et pas directement l'accuracy ?
5. D'où vient la forme X transposé × gradient dans une Dense ?
6. Pourquoi ne pas appliquer deux fois la dérivée de Softmax ?
7. Pourquoi des poids aléatoires ? Pourquoi He avec ReLU ?
8. Que changent Adam, Dropout et BatchNorm, chacun séparément ?
9. Pourquoi l'accuracy peut-elle monter quand la perte monte aussi ?
10. Pourquoi séparer les données avant l'augmentation ?
11. Comment prouver que le calcul du gradient est correct ?
12. Que représente exactement le 94,26 % annoncé ?
13. Le tactical utilise des règles : où reste l'apprentissage ?
14. Pourquoi pas un CNN ? Qu'est-ce que vous avez réellement comparé ?
15. Quelle limite du projet amélioreriez-vous en premier ?

Pour la question 11 : comparer le gradient analytique à
\([L(w+\epsilon)-L(w-\epsilon)]/(2\epsilon)\). Les tests de
`tests/test_training.py` vérifient notamment CCE et Focal Loss, gradients sur
probabilités et logits, et une mise à jour complète. Ne pas extrapoler ces
contrôles à une preuve exhaustive sur toutes les entrées possibles.

Pour la question 15 : proposer une expérience concrète, par exemple un contrôle
à capacité comparable entre basic et tactical, plusieurs graines, puis une
évaluation finale sur un test distinct après audit des recouvrements. Ne pas
réutiliser ce test pour choisir les hyperparamètres.

Une réponse solide peut être : « Nous avons choisi X pour cette raison ; nous
avons mesuré cet effet dans ce protocole ; cette limite reste à tester. »
