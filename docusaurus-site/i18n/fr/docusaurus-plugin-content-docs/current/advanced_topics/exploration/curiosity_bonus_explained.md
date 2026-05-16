# Bonus de Curiosité (Motivation Intrinsèque) 🧭

## Qu'est-ce que c'est ?

Imaginez un tout-petit déposé dans une nouvelle pièce. Personne ne le paie, personne n'applaudit — et pourtant, il se dirige droit vers le placard qu'il n'a pas encore ouvert, le bouton sur lequel il n'a pas encore appuyé, le jouet bruyant dans le coin. Il est guidé par une **récompense interne** : *"Ça a l'air nouveau. Vas voir ce que c'est."*

Un **bonus de curiosité** donne à un agent d'apprentissage par renforcement la même impulsion interne. La récompense réelle de l'environnement (la récompense "extrinsèque" — points, argent, victoire) reste exactement telle qu'elle est. Nous ajoutons simplement une seconde récompense, auto-générée, pour avoir visité des choses que l'agent juge *nouvelles* ou *surprenantes*, et nous l'entraînons sur la somme des deux :

```
récompense apprise par l'agent = récompense réelle + beta * bonus de curiosité
```

`beta` est un bouton de réglage qui commence à une valeur élevée (sois curieux !) et diminue avec le temps (arrête de traîner, va exploiter ce que tu as appris).

## Pourquoi s'en préoccuper ? Le problème de la récompense éparse (Sparse-Reward)

Les agents RL normaux apprennent des récompenses qu'ils reçoivent réellement. Cela fonctionne très bien quand les récompenses sont partout ("+1 à chaque étape où tu restes debout" dans CartPole). Cela s'effondre lorsque la récompense est **éparse** (sparse) — zéro, zéro, zéro, ... , zéro, et enfin un +1 après une longue séquence très spécifique d'actions correctes.

Exemples réels de récompenses éparses :

- **Montezuma's Revenge** (le jeu Atari) : votre premier point n'arrive qu'après environ 100 mouvements précis — descendre une échelle, esquiver un crâne, remonter, attraper une clé. Jusque-là, le score reste désespérément à zéro.
- **Un cadenas à combinaison.** 9 999 mauvais codes ne vous donnent rien ; un seul vous donne le prix.
- **Découverte de médicaments / expériences scientifiques.** Des milliers d'essais ratés, puis un qui fonctionne.
- **Écrire une longue démonstration ou un programme.** Aucun point partiel n'est accordé tant que l'ensemble n'est pas vérifié.

Dans ces contextes, un agent qui ne jure que par la récompense est comme un étudiant qui refuse d'étudier à moins d'être payé par bonne réponse à l'examen final — il ne commence jamais. La curiosité est le bonus qui dit *"explorer est sa propre récompense,"* de sorte que l'agent continue de chercher jusqu'à ce qu'il tombe sur le vrai prix.

## Deux types de curiosité (tous deux implémentés dans `curiosity_bonus.py`)

### 1. Nouveauté basée sur le comptage (Count-based novelty) : "J'ai à peine été ici"

Le signal de nouveauté le plus simple possible. On tient un décompte `N(s, a)` du nombre de fois où l'action `a` a été effectuée dans l'état `s`, et on s'accorde un bonus qui diminue à mesure que ce décompte augmente :

```
bonus de curiosité = 1 / sqrt( N(s, a) + 1 )
```

La première fois que vous essayez quelque chose : bonus = 1,0. Après 100 essais : bonus = 0,1. Après 10 000 essais : 0,01. L'agent est récompensé pour aller là où il n'a pas encore été, et l'attrait s'estompe naturellement pour les endroits déjà bien explorés.

**Analogie concrète :** un touriste avec une liste de "lieux que je n'ai pas visités". Un tout nouveau quartier ? Priorité absolue. Le café où vous êtes allé cinquante fois ? Plus très excitant.

C'est l'une des idées les plus anciennes (MBIE-EB, UCB). Sa faiblesse : dans un monde immense ou continu, on ne visite jamais deux fois *exactement* le même état, donc le décompte brut est toujours 1 — c'est pourquoi le type suivant existe.

### 2. Nouveauté basée sur l'erreur de prédiction (Prediction-error novelty) : "Je ne l'avais pas vu venir"

C'est l'idée derrière le célèbre **ICM** (Intrinsic Curiosity Module, Pathak et al. 2017) et son cousin **RND** (Random Network Distillation, Burda et al. 2018). Au lieu de compter, l'agent utilise un petit **modèle qui essaie de prédire ce qui va se passer ensuite** — "si je suis ici et que je fais ça, où vais-je finir ?" — et se récompense en fonction de **l'erreur du modèle** :

```
bonus de curiosité = surprise = -log P( l'état réellement atteint | où j'étais, ce que j'ai fait )
```

- Une situation que le modèle n'a jamais vue → il prédit mal → grande surprise → gros bonus → "va explorer là-bas !"
- Une situation que le modèle a vue cent fois → il prédit parfaitement → zéro surprise → zéro bonus → "déjà vu, compris, on passe à autre chose."

**Analogie concrète :** un enfant qui apprend comment le monde fonctionne en jouant. Pousser un verre de la table pour la *première* fois est fascinant (il s'est brisé !). La centième fois, vous saviez déjà qu'il se briserait — ce n'est plus intéressant. Curiosité = l'écart entre ce que vous attendiez et ce qui s'est réellement passé.

Dans notre code tabulaire, le "modèle" est juste un tableau de décomptes de transitions, et "l'erreur" est la surprise (surprisal) `-log P`. Les vrais ICM/RND utilisent des réseaux de neurones pour que la même idée fonctionne sur des pixels bruts — mais le principe est identique.

> **Pourquoi deux versions ?** La méthode basée sur le comptage est extrêmement simple et constitue une excellente ligne de base. L'erreur de prédiction s'adapte aux mondes vastes et sans répétition, et donne un signal plus *net* : dans un environnement déterministe, dès que vous avez vu une transition une fois, la surprise tombe instantanément à ~0, alors qu'un bonus de comptage ne s'estompe que lentement en `1/sqrt(N)`. Dans nos expériences, l'agent avec erreur de prédiction résout MiniMontezuma en quelques dizaines d'épisodes ; l'agent de comptage y parvient aussi, mais plus lentement et de manière moins fiable.

## Ce que fait notre code

`curiosity_bonus.py` entraîne un simple **Q-learner tabulaire** sur `MiniMontezumaEnv` — un minuscule monde en grille de deux pièces où vous devez marcher jusqu'à une **clé**, la ramasser (ce qui ouvre la **porte**), la traverser et atteindre le **trésor**. La récompense (+1) n'apparaît *que* sur le trésor, après environ 15 mouvements parfaits. Le script lance trois agents et les compare :

| Agent | Comportement sur MiniMontezuma |
|-------|-------------------------------|
| **epsilon-greedy (pas de curiosité)** | Erre près du départ, n'atteint *jamais* la clé, le score reste à 0 pour toujours. |
| **bonus basé sur le comptage** | Trouve la clé de manière fiable ; parcourt toute la chaîne jusqu'au trésor dans environ 40 % des épisodes. Fonctionne — mais avec un peu de bruit. |
| **bonus d'erreur de prédiction** | Atteint la clé *et* le trésor dès les 20–25 premiers épisodes ; à mesure que `beta` diminue, il finit par résoudre le jeu à chaque épisode. |

La figure montre :
- une courbe d'apprentissage : *P(atteindre le trésor)* au cours de l'entraînement,
- une seconde courbe pour l'étape intermédiaire *P(ramasser la clé)*,
- et des **cartes de chaleur (heat-maps) de fréquentation des états** sur la grille — l'agent sans curiosité reste un petit pâté près du départ ; les agents curieux inondent les *deux* pièces.

## Le mécanisme en une image

```
            sans curiosité                       avec bonus de curiosité
   récompense :  0 0 0 0 0 0 0 0 ... 0  (+1?)        0 0 0 0 0 0 0 0 ... 0  (+1!)
                 └──── rien pour apprendre ────┘     └ + 0.4 0.3 0.9 0.2 ... ┘  (auto-générée,
                                                       dense, pointe "vers la nouveauté")
   résultat :  marche aléatoire, ne trouve jamais +1   balaye systématiquement le monde,
                                                        tombe sur +1, puis le bonus s'estompe
```

Le bonus de curiosité transforme le "je n'ai pas vu ça" en récompense, de sorte que l'agent **pousse délibérément vers les territoires inexplorés** au lieu de s'agiter au hasard. Et comme le bonus diminue à mesure que les choses deviennent familières (et que `beta` décroît), une fois que l'agent a trouvé la récompense réelle, il s'arrête naturellement de traîner et commence à exploiter son savoir.

## Quelques bémols honnêtes

- **Le problème de la "télévision bruyante" (noisy-TV problem).** Un agent à erreur de prédiction peut être hypnotisé par une source de pur hasard (une télé affichant de la neige, un lancer de dés) — il ne peut *jamais* le prédire, donc la surprise ne s'estompe jamais. La véritable astuce d'ICM est de prédire dans un *espace de caractéristiques appris* qui ignore ce que l'agent ne peut pas contrôler ; RND contourne le problème différemment. Notre monde en grille déterministe n'a pas de télé bruyante, donc nous n'y sommes pas confrontés.
- **La curiosité est un moyen, pas une fin.** C'est pourquoi `beta` diminue. Un agent qui resterait curieux au maximum pour toujours ne se poserait jamais pour réellement *gagner*.
- **Passer à l'exploration profonde reste difficile.** Un bonus dans la récompense aide, mais le Q-learning tabulaire classique est lent à propager l'optimisme résultant le long d'une chaîne étendue (voir `compare_exploration.py`). Craquer Montezuma avec des pixels a nécessité plus de puissance — RND avec un réseau de neurones, DQN bootstrapé, Go-Explore.

## Mots-clés à retenir

| Mot | Signification |
|------|---------|
| **Récompense intrinsèque** | Une récompense que l'agent génère pour lui-même, distincte de celle de l'environnement |
| **Récompense extrinsèque** | La récompense réelle de l'environnement (points, victoire/défaite) |
| **Récompense éparse (Sparse reward)** | La récompense est nulle presque partout ; vous ne l'obtenez qu'après une longue séquence correcte |
| **Nouveauté / surprise** | À quel point un état (ou une transition) est nouveau ou inattendu — ce que la curiosité récompense |
| **Bonus basé sur le comptage** | Nouveauté ≈ `1/sqrt(nombre de visites)` — le bonus d'exploration classique |
| **ICM** | Intrinsic Curiosity Module : nouveauté = erreur de prédiction d'un modèle (dans un espace de caractéristiques appris) |
| **`beta`** | Le poids du bonus de curiosité ; généralement diminué vers 0 pour que l'agent finisse par exploiter |

## Résumé en une phrase

> **Un bonus de curiosité est une récompense auto-attribuée pour la nouveauté — il fabrique un signal dense de type "va explorer là-bas" qui entraîne l'agent à travers des mondes à récompenses éparses qu'il ne résoudrait jamais autrement, puis s'efface poliment une fois que tout est devenu familier.**
