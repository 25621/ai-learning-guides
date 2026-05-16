# S'entraîner sur Montezuma's Revenge 🏛️🔑

## Pourquoi ce jeu est célèbre (dans le milieu du RL)

En 2015, le DQN de DeepMind a appris à jouer à des dizaines de jeux Atari à un niveau surhumain à partir de pixels bruts. Cela a fait la une des journaux. Mais enfoui dans le tableau des résultats se trouvait un jeu où le DQN a obtenu un score de **0** — le même résultat que s'il n'avait rien fait du tout : **Montezuma's Revenge**.

Pourquoi ? Regardez ce que le jeu vous demande dès la toute première salle :

1. Descendre une échelle.
2. Traverser une corniche.
3. Sauter par-dessus un crâne qui roule (un mauvais timing → vous mourrez).
4. Monter une autre échelle.
5. Attraper la clé.

Cela représente environ **100 pressions précises sur les boutons**, et le jeu ne vous donne **pas un seul point** tant que la clé n'est pas en main. Le signal de récompense est un **zéro** plat et sans relief pendant toute la séquence d'ouverture.

Un agent RL normal apprend en s'ajustant vers les récompenses qu'il reçoit réellement. Si la récompense est nulle partout où il peut aller, il n'y a **rien à apprendre** — c'est comme essayer de trouver le fond d'une vallée parfaitement plate en cherchant une pente vers le bas. Le DQN s'est donc contenté de s'agiter sur la plateforme de départ pour l'éternité. Montezuma est devenu *le* point de référence (benchmark) pour l'**exploration difficile** : le jeu que l'on ne peut battre que si l'on explore de manière *intelligente*, et non aléatoire.

La percée est survenue en 2018 avec la **Random Network Distillation (RND)** — et l'astuce consistait précisément à ajouter un **bonus de curiosité intrinsèque** pour que l'agent se récompense *lui-même* lorsqu'il atteint de nouveaux écrans. Soudainement, il disposait d'un signal dense l'attirant plus profondément dans le niveau. Le RND a obtenu un score surhumain sur Montezuma. (Plus tard : Go-Explore, Agent57, …)

## Exemples concrets de récompenses parcimonieuses de type "Montezuma"

- **Une serrure à combinaison / une chasse au trésor avec des indices cryptiques.** Pas de crédit partiel. Vous êtes à zéro jusqu'à ce que vous atteigniez soudainement le prix.
- **Faire accepter un article de recherche, ou rendre une startup rentable.** Des mois sans récompense externe, puis (peut-être) une très grosse.
- **Un parcours de speedrun dans un jeu vidéo.** Des dizaines d'entrées parfaites au pixel et à la frame près sans aucun feedback jusqu'à ce que l'astuce fonctionne ou non.
- **Escape rooms.** La salle ne vous dit presque rien tant que vous n'avez pas enchaîné plusieurs découvertes.

Dans tous ces cas, "essayer des trucs au hasard" est sans espoir. Vous devez explorer *systématiquement* — et un signal interne du type "oh, c'est nouveau, continuons" est ce qui vous permet de rester systématique.

## Pourquoi nous ne nous entraînons pas réellement sur le Montezuma en pixels ici

Le faire correctement impliquerait :

- un réseau convolutionnel pour voir l'écran RGB de 210×160,
- de l'empilement de frames (pour que l'agent sache dans quelle direction le crâne bouge),
- un module RND (deux réseaux supplémentaires : une "cible" aléatoire fixe et un "prédicteur" entraîné),
- et **des dizaines de millions de frames d'environnement** — soit de nombreuses heures de calcul sur GPU.

C'est un projet de recherche, pas un script pédagogique. `montezuma_revenge.py` fait donc deux choses honnêtes à la place :

### 1. Il interagit avec le vrai jeu (si `ale-py` est installé)

Il charge `ALE/MontezumaRevenge-v5` via Gymnasium, lance un **agent aléatoire uniforme pendant 2000 étapes**, et rapporte la récompense totale. Le nombre qu'il affiche est presque toujours **0.0** — l'expression abstraite "récompense parcimonieuse" devient un fait concret, vérifiable par vous-même. Si le package Atari n'est pas installé, il affiche la commande `pip install` et passe à la suite.

### 2. Il entraîne un agent tabulaire sur un "modèle réduit" : `MiniMontezumaEnv`

Il s'agit d'un minuscule monde de grille (gridworld) avec le même *squelette* que la première salle de Montezuma :

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = départ (start)
#.....D.......#     K = clé (key)      D = porte (passable uniquement avec la clé)
#..K..#.......#     T = trésor (le SEUL carreau qui donne une récompense)
###############
```

Pour gagner, vous devez : marcher jusqu'à la **clé** (~6 mouvements), la ramasser ; marcher jusqu'à la **porte** (~4 mouvements) — qui s'ouvre alors ; traverser et atteindre le **trésor** (~5 mouvements). Environ **15 mouvements parfaits**, avec **zéro feedback jusqu'au trésor**. Le drapeau `has_key` fait partie de l'état de l'agent, donc une fois que vous avez la clé, il y a toute une deuxième salle de *nouveaux* états à découvrir — tout comme de nouveaux écrans s'ouvrant dans le vrai jeu.

Nous entraînons ensuite un simple **Q-learner tabulaire** deux fois :

| Agent | Résultat sur MiniMontezuma |
|-------|--------------------------|
| **sans curiosité (epsilon-greedy)** | Le retour reste à **0** pendant les 1 500 épisodes. Il n'atteint même jamais la clé. (Cela vous rappelle quelque chose ? C'est le DQN sur le vrai jeu.) |
| **avec un bonus de curiosité par erreur de prédiction** | Atteint le trésor en ~20–25 épisodes puis apprend le **parcours optimal de 15 étapes**. (C'est l'idée du RND, réduite pour tenir dans une table Q.) |

La figure montre les deux courbes d'apprentissage côte à côte, ainsi que le parcours réel appris par l'agent curieux, dessiné sur la grille (départ → clé → porte → trésor). Le script affiche également ce parcours sous forme de frames ASCII.

## La Leçon

> **La "récompense parcimonieuse" n'est pas une bizarrerie d'un jeu Atari étrange — c'est la norme dans tout monde où le succès nécessite une séquence longue et spécifique d'actions.** Un agent uniquement basé sur la récompense (DQN classique) ne peut littéralement pas démarrer : il n'y a pas de gradient à suivre. Un bonus de curiosité en fabrique un — un signal dense, auto-généré, "c'est nouveau, continue" — et ce signal est ce qui porte l'agent à travers le désert de zéros jusqu'à la première récompense réelle. Tout ce qui suit (RND, Go-Explore, Agent57) est une version à plus grande échelle, basée sur des réseaux de neurones, de ce même principe.

## Mots clés à retenir

| Mot | Signification |
|------|---------|
| **Exploration difficile** | Problèmes où l'on ne réussit qu'en explorant intelligemment ; l'exploration aléatoire échoue |
| **Récompense parcimonieuse (Sparse)** | La récompense est nulle presque partout ; on ne l'obtient qu'après une longue séquence correcte |
| **Montezuma's Revenge** | Le jeu Atari sur lequel les agents deep-RL classiques (DQN, A3C) ont obtenu 0 — le benchmark canonique de l'exploration difficile |
| **RND (Random Network Distillation)** | La méthode de 2018 qui a battu Montezuma en utilisant un bonus de curiosité par erreur de prédiction |
| **Go-Explore** | "Mémoriser les états prometteurs, y revenir, puis explorer à partir de là" — un autre champion de Montezuma |
| **Modèle réduit (Scale model)** | Un petit environnement peu coûteux qui conserve la *structure* d'un problème complexe pour l'étudier rapidement |

## Résumé en une phrase

> **Montezuma's Revenge est le jeu qui a appris au RL que "des récompenses que l'on ne reçoit jamais ne peuvent rien vous apprendre" — et la solution, hier comme aujourd'hui, est un bonus de curiosité qui permet à l'agent de se récompenser lui-même pour son exploration jusqu'à ce qu'il trouve le véritable prix.**
