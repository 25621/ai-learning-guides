# Politique conditionnée par l'objectif

## L'idée maîtresse : Une seule politique pour tout diriger

Imaginez que vous êtes un chauffeur-livreur. Vous n'avez pas besoin d'une compétence totalement différente pour chaque adresse. Vous savez conduire, lire un plan et naviguer dans le trafic — il vous suffit de saisir la *destination du jour* et c'est parti.

Une **politique conditionnée par l'objectif** (goal-conditioned policy) fonctionne de la même manière. Au lieu d'entraîner un agent qui ne peut se rendre qu'à un seul objectif fixe, nous entraînons un agent unique qui accepte n'importe quel objectif en entrée et trouve comment y arriver.

---

## En quoi cela diffère-t-il du RL standard

Dans l'apprentissage par renforcement (RL) standard (tel que vu dans les phases précédentes), la fonction de récompense est figée : "atteindre la case (7, 7), obtenir +1." L'agent n'apprend qu'une seule chose : comment atteindre *cette* case précise.

Dans le RL conditionné par l'objectif, la récompense dépend du fait que l'agent atteigne ou non *l'objectif qui lui a été assigné pour cette fois*. La politique apprend :

> **"Étant donné l'endroit où je suis et celui où je veux être, que dois-je faire ?"**

L'objectif voyage *avec* l'agent, comme une destination saisie dans une application de navigation.

---

## Le problème des récompenses éparses (Sparse Rewards)

C'est là que le bât blesse : apprendre à partir de récompenses éparses (seulement +1 à l'objectif, 0 partout ailleurs) est extrêmement difficile. La plupart des tentatives échouent — l'agent erre au hasard, ne rencontre jamais l'objectif, et le réseau ne reçoit rien d'utile pour apprendre.

Imaginez essayer d'apprendre à lancer des fléchettes les yeux bandés. Vous lancez mille fois et vous ratez toujours la cible. Après mille échecs, vous n'avez toujours aucune idée de ce qu'est un "bon lancer".

C'est ici qu'intervient le **Hindsight Experience Replay (HER)**.

---

## Hindsight Experience Replay : Tirer profit de l'échec

L'astuce de HER est d'une simplicité magnifique. Après un épisode raté, HER pose la question :

> *"Même si tu n'as pas atteint ton objectif... où as-tu finalement atterri ?"*

Il **rejoue alors ce même épisode**, mais en faisant comme si la position finale réelle de l'agent **était** l'objectif depuis le début. Soudain, un épisode raté devient une réussite — pour un objectif différent.

C'est comme un joueur de basket qui s'entraîne au tir et rate systématiquement le panier. HER dirait : "D'accord, tu as touché le mur de gauche à chaque fois. Félicitations — tu es excellent pour toucher le mur de gauche ! Enregistrons ces lancers comme des tentatives réussies pour toucher le mur de gauche." Au fil du temps, le joueur développe une compétence pour atteindre *n'importe quelle* cible, et finit par transférer cette compétence vers le vrai panier de basket.

Cela transforme des milliers d'"échecs" en une riche bibliothèque de navigations *réussies* vers de nombreux endroits différents. L'agent apprend à les atteindre tous, ce qui se généralise ensuite à la cible réelle.

---

## Analogie de la vie réelle : Un bambin apprenant à empiler des blocs

Un enfant qui essaie de mettre un bloc dans un seau rate constamment. Mais chaque "échec" dépose le bloc *quelque part*. Si vous rejouez chaque échec comme "tu essayais de le mettre *juste là* — et tu as réussi !", l'enfant développe une motricité fine sur toute la surface de la table. Bientôt, il est capable de placer un bloc n'importe où — y compris dans le seau.

---

## Ce que fait notre code

Le script `goal_conditioned_policy.py` s'exécute dans un **labyrinthe de 7x7** avec des murs. Au début de chaque épisode, une case objectif aléatoire est choisie. L'agent doit la trouver.

La politique prend deux entrées à chaque étape :
1. Où l'agent se trouve actuellement.
2. Où il veut aller.

Après chaque épisode (réussi ou non), HER génère plusieurs "réussites" synthétiques supplémentaires en réétiquetant les positions réellement visitées comme des objectifs alternatifs.

L'entraînement dure 3 000 épisodes avec un taux d'exploration décroissant — l'agent explore davantage au début, puis fait de plus en plus confiance à ce qu'il a appris.

---

## Ce que montrent les graphiques

![Résultats de la politique conditionnée par l'objectif](outputs/goal_conditioned_policy.png)

**Gauche — Taux de réussite au fil de l'entraînement :** Chaque épisode est soit une réussite (objectif atteint), soit un échec. La courbe monte régulièrement à mesure que la compétence de navigation universelle de l'agent s'améliore. À la fin, l'agent atteint n'importe quel objectif presque à chaque fois.

**Droite — Carte thermique du taux de réussite par objectif :** Après l'entraînement, nous testons l'agent sur chaque case objectif possible et colorons chaque case en fonction de la fréquence à laquelle l'agent l'atteint. Le vert signifie que l'agent atteint ce point de manière fiable ; le rouge signifie qu'il a encore des difficultés. Un agent bien entraîné affiche principalement du vert sur l'ensemble du labyrinthe.

---

## Où cela apparaît-il dans le monde réel ?

| Application | L'"objectif" |
|-------------|--------------|
| Bras robotique atteignant une cible | Position cible en 3D |
| Voiture autonome | Coordonnées GPS |
| Assistant de modèle de langage | Instruction de l'utilisateur |
| Personnage non-joueur dans un jeu vidéo | N'importe quel point de passage sur la carte |

Les politiques conditionnées par l'objectif sont l'un des piliers de HIRO (Hierarchical RL with subgoals) — le gestionnaire de haut niveau choisit un sous-objectif, et l'ouvrier de bas niveau est précisément ce type de politique conditionnée par l'objectif.

---

## Résumé en une phrase

> **Une politique conditionnée par l'objectif est un agent capable de naviguer vers n'importe quelle destination — et HER rend possible l'apprentissage par l'échec en faisant comme si chaque tir raté visait l'endroit où il a atterri.**
