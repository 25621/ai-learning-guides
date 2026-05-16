# Frozen Lake avec une politique aléatoire 🧊

## Qu'est-ce que Frozen Lake ?

Imaginez que vous jouez sur un **étang gelé** avec vos amis.

La glace est généralement solide, mais certains endroits comportent des **trous** — si vous marchez sur un trou, vous tombez dedans et la partie est terminée ! À une extrémité de l'étang se trouve un **cadeau** 🎁. Votre mission est de glisser du **départ** jusqu'au **cadeau** sans tomber.

Voici à quoi ressemble le lac gelé (une grille de 4×4 carrés) :

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Start (Départ : là où vous commencez)
- **F** = Frozen ice (Glace : zone sûre !)
- **H** = Hole (Trou : vous tombez, fin de partie 😨)
- **G** = Goal (Objectif : le cadeau ! 🎁)

---

## La difficulté : La glace glisse !

Sur un vrai étang gelé, quand vous essayez de marcher vers la *droite*, la glace vous fait parfois glisser vers le *haut* ou vers le *bas* à la place ! C'est ce qui rend le jeu difficile.

Même si vous *voulez* aller à droite, le jeu peut vous faire glisser ailleurs. C'est ce qu'on appelle la **stochasticité** — un mot savant pour dire que "les choses ne se passent pas toujours comme prévu".

---

## Qu'est-ce qu'une politique aléatoire ?

Une **politique** (policy) est simplement un plan : "Dans telle situation, je ferai TELLE action."

Une **politique aléatoire** signifie : "Je n'ai aucun plan ! Je choisirai une direction au hasard à chaque fois — haut, bas, gauche ou droite — comme si je lançais un dé !"

C'est comme un bébé qui marcherait sur la glace sans avoir aucune idée d'où se trouve le cadeau.

---

## Ce que notre code a révélé

Nous avons testé la politique aléatoire sur **1 000 parties** :

| Résultat | Valeur |
|----------|--------|
| **Nombre de fois où le cadeau a été atteint** | 11 sur 1 000 (1,1 %) |
| **Nombre moyen de pas par partie** | 7,5 pas |
| **Partie la plus courte** | 2 pas |
| **Partie la plus longue** | 33 pas |

La plupart du temps, le marcheur aléatoire est tombé rapidement dans un trou. Seulement 1 partie sur 100 s'est terminée par la découverte du cadeau !

---

## Pourquoi est-ce utile ?

Même si la politique aléatoire est médiocre, elle nous donne une **référence** (baseline) — un point de départ pour comparer nos futurs résultats.

Quand nous construirons plus tard une politique *intelligente* (en utilisant le Q-learning ou d'autres algorithmes), nous pourrons dire : "Notre agent intelligent réussit 75 % du temps — c'est bien mieux que les 1 % du marcheur aléatoire !"

**Exemple de la vie réelle :** Imaginez essayer de trouver votre salle de classe dans une nouvelle école en tournant au hasard à gauche ou à droite à chaque couloir. Vous finirez peut-être par y arriver, mais cela prendrait énormément de temps ! Une politique intelligente, c'est comme avoir un plan.

---

## Ce que montre la carte thermique (Heatmap)

Dans notre illustration, la **carte thermique** montre quels carrés le marcheur aléatoire a visités le plus souvent :

- Le carré de **Départ** est très fréquenté (chaque partie commence là).
- Les carrés près des **trous** sont moins visités (le marcheur tombe souvent avant de les atteindre).
- L'**Objectif** est très rarement atteint car le marcheur aléatoire n'y parvient presque jamais.

Cela nous apprend une chose importante : la politique aléatoire reste bloquée près du début et n'explore jamais vraiment tout le lac.

---

## Mots-clés à retenir

- **Politique (Policy)** : Votre plan d'action pour chaque situation.
- **Politique aléatoire** : Aucun plan — vous choisissez juste une action au hasard !
- **Référence (Baseline)** : Un résultat de base qui sert de comparaison (à quel point pouvons-nous faire mieux ?).
- **Stochastique** : Les choses ne se passent pas toujours comme prévu (comme sur de la glace glissante !).
- **Taux de réussite** : À quelle fréquence avons-nous gagné ? (Ici : 1,1 % — très faible !).

L'idée à retenir : **Une politique aléatoire est un point de départ. Apprendre réellement, c'est construire un meilleur plan !**
