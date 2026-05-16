# Agent Q-Learning pour Frozen Lake 🧊

## Qu'est-ce que c'est ?

Imaginez un étang gelé avec de la glace glissante. Il y a une **case de départ** (Start) et une **case d'arrivée** (Goal) avec quelques **trous** (Holes) au milieu. Si vous tombez dans un trou, vous recommencez !

La glace est glissante, donc même si vous essayez d'aller à droite, vous pourriez glisser vers le haut ou vers le bas à la place. Un **agent Q-Learning** est un robot qui apprend — en essayant encore et encore — comment aller du départ à l'arrivée sans tomber !

---

## Que signifie le "Q" dans Q-Learning ?

Le **"Q"** signifie **"Quality"** (Qualité) — plus précisément, la *qualité* de prendre une action particulière dans une situation particulière.

Voyez cela comme une évaluation de restaurant : "Quelle est la qualité de la pizza dans CE restaurant ?" Q(s, a) demande : "Quelle est la qualité de l'action **a** quand je suis dans l'état **s** ?"

Une valeur Q élevée signifie : "Excellent choix ! Cette action mène à beaucoup de récompenses."
Une valeur Q faible signifie : "Mauvaise idée ! Cette action mène généralement à des ennuis."

**Exemple concret :** Imaginez que vous êtes un enfant qui décide s'il doit manger des bonbons avant le dîner. Votre valeur Q pour "manger des bonbons maintenant" pourrait être élevée tout de suite (c'est délicieux !) mais faible globalement (maman se fâche, vous vous sentez mal plus tard). Le Q-learning apprend à prendre en compte ces conséquences futures — pas seulement le sentiment immédiat !

---

## La grande idée : Un tableau magique de scores

Le Q-Learning construit un grand tableau appelé la **Q-table**. Chaque ligne est une case sur la glace, et chaque colonne est une action (gauche, droite, haut, bas). Les nombres à l'intérieur sont des **scores** : "À quel point est-il bon de prendre cette action depuis cette case ?"

Chaque fois que le robot tente un mouvement :
1. Il reçoit un feedback (est-il tombé ? a-t-il atteint l'objectif ?)
2. Il met à jour le score dans le tableau en utilisant cette formule :

> **Nouveau Score = Ancien Score + Taux d'Apprentissage × (Ce qui s'est réellement passé − Ce que j'attendais)**

Le robot se demande essentiellement : "Ce mouvement était-il meilleur ou pire que ce que je pensais ?"

**Exemple concret :** Pensez à un bébé qui apprend à marcher. Chaque fois qu'il essaie de faire un pas et tombe, il apprend que "ce pas était mauvais". Chaque fois qu'il réussit, il se souvient que "ça a marché !". Après de nombreux essais, il comprend comment marcher. Le Q-learning fait la même chose, mais avec un tableau !

---

## Ce qui rend le Q-Learning spécial : C'est "Off-Policy" !

Voici quelque chose d'astucieux : quand le Q-Learning met à jour son tableau, il *suppose toujours qu'il fera le mouvement parfait la prochaine fois*, même si pendant l'entraînement il explore parfois des mouvements aléatoires.

Cela rend le Q-Learning **hors-politique** (off-policy) : la stratégie qu'il *apprend* (toujours choisir la meilleure action connue) est distincte de la stratégie qu'il *suit* pendant l'entraînement (parfois choisir une action aléatoire pour explorer). Concretement, la mise à jour de la Q-table utilise la valeur Q *maximale* de l'état suivant — le meilleur théorique — même lorsque le mouvement suivant réel du robot sera aléatoire.

En termes simples : le robot peut errer aléatoirement vers la gauche pour explorer, mais son apprentissage calcule toujours comme s'il prenait la *meilleure* action suivante. Cette séparation permet au Q-Learning de converger vers la stratégie optimale quel que soit son degré d'exploration.

---

## Ce que notre code a trouvé

Nous nous sommes entraînés pendant **50 000 épisodes** sur le Frozen Lake 4×4 glissant :

| Métrique | Résultat |
|--------|--------|
| Taux de réussite de l'évaluation gloutonne | **73,1%** |
| Objectif jalon (>70%) | ✓ **RÉUSSI** |

La glace est très glissante, donc même la meilleure politique ne peut pas gagner 100% du temps !

La Q-table apprise montre que l'agent a compris : descendre et aller à droite tout en évitant les trous.

---

## Exemples concrets

- **Voiture autonome** : Apprendre quelles voies prendre aux intersections par des essais successifs.
- **Systèmes de recommandation** : Apprendre quels films suggérer en fonction du fait que les utilisateurs ont aimé les suggestions précédentes.
- **IA de jeu vidéo** : Un personnage qui apprend à naviguer dans un labyrinthe en essayant de nombreux chemins.

---

## Mots-clés à retenir

- **Q-table** : Le tableau de "quelle est la qualité de chaque action dans chaque état"
- **Q(s, a)** : Le score pour avoir pris l'action a dans l'état s
- **Récompense (Reward)** : Ce que l'agent reçoit après avoir pris une action (+1 pour avoir atteint l'objectif, 0 sinon)
- **Off-policy** : Apprend la stratégie optimale même en explorant aléatoirement
- **ε-greedy** (ε = epsilon) : La plupart du temps, fait la meilleure action connue ; parfois explore aléatoirement
- **Facteur d'actualisation γ** (γ = gamma) : Valeur des récompenses futures (comme préférer l'argent maintenant plutôt que plus tard)

L'idée maîtresse : **Le Q-Learning construit un "antisèche" pour chaque situation, et continue de l'améliorer jusqu'à ce qu'il connaisse le meilleur mouvement partout.**
