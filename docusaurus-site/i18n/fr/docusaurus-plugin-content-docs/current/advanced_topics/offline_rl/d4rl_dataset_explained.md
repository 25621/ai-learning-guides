# Jeux de Données de Référence D4RL 📦

## Qu'est-ce que c'est ?

Imaginez que vous vouliez apprendre à un robot à faire sauter des crêpes. Le laisser s'entraîner sur une vraie cuisinière pendant un mois serait lent, dangereux et coûteux. Mais vous disposez de dix ans de vidéos de chefs faisant sauter des crêpes (certaines réussies, d'autres ratées, d'autres aléatoires). Pouvez-vous enseigner au robot à partir de *ces seules données*, sans jamais le laisser toucher une vraie poêle ?

C'est ce qu'on appelle l'**apprentissage par renforcement hors ligne (offline reinforcement learning)**. L'agent apprend à partir d'un ensemble de données fixe d'expériences passées — pas d'environnement en direct. La partie la plus difficile est que l'agent ne peut jamais *tester* ce qu'il a appris avant la toute fin.

Pour que l'étude de ce domaine soit équitable, la communauté des chercheurs avait besoin d'un *ensemble de données standard*. C'est là qu'intervient **D4RL** (**D**atasets for **D**eep **D**ata-**D**riven **R**einforcement **L**earning) : une collection de transitions pré-enregistrées pour des tâches de contrôle classiques, publiée par l'UC Berkeley en 2020. Chaque article scientifique s'entraîne sur les mêmes données, de sorte que les résultats sont comparables.

---

## Que contient un jeu de données D4RL ?

Pour chaque tâche, D4RL propose **quatre niveaux de qualité** :

| Niveau | Origine des données | Pourquoi c'est important |
|-------|---------------------------|----------------|
| **random** (aléatoire) | Une politique qui choisit des actions uniformément au hasard | Cas le plus défavorable : peut-on encore apprendre quelque chose d'utile ? |
| **medium** (moyen) | Une politique partiellement entraînée (environ la moitié du score expert) | Réaliste : la plupart des données enregistrées sont médiocres |
| **expert** | Une politique proche de la convergence | Cas idéal : peut-on égaler la politique source ? |
| **medium-replay** | L'intégralité du *tampon de rejeu (replay buffer)* utilisé pour entraîner la politique medium | Mixte : contient les échecs du début ET les succès ultérieurs |

La différence entre `medium` et `medium-replay` est cruciale :
- **`medium`** est généré en prenant une seule politique "moyenne" fixe et en la laissant jouer de nombreuses parties. Toutes les données reflètent ce niveau de compétence stable et moyen.
- **`medium-replay`** est un journal historique. Il contient toutes les expériences accumulées *pendant l'apprentissage* à partir de zéro jusqu'au niveau moyen. Il mélange des transitions **bonnes et mauvaises** — exactement ce à quoi ressemble un journal de données réel (les premières tentatives maladroites d'un robot *et* son comportement affiné plus tard, le tout dans le même panier).

---

## Exemples réels de jeux de données hors ligne

- **Dossiers médicaux.** Des années de triplets (état_patient, traitement, résultat). On ne peut pas randomiser les traitements sur des personnes vivantes, mais on peut apprendre une meilleure politique à partir des données historiques.
- **Journaux de clavardage (chat) du service client.** Des millions d'enregistrements (message_utilisateur, réponse_agent, satisfaction). Entraîner un meilleur assistant sans importuner davantage d'utilisateurs.
- **Données de flottes de conduite autonome.** Chaque voiture Tesla / Waymo télécharge ses trajets. La flotte constitue un gigantesque ensemble de données medium-replay.
- **Systèmes de recommandation.** Les journaux de clics de l'année dernière sont un ensemble de données figé : on ne peut pas ré-afficher les mêmes publicités aux mêmes utilisateurs.

Dans ces quatre cas, **vous ne pouvez pas demander à l'environnement un nouvel échantillon.** Le jeu de données est tout ce que vous avez. Pour toujours.

---

## Ce que fait notre code

Les véritables jeux de données D4RL sont enregistrés sur des tâches de locomotion MuJoCo (Multi-Joint dynamics with Contact) (comme HalfCheetah, Hopper, Walker2d, Ant — ce sont des simulations physiques 3D avancées où des robots virtuels apprennent à marcher et à courir). MuJoCo étant complexe à installer, nous recréons la **même structure à quatre niveaux sur CartPole-v1** — l'environnement standard pour débutants des phases précédentes. Les leçons apprises sont directement transposables.

Le script `d4rl_dataset.py` :

1. **Entraîne un DQN** (Deep Q-Network, un algorithme de RL standard) sur CartPole jusqu'à ce qu'il résolve la tâche (score ≥ 475).
2. **Prend deux instantanés (checkpoints)** en cours de route :
   - "medium" — au moment où le score récent a franchi 150
   - "expert" — au moment où le score récent a franchi 475
3. **Sauvegarde l'intégralité du tampon de rejeu (replay buffer) de la politique medium** — chaque transition qu'elle a vue. C'est notre jeu de données "medium-replay".
4. **Exécute trois nouvelles politiques** pour 10 000 transitions chacune :
   - `random`   — aléatoire uniforme
   - `medium`   — le checkpoint medium + un bruit ε=0,10
   - `expert`   — le checkpoint expert + un bruit ε=0,02
5. **Enregistre quatre fichiers `.npz`** (format de tableau compressé de NumPy) dans `outputs/`, contenant chacun les tableaux `obs / action / reward / next_obs / terminal`.

Ces quatre fichiers servent d'entrées aux scripts `cql.py` et `behavioral_cloning.py`.

---

## Ce que vous devriez voir lors de l'exécution

Un résumé textuel s'affiche dans la console et est enregistré dans `outputs/d4rl_summary.txt` :

```
dataset         |   N    |  score moyen  |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

Il génère également un histogramme (`outputs/d4rl_returns.png`) montrant comment les quatre jeux de données se chevauchent. Points clés à noter :

- **Random** se regroupe autour de 20 (la longueur moyenne d'un épisode aléatoire de CartPole).
- **Expert** se regroupe au plafond de 500.
- **Medium** se situe entre les deux, avec une grande variance.
- **Medium-replay** possède une longue "queue" vers la droite — il est principalement composé d'échecs précoces (faibles scores) mais s'étend vers des scores plus élevés à mesure que l'agent apprenait.

---

## Pourquoi le jeu de données est important

Quel que soit le jeu de données sur lequel vous entraînez votre algorithme hors ligne, vous fixez un *plafond* à ce qui est possible :

- **À partir d' `expert`** — même un algorithme simple comme le BC (Behavioral Cloning - Clonage Comportemental, qui se contente de copier exactement les données) peut réussir, car toutes les données sont de bonne qualité.
- **À partir de `random`** — vous avez besoin d'un algorithme intelligent capable d' *assembler (stitch together)* les rares bonnes transitions (trouver un chemin vers le succès en combinant de courtes séquences d'actions réussies provenant de différentes tentatives). Le BC échouera complètement.
- **À partir de `medium-replay`** — le cas le plus réaliste et le plus intéressant. Les bons algorithmes (comme **CQL** — Conservative Q-Learning, qui évite d'être trop confiant vis-à-vis des actions qu'il n'a jamais vues) peuvent parfois **dépasser la qualité moyenne des données** car ils extraient une structure à partir de signaux mixtes. Les algorithmes simples (BC) régressent vers la moyenne.

Nous verrons exactement cela dans les deux scripts suivants.

---

## Mots-clés à retenir

| Mot | Signification |
|------|---------|
| **Offline RL** (RL hors ligne) | S'entraîner à partir d'un jeu de données fixe ; aucune interaction avec l'environnement n'est autorisée |
| **Behaviour policy** (Politique de comportement) | La politique qui a *produit* le jeu de données |
| **Dataset quality** (Qualité des données) | La qualité de la politique de comportement (aléatoire / moyen / expert) |
| **Replay buffer** (Tampon de rejeu) | L'historique complet des transitions vues pendant un entraînement |
| **Distribution shift** (Décalage de distribution) | L'écart entre les actions présentes dans le jeu de données et celles que votre politique entraînée veut entreprendre. Comme le jeu de données ne montre jamais ce qui se passe quand la nouvelle politique tente quelque chose qui n'a pas été enregistré, les estimations de valeur de l'algorithme pour ces nouvelles actions peuvent être dangereusement erronées. |

---

## Résumé en une phrase

> **D4RL transforme le RL en un benchmark de type apprentissage supervisé : les mêmes données pour tout le monde, pas de triche possible avec l'environnement, et que le meilleur algorithme gagne.**
