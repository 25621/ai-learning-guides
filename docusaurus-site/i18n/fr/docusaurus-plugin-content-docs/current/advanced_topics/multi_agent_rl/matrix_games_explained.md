# Jeux de Matrice : Le Monde Multi-Agent le Plus Simple 🎲

## Qu'est-ce qu'un Jeu de Matrice ?

Imaginez que vous et un ami choisissiez chacun un signe de la main — **pierre, feuille ou ciseaux** — *en même temps*. Vous ne voyez pas le choix de l'autre. Le gagnant est déterminé par un petit tableau :

|        | Pierre | Feuille | Ciseaux |
|--------|:----:|:-----:|:--------:|
| Pierre   |  0,0  | -1,+1 | +1,-1 |
| Feuille  | +1,-1 |  0,0  | -1,+1 |
| Ciseaux  | -1,+1 | +1,-1 |  0,0  |

Ce tableau constitue le *monde entier* du jeu. Pas de mouvement, pas de temps, pas de carte. Juste une décision ponctuelle. Nous appelons cela un **jeu de matrice** car la matrice des gains (payoffs) représente l'environnement complet.

Les jeux de matrice sont l'endroit le plus pur pour étudier le **RL multi-agent**, car la seule chose qui peut changer pendant l'entraînement est la *politique* de chaque joueur — la probabilité de choisir chaque action.

---

## Pourquoi c'est "Multi-Agent"

Dans le RL mono-agent, l'environnement est fixe : le vent souffle toujours de la même manière, le sol ne bouge jamais. L'agent s'améliore et finit par gagner.

Dans un jeu de matrice, votre "environnement" est *un autre agent apprenant*. À mesure qu'il devient plus intelligent, ce qui est considéré comme un bon coup pour vous *change*. C'est ce qu'on appelle la **non-stationnarité**, et c'est le problème central du RL multi-agent.

> Si vous jouez toujours Pierre, votre adversaire finira par toujours jouer Feuille. Vous passez donc aux Ciseaux. Il passe donc à la Pierre. Vous passez donc à la Feuille... et ainsi de suite. Le "meilleur coup" ne reste jamais en place.

La solution classique est celle des **stratégies mixtes** : ne choisissez pas une action de manière déterministe — introduisez du hasard de sorte que l'adversaire ne puisse pas vous exploiter.

---

## Les Trois Jeux auxquels nous Jouons

### 1) Pierre-Feuille-Ciseaux (somme nulle)
- Le gain d'un joueur est la perte de l'autre.
- L'**équilibre de Nash** est : chaque joueur choisit chaque action avec une probabilité de ⅓. Toute déviation est exploitable.
- Nous nous attendons à ce que nos deux apprenants Q oscillent autour de ⅓-⅓-⅓ — jamais parfaitement stables, car chaque fois que l'un dérive, l'autre réagit.

### 2) Dilemme du Prisonnier (somme générale)
Deux suspects sont interrogés séparément :

|           | Coopérer | Trahir |
|-----------|:---------:|:------:|
| Coopérer |   3, 3    |  0, 5  |
| Trahir   |   5, 0    |  1, 1  |

- "Trahir" bat "Coopérer" peu importe ce que fait l'autre — c'est une **stratégie dominante**.
- Les deux joueurs sont rationnels → les deux trahissent → les deux reçoivent 1, même si (Coopérer, Coopérer) rapportait 3 chacun. La meilleure réponse égoïste détruit le bien-être du groupe.
- Nous attendons du Q-learning qu'il converge proprement vers (Trahir, Trahir).

### 3) Chasse au Cerf (Stag Hunt - coordination)
Deux chasseurs peuvent soit abattre un cerf ensemble (grosse récompense), soit se contenter chacun d'un lièvre (petite récompense mais sûre) :

|       | Cerf | Lièvre |
|-------|:----:|:----:|
| Cerf  | 4, 4 | 0, 3 |
| Lièvre| 3, 0 | 2, 2 |

- (Cerf, Cerf) est **dominant en termes de gain** — le meilleur pour les deux.
- (Lièvre, Lièvre) est **dominant en termes de risque** — sûr si vous ne faites pas confiance à votre partenaire.
- Le résultat dépend des conditions initiales : des apprenants Q indépendants finissent souvent dans le *pire* équilibre (Lièvre, Lièvre) car les lièvres sont plus sûrs à apprendre.

---

## Exemples Concrets

- **Fixation des prix dans un duopole.** Deux cafés dans la même rue choisissent chacun un prix chaque matin. La forme de la matrice des gains décide s'ils finissent par un prix "coopératif" élevé (bon pour eux, mauvais pour les clients) ou un prix bas et agressif.
- **Protocoles réseau.** Les routeurs et les émetteurs choisissent des stratégies de timing ; le résultat de la congestion du réseau est déterminé par le gain de type jeu de matrice (passer vs reculer).
- **Enchères.** Chaque enchérisseur choisit une offre sans connaître celles des autres ; les gains dépendent de l'ensemble du vecteur. L'équilibre de Nash est une *stratégie d'enchère*, pas un simple nombre.

---

## Ce que Fait Notre Code

Pour chaque jeu, nous :
1. Créons deux apprenants Q sans état (Q est juste un nombre par action — il n'y a pas d'états dans un jeu ponctuel).
2. Bouclons pendant 20 000 étapes. À chaque étape : les deux agents choisissent simultanément une action ε-greedy, reçoivent une récompense, et mettent à jour leurs valeurs Q.
3. Suivons la **fréquence empirique des actions** de chaque agent sur une fenêtre glissante de 500 étapes. Au lieu de regarder des probabilités abstraites, nous comptons les actions qu'ils ont réellement choisies récemment (ex: "dans les 500 derniers tours, ils ont joué Pierre 40% du temps"). Cela nous donne une image pratique et en temps réel de l'évolution de leur stratégie.
4. Traçons les fréquences au fil du temps, enregistrons dans `outputs/<jeu>.png`, et affichons les valeurs Q finales.

### Ce que vous devriez voir

| Jeu | Résultat attendu du graphique |
|------|------------------------------|
| **Pierre-Feuille-Ciseaux** | Les deux joueurs gravitent autour de ⅓-⅓-⅓ mais avec des oscillations visibles. Les courbes se poursuivent — comportement cyclique classique. |
| **Dilemme du Prisonnier** | La fréquence "Trahir" des deux joueurs grimpe rapidement vers ~1.0. "Coopérer" est écrasé. |
| **Chasse au Cerf** | La plupart des graines aléatoires (seeds) s'installent sur (Lièvre, Lièvre). Quelques graines chanceuses atteignent (Cerf, Cerf) — essayez de changer la seed dans le script pour voir le basculement. |

---

## Où l'Apprentissage Indépendant Échoue

Nos agents sont *indépendants* — ils ne voient que leur propre récompense, jamais l'action ou les valeurs Q de l'adversaire. C'est la base la plus simple et elle a des limites :

- Elle **ne peut pas garantir la convergence** dans les jeux à somme générale.
- Elle peut rester bloquée dans de **mauvais équilibres** (Chasse au Cerf).
- Elle **ne peut pas modéliser l'adversaire**.

Les algorithmes multi-agents réels corrigent cela en raisonnant explicitement sur l'autre apprenant. Voici ce que chacun fait, en termes simples :

| Algorithme | Idée centrale | Analogie concrète |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Fictitious play** | Tient un décompte de la fréquence à laquelle votre adversaire a choisi chaque action. Suppose que demain il fera ce qu'il a toujours fait — puis choisit votre propre meilleure réponse à cette croyance. | Observer les habitudes d'un adversaire sur plusieurs parties d'échecs et ajuster votre ouverture en conséquence. |
| **CFR (Counterfactual Regret Minimisation)** | Après chaque tour, demande *"À quel point ai-je regretté de ne pas avoir choisi chaque autre action ?"* Déplace progressivement la probabilité vers les actions que vous regrettez d'avoir manquées. Utilisé au poker car il gère les jeux à **information imparfaite** (vous ne voyez pas les cartes de l'adversaire). | Après une main de poker, la rejouer en pensant : *"J'aurais dû miser plus — je le ferai la prochaine fois."* |
| **LOLA (Learning with Opponent-Learning Awareness)** | Votre étape de gradient prend en compte le fait que l'adversaire fait *aussi* une étape de gradient. Vous optimisez votre propre mise à jour tout en anticipant la prochaine mise à jour de l'adversaire — deux coups d'avance au lieu d'un. | Négocier un accord en pensant : *"Si je propose X, ils contre-attaqueront avec Y, donc je devrais commencer par Z."* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | Le *critique* de chaque agent (estimateur de valeur) est entraîné avec la **vue globale** : il voit les observations et les actions de tout le monde. L'*acteur* (la politique déployée) n'utilise toujours que des informations locales — c'est le modèle CTDE (*Centralized Training with Decentralized Execution*). | Un entraîneur de basket qui surveille tout le terrain (critique centralisé) mais apprend à chaque joueur à ne réagir qu'à ce qu'il peut voir (acteur décentralisé). |

Mais le Q-learning indépendant est la bonne première étape. Vous voyez le problème de la non-stationnarité vous sauter aux yeux, et les corrections prennent tout leur sens par la suite.

---

## Mots Clés à Retenir

| Mot | Signification |
|------|---------|
| **Matrice des gains** | Le tableau qui définit un jeu multi-agent ponctuel |
| **Équilibre de Nash** | Un profil de politique où aucun agent ne peut s'améliorer en déviant seul |
| **Stratégie mixte** | Une politique qui tire au sort parmi plusieurs actions |
| **Non-stationnarité** | L'environnement (= les autres agents) change continuellement à mesure qu'il apprend |
| **Apprenant indépendant** | Un agent qui ignore l'existence d'autres apprenants |
| **Somme nulle** | Le gain d'un agent est exactement la perte de l'autre |
| **Somme générale** | Les deux agents peuvent gagner, perdre, ou n'importe quoi entre les deux |

---

## Résumé en une phrase

> **Dans les jeux de matrice, l'"environnement" est un autre apprenant — donc le meilleur coup ne cesse de bouger.**

C'est l'idée fondamentale derrière chaque algorithme multi-agent que vous rencontrerez plus tard, du self-play au MADDPG en passant par le MARL avec communication.
