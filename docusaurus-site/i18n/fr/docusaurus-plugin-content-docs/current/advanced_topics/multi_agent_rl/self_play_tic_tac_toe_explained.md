# Self-Play (Auto-apprentissage) : Enseigner à un agent en le laissant jouer contre lui-même ♟️

## Qu'est-ce que le Self-Play ?

Imaginez un enfant qui veut devenir très bon aux échecs mais n'a personne avec qui jouer. Alors, elle joue contre elle-même. Main gauche contre main droite. À chaque partie, les *deux* côtés essaient de gagner. À chaque partie, les *deux* côtés apprennent ce qui a fonctionné.

C'est cela le **self-play** : un seul agent agit en tant que deux joueurs, et chaque mouvement devient une leçon pour celui qui joue ensuite. Pas de professeur, pas d'adversaire expert. Juste un apprenant qui est aussi sa propre échelle de progression.

Le self-play semble être une astuce — n'a-t-on pas besoin d'un véritable adversaire ? — mais c'est le moteur des jalons les plus célèbres du RL de la dernière décennie : **AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. Ils utilisent tous le self-play. La raison est simple : à mesure que vous vous améliorez, votre adversaire s'améliore d'autant. Le défi correspond toujours à votre niveau de compétence.

---

## Pourquoi ça marche

Trois choses rendent le self-play spécial :

1. **Des adversaires infinis.** Vous ne manquez jamais de parties. L'adversaire est toujours présent et gratuit.
2. **Un programme d'apprentissage qui grandit avec vous.** Un débutant ne peut jouer que contre d'autres débutants. À mesure que vous vous améliorez, votre ombre s'améliore aussi — automatiquement.
3. **Symétrie.** Dans un jeu à somme nulle (la victoire d'un joueur est la perte de l'autre), un seul ensemble de valeurs Q décrit les deux côtés ; il suffit d'inverser le signe lorsque c'est au tour de l'autre joueur. Ainsi, une *seule* table Q peut s'auto-enseigner.

Le morpion (Tic-tac-toe) est le banc d'essai parfait : assez petit pour tenir dans un dictionnaire, mais assez complexe pour que le choix aléatoire des coups mène presque toujours à une défaite contre un joueur stratégique.

---

## Une analogie de la vie réelle

- **Pratiquer le tennis contre un mur.** Vous ne pouvez pas perdre contre un mur, mais vous pouvez pratiquer vos services. Le self-play consiste à faire cela des deux côtés — vous êtes le mur *et* le joueur, et vous passez de l'un à l'autre.
- **Un club de débat qui argumente les deux côtés.** Les meilleurs débatteurs émergent en défendant toujours le point de vue opposé à leurs convictions personnelles. Chaque argument entraîne à la fois l'attaque et la défense.
- **AlphaGo Zero.** Il a appris à partir de zéro partie humaine. En partant de mouvements aléatoires, il a joué des millions de parties contre lui-même ; en quelques jours, il était meilleur que tous les programmes de Go précédents, y compris celui qui a battu Lee Sedol.

---

## Ce que fait notre code

Nous apprenons une table Q pour le *joueur dont c'est le tour* :

```
Q[(plateau, joueur_dont_c'est_le_tour)][action] = rendement attendu pour ce joueur
```

La boucle d'entraînement est la suivante :

1. Commencer avec un plateau vide. `joueur = X`.
2. Les deux joueurs agissent avec le **même agent**, en utilisant ε-greedy.
3. Après chaque partie, revenir en arrière à travers chaque triplet (plateau, joueur, action) de l'historique et appliquer la mise à jour Q-learning.
4. La récompense change de signe selon les tours : si X gagne, chaque mouvement effectué par X reçoit +1 (ou propage la valeur d'un futur état gagnant) ; chaque mouvement effectué par O reçoit -1.
5. Nous diminuons lentement (décroissance) notre taux d'exploration (ε) de 0,2 → 0,02, de sorte que l'agent s'en tienne à son meilleur jeu à la fin de l'entraînement au lieu d'essayer des mouvements aléatoires.

Tous les 2 500 épisodes, nous évaluons l'agent contre un **adversaire aléatoire** (nous gelons le processus d'apprentissage afin qu'aucune nouvelle mise à jour ne soit effectuée sur la table Q pendant l'évaluation, et les deux côtés jouent de manière gourmande/greedy). L'agent devrait gagner ou faire match nul dans environ 100 % de ces parties après suffisamment de self-play.

### Ce que vous devriez voir

Après 50 000 épisodes de self-play :

| Confrontation | Résultat attendu |
|---------------|-------------------|
| Agent entraîné vs Adversaire aléatoire (1000 parties) | **~95-99% de victoires ou nuls**, pratiquement 0% de défaites |
| Agent entraîné vs Lui-même (200 parties gourmandes) | **Les 200 sont des nuls**. Le morpion est un jeu qui se termine toujours par un match nul si les deux joueurs jouent parfaitement. Le fait que le self-play aboutisse à des nuls à chaque partie est un signe de convergence. |

Le graphique `outputs/self_play_tic_tac_toe.png` montre les fractions de victoire/nul/défaite de l'agent contre un adversaire aléatoire au fil du temps :
- Le taux de victoire commence à environ 60 % (lorsque les deux joueurs jouent au hasard, le premier joueur a un avantage inhérent car il peut placer plus de pions sur le plateau, ce qui conduit à un taux de victoire de base d'environ 60 % pour le joueur X).
- Il grimpe à plus de 90 %.
- Le taux de défaite tombe à presque 0 %.

Le script affiche également un exemple de partie coup par coup à la fin pour que vous puissiez voir l'agent jouer.

---

## Attention à ces subtilités

- **L'inversion des signes est importante.** Un bug courant : oublier que "l'adversaire maximisant sa valeur" signifie *minimiser la nôtre* dans la cible de propagation (bootstrap). La mise à jour dans notre code utilise `cible = recompense - gamma * max(Q[suivant, adversaire])`.
- **La symétrie n'est pas exploitée ici.** Une véritable mise en œuvre standardiserait (canoniserait) les plateaux (ce qui signifie qu'elle ferait pivoter ou refléterait n'importe quel état du plateau dans une "forme normale" standard et unique afin que l'agent reconnaisse les situations de plateau identiques) pour partager les valeurs Q entre 8 symétries. Nous sautons cette étape — l'espace d'états est assez petit pour être traité par force brute.
- **La table Q s'agrandit.** Après 50 000 parties de self-play, vous verrez quelques milliers de clés état-joueur. C'est tout à fait acceptable ici ; pour les échecs ou le Go, vous auriez besoin d'un réseau de neurones à la place, c'est pourquoi **AlphaZero remplace la table par un CNN + MCTS**.

---

## Où le self-play échoue

- **Jeux qui ne sont pas à somme nulle.** "Les deux côtés sont contents" est incompatible avec un jeu symétrique ; vous ne pouvez pas simplement inverser un signe.
- **Rôles asymétriques.** Si l' "attaquant" et le "défenseur" ont des espaces d'action différents, vous avez besoin de deux réseaux distincts.
- **Cyclage de stratégie.** Le pur self-play peut s'enfermer dans des cycles de type pierre-feuille-ciseaux. AlphaStar a résolu ce problème en conservant une grande *réserve* (ou "ligue") de versions passées enregistrées de l'agent et en choisissant des adversaires dans cette réserve au hasard, de sorte que l'agent apprenne à battre de nombreux styles de jeu différents plutôt que seulement le style actuel.
- **Piratage de récompense (Reward hacking).** Le self-play rend les deux côtés plus intelligents, mais seulement pour le jeu *tel que vous l'avez défini*. Si votre système de récompense contient des failles involontaires (comme récompenser un joueur simplement pour avoir survécu plus longtemps au lieu de gagner), les deux côtés exploiteront mutuellement la faille, menant à un comportement bizarre et inutile au lieu de maîtriser le véritable jeu.

---

## Mots-clés à retenir

| Mot | Signification |
|------|---------|
| **Self-play** | Le même agent joue les deux côtés d'un jeu. |
| **Somme nulle** | Le gain d'un joueur = la perte de l'autre. |
| **Symétrie** | Une table Q peut servir aux deux côtés si vous inversez les signes. |
| **Population play** | Self-play avec de *nombreuses* versions passées de soi-même comme adversaires (AlphaStar). |
| **Curriculum** | Une progression naturelle de la difficulté — le self-play l'obtient gratuitement. |
| **MCTS** | Monte-Carlo Tree Search — l'algorithme de planification qu'AlphaZero associe au self-play. |

---

## Résumé en une phrase

> **Le self-play transforme l'amélioration en sa propre échelle : chaque fois que vous devenez meilleur, votre adversaire l'est aussi — automatiquement.**

Cette idée, mise à l'échelle avec des **réseaux de neurones** (fonctions mathématiques inspirées du cerveau qui apprennent des modèles à partir de données) et la recherche arborescente (tree search), a battu les meilleurs humains au Go, aux échecs, au shogi, à Dota 2 et à StarCraft.
