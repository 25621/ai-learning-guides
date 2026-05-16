# Architecture Option-Critic

## L'Idée Maîtresse : Travailler par Chapitres, pas Mot à Mot

Imaginez que vous écrivez un roman. Vous ne planifiez pas chaque mot individuel avant de commencer. Au lieu de cela, vous réfléchissez en **chapitres** : "Le chapitre 1 présente le héros. Le chapitre 2 est la quête. Le chapitre 3 est la confrontation finale." À l'intérieur de chaque chapitre, vous déterminez les détails au fur et à mesure.

C'est exactement ainsi que l'architecture Option-Critic envisage les décisions.

---

## Qu'est-ce qu'un Agent "Plat" ?

Un agent RL normal (comme ceux des phases 3 et 4 du programme) prend ses décisions une action à la fois, à chaque étape. C'est comme un GPS qui recalculerait tout l'itinéraire à partir de zéro chaque fois que vous avancez d'un mètre. Cela fonctionne, mais c'est épuisant et lent à apprendre.

---

## Qu'est-ce qu'une "Option" ?

Une **option** est une **compétence nommée** — une mini-politique que l'agent peut exécuter pendant plusieurs étapes consécutives avant de rendre le contrôle.

Pensez-y comme à un manager déléguant à des spécialistes :

| Qui | Ce qu'ils font |
|-----|-------------|
| **Gestionnaire (méta-politique)** | Décide *quel* spécialiste envoyer sur une mission |
| **Spécialiste A** | Expert pour naviguer dans la salle en haut à gauche |
| **Spécialiste B** | Expert pour franchir les portes |
| **Spécialiste C** | Expert pour foncer vers l'objectif |
| **Spécialiste D** | Généraliste de soutien |

Le gestionnaire choisit un spécialiste. Le spécialiste travaille jusqu'à ce qu'il décide qu'il a terminé (c'est ce qu'on appelle la **terminaison**). Ensuite, le gestionnaire choisit à nouveau.

---

## Les Trois Pièces Mobiles

Chaque option possède trois composants — considérez-les comme la **description de poste** du spécialiste :

1. **Initiation (Initiation)** : Quand ce spécialiste peut-il être sollicité ? *(ex : "Le Spécialiste A ne s'active qu'à proximité de la salle en haut à gauche.")*
2. **Politique intra-option (Intra-option policy)** : Que fait le spécialiste pendant qu'il travaille ? *(ex : "Marcher vers le coin supérieur gauche.")*
3. **Terminaison (Termination)** : Quand le spécialiste rend-il le contrôle ? *(ex : "S'arrêter une fois qu'une porte est atteinte.")*

La beauté de l'Option-Critic est que ces trois éléments sont **appris automatiquement** — vous ne concevez pas les spécialistes à la main. L'algorithme découvre de lui-même qu'il est utile d'avoir une option pour chaque pièce, ou une pour foncer vers l'objectif.

---

## Une Journée dans la Vie d'un Agent Option-Critic

1. L'agent entre dans une nouvelle pièce (état).
2. Le **Gestionnaire** observe la pièce et choisit une option — disons, l'Option 2.
3. Le **spécialiste de l'Option 2** prend le relais : il marche vers la porte, étape par étape.
4. À un moment donné, l'Option 2 dit "J'ai fini ici" (terminaison).
5. Le **Gestionnaire** se réveille et choisit une nouvelle option adaptée à la nouvelle situation.
6. Recommencer.

Comparez cela à l'agent plat : l'agent plat hésite sur chaque petit pas. L'Option-Critic délègue des pans entiers de comportement, laissant chaque spécialiste devenir excellent dans sa tâche spécifique.

---

## Pourquoi cela Aide-t-il ?

Dans un labyrinthe, l'agent doit atteindre un objectif qui peut se trouver à 30-50 pas. Avec un apprentissage plat, chaque pas sur le chemin est tout aussi "invisible" jusqu'à ce que la récompense arrive enfin à la fin — ce signal doit remonter à travers des dizaines d'étapes.

Avec les options, le chemin se divise en **sous-tâches**. Chaque sous-tâche reçoit son propre mini-signal de récompense (atteindre la porte, entrer dans la pièce suivante). L'apprentissage se propage par segments plus courts. **L'agent apprend plus vite sur des problèmes qui nécessitent de nombreuses étapes.**

C'est l'idée centrale de tout le RL Hiérarchique — et l'Option-Critic en est l'une des implémentations les plus élégantes.

---

## Ce que Fait Notre Code

Le script `option_critic.py` place un agent Option-Critic dans un **monde de grille de 7x7** avec un objectif fixe. L'agent commence n'importe où dans la grille et doit naviguer vers la cellule cible.

L'agent dispose de quatre options et doit apprendre simultanément :

- Une politique pour chaque option (où marcher)
- Quand terminer chaque option (condition de terminaison)
- Une méta-politique pour choisir entre les options

La récompense utilise une **mise en forme (shaping) basée sur le potentiel** — l'agent reçoit un petit bonus à chaque pas qu'il fait vers l'objectif, en plus du +1 pour l'avoir atteint. Ce feedback dense rend l'apprentissage suffisamment stable pour voir les options à l'œuvre en 2 500 épisodes.

Aucun humain ne lui dit jamais ce que chaque option doit faire. L'algorithme découvre par lui-même les zones de la grille dans lesquelles chaque option se spécialise.

---

## Ce que Montrent les Graphiques

![Courbes d'apprentissage Option-Critic](outputs/option_critic.png)

**À gauche — Retour mis en forme (Shaped Return) :** Un retour plus élevé signifie que l'agent atteint l'objectif de manière plus fiable *et* emprunte des chemins plus courts (le shaping donne un bonus par pas rapprochant de la cible). La courbe qui monte puis se stabilise montre que les options apprennent à se coordonner.

**À droite — Étapes vers l'objectif (Steps to Goal) :** Moins d'étapes signifient que l'agent a trouvé un chemin plus efficace. La tendance à la baisse montre que les options mûrissent pour devenir des compétences cohérentes qui guident l'agent plus directement vers l'objectif.

Les courbes lissées montrent la tendance générale sur des fenêtres de 100 épisodes — un certain bruit est normal en RL, surtout lorsque plusieurs composants (options, terminaison, méta-politique) apprennent simultanément.

---

## Résumé en une phrase

> **L'Option-Critic apprend à un agent à travailler par compétences plutôt que par étapes isolées — un gestionnaire choisit quel spécialiste s'exécute, chaque spécialiste fait son travail, et tout le système apprend ensemble à partir du même signal de récompense.**
