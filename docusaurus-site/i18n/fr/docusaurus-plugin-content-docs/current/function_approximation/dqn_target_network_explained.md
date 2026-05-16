# Réseau Cible : Stabiliser la cible 🎯

## Le problème de la cible mouvante

Imaginez que vous essayiez d'atteindre le centre d'une cible avec un arc et des flèches. Vous tirez, regardez où votre flèche a atterri et ajustez votre tir pour la prochaine fois. Simple, n'est-ce pas ?

Imaginez maintenant que la cible BOUGE chaque fois que vous tirez ! Chaque flèche que vous envoyez change légèrement l'endroit où se trouvera la cible pour le prochain tir. Vous ne vous amélioreriez jamais — vous seriez en train de courir après une cible qui s'échappe sans cesse.

C'est exactement le problème de DQN sans réseau cible !

---

## Pourquoi les cibles Q bougent-elles ?

Dans DQN, la cible pour chaque mise à jour est :
> target = reward + γ × max(Q(next_state))

Ici, **γ (gamma)** est le **facteur de remise** (discount factor) — un nombre entre 0 et 1 (généralement 0,99) qui contrôle l'importance que l'agent accorde aux récompenses *futures* par rapport aux récompenses *immédiates*.

**Exemple de la vie réelle :** Imaginez que quelqu'un vous propose un biscuit maintenant, ou deux biscuits demain. Si vous voulez vraiment des biscuits tout de suite, votre γ est bas (vous dévalorisez fortement le futur). Si vous êtes patient et heureux d'attendre, votre γ est élevé (les récompenses futures comptent presque autant que celles de maintenant). En RL, γ = 0,99 signifie "une récompense à l'étape suivante vaut 99 % d'une récompense immédiate".

Les valeurs Q sur le côté droit proviennent... du même réseau que nous sommes en train d'entraîner !

Ainsi, chaque fois que nous mettons à jour le réseau (pour améliorer les valeurs Q), nous changeons également les cibles. C'est une boucle de rétroaction :

1. Mise à jour du réseau → les valeurs Q changent.
2. Les valeurs Q changent → les cibles changent.
3. Les cibles changent → mise à jour du réseau différemment.
4. On recommence indéfiniment — instabilité !

**Exemple de la vie réelle :** Essayer de se peser sur une balance qui change ses mesures chaque fois que vous montez dessus. Vous ne connaîtriez jamais votre poids réel !

---

## La solution : Figurer la cible ! ❄️

Le **Réseau Cible (Target Network)** est une COPIE du réseau Q principal qui est figée.

- **Réseau en ligne** (`qnet`) : Mis à jour à chaque étape d'entraînement — apprend rapidement.
- **Réseau cible** (`target_net`) : Copie figée — mise à jour seulement toutes les 100 étapes.

Nous utilisons le réseau cible FIGÉ pour calculer les cibles :
> target = reward + γ × max(Q_TARGET(next_state))

La cible reste immobile pendant 100 étapes ! Cela donne au réseau en ligne un objectif stable à viser. Ensuite, nous copions les poids du réseau en ligne vers le réseau cible, nous le figeons à nouveau, et nous recommençons.

**Exemple de la vie réelle :** Pensez à un élève et à un professeur. Le professeur donne des devoirs (la cible). L'élève apprend et s'améliore. Après 100 leçons, le professeur MET À JOUR les devoirs pour qu'ils soient plus difficiles. Le professeur ne change pas toutes les minutes — ce serait trop chaotique !

---

## La recette complète du DQN 🍕

L'algorithme DQN complet (experience replay + réseau cible) est le suivant :

```
1. Initialiser le réseau en ligne Q et le réseau cible Q_target (mêmes poids).
2. Créer le replay buffer (boîte à souvenirs).

À chaque étape dans l'environnement :
  a. Choisir une action en utilisant ε-greedy avec Q.
  b. Stocker (état, action, récompense, état_suivant) dans le buffer.

Toutes les 4 étapes :
  c. Piocher un mini-lot aléatoire dans le buffer.
  d. Calculer les cibles en utilisant Q_TARGET (figé !).
  e. Mettre à jour Q pour minimiser la perte.

Toutes les 100 étapes :
  f. Copier les poids de Q → Q_TARGET (synchronisation de la cible).
```

C'est exactement l'algorithme décrit dans l'article sur le DQN de DeepMind (2015) !

---

## Ce que montre la comparaison

Lorsque vous lancez `dqn_target_network.py`, vous verrez :

**Sans réseau cible (DQN + replay uniquement) :**
- L'entraînement peut être "correct" mais avec des rechutes périodiques.
- Les valeurs Q peuvent diverger (exploser ou osciller).
- L'apprentissage est moins prévisible.

**DQN complet (replay + réseau cible) :**
- Progression de l'apprentissage plus constante vers le haut.
- Les valeurs Q restent dans une fourchette raisonnable.
- Convergence plus rapide vers le seuil de résolution (195+ sur CartPole).

---

## La "Triade Mortelle" ☠️

En apprentissage par renforcement, la combinaison de trois éléments crée de l'instabilité :

1. **Approximation de fonction** (réseau de neurones au lieu d'une table) ← nous l'utilisons.
2. **Bootstrapping** (utiliser des valeurs Q pour estimer des valeurs Q) ← nous l'utilisons.
3. **Apprentissage hors politique (Off-policy learning)** (Q-learning utilise le max, pas la politique réelle) ← nous l'utilisons.

Tous les trois ensemble forment la "triade mortelle" (deadly triad). DQN dompte cela grâce à :
- L'experience replay → casse les corrélations.
- Le réseau cible → casse la boucle de rétroaction.

Cela ne résout pas entièrement le problème, mais le rend gérable !

---

## Vocabulaire clé

| Mot | Signification |
|-----|---------------|
| **Réseau Cible (Target Network)** | Une copie figée du réseau Q utilisée uniquement pour calculer les cibles |
| **Réseau en ligne (Online Network)** | Le réseau Q activement entraîné |
| **Synchronisation (Sync)** | Copier les poids du réseau en ligne vers le réseau cible |
| **Boucle de rétroaction** | Lorsque la sortie d'un système modifie son propre signal d'entrée (peut causer de l'instabilité) |
| **Triade Mortelle** | La combinaison de l'approximation de fonction, du bootstrapping et du off-policy qui cause l'instabilité |

---

## Impact dans le monde réel

En 2015, DeepMind a publié son article sur le DQN montrant une IA capable de jouer à 49 jeux Atari à un niveau surhumain — en utilisant JUSTE ces deux astuces (replay + réseau cible).

Avant cela, on pensait qu'il était impossible d'entraîner des réseaux de neurones avec le RL à cause de l'instabilité. DeepMind a prouvé le contraire et a lancé la révolution du Deep RL !

Ensuite, nous appliquerons cette recette complète du DQN à Atari Pong — un vrai jeu vidéo avec des pixels bruts comme entrée !
