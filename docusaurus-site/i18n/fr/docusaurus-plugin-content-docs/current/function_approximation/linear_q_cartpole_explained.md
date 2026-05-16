# Q-Learning Linéaire pour CartPole 🎪

## Qu'est-ce que CartPole ?

Imaginez un balai en équilibre vertical sur votre doigt. Si vous déplacez votre doigt légèrement vers la gauche ou la droite, vous pouvez empêcher le balai de tomber. C'est le **CartPole** !

Un petit robot est assis sur un chariot (une boîte sur roues) et possède une perche fixée sur le dessus. Le robot ne peut pousser le chariot que vers la **gauche** ou la **droite**. Il doit apprendre à maintenir cette perche en équilibre le plus longtemps possible — tout comme vous avec votre balai !

Le robot peut observer 4 éléments du monde :
1. La position du chariot
2. La vitesse du chariot
3. L'angle d'inclinaison de la perche
4. La vitesse de rotation de la perche

---

## Le Problème : Trop d'États !

Vous souvenez-vous du Q-learning de la Phase 2 ? Il utilisait un grand tableau pour mémoriser la qualité de chaque action dans chaque situation (état). Cela fonctionnait très bien pour Frozen Lake — il n'y avait que 16 cases sur la glace.

Mais CartPole est différent ! Le chariot peut être à **n'importe quelle position**, se déplaçant à **n'importe quelle vitesse**, avec la perche à **n'importe quel angle**. Il y a pratiquement une **infinité d'états possibles** ! Nous ne pouvons pas créer un tableau avec une infinité de lignes. Il nous faudrait un carnet de la taille de l'univers !

**Exemple concret :** Imaginez que vous apprenez à faire du vélo. Vous ne pouvez pas mémoriser chaque oscillation possible — il y en a trop ! Au lieu de cela, vous apprenez une **règle** : "quand je penche à gauche, je pousse à droite ; quand je penche à droite, je pousse à gauche." Une règle simple fonctionne pour TOUTES les oscillations.

---

## La Solution : Une Formule Magique

L'**approximation de fonction linéaire** remplace le tableau géant par une **formule minuscule** :

> **Score(situation, action) = w₁ × position_chariot + w₂ × vitesse_chariot + w₃ × angle_perche + w₄ × vitesse_perche**

- Les nombres `w` sont appelés des **poids** — ils sont comme des boutons que l'on peut tourner.
- Nous apprenons des **poids différents pour chaque action** (pousser à gauche et pousser à droite).
- La formule donne un score indiquant la qualité de chaque action à l'instant présent.

**Exemple concret :** Pensez à une recette simple : "1 tasse de farine + 2 œufs + ½ tasse de beurre." Les poids (1, 2, ½) indiquent l'importance de chaque ingrédient. Nous apprenons la recette des bonnes décisions !

---

## Comment Apprend-il ?

Le robot essaie des choses, reçoit des commentaires et ajuste les poids :

1. **Le robot pousse le chariot** (choisit l'action avec le score le plus élevé).
2. **La physique opère** (la perche s'incline un peu, le chariot se déplace).
3. **Le robot reçoit une récompense** (+1 pour chaque étape où la perche reste debout, 0 si elle tombe).
4. **Le robot demande :** "Le résultat réel était-il meilleur ou pire que ce que j'avais prédit ?"
5. **Le robot ajuste les poids** pour se rapprocher de la réalité la prochaine fois.

C'est ce qu'on appelle la **Mise à jour TD par Semi-Gradient** — un nom savant pour dire "ajuster un peu la recette en fonction de la surprise".

> **Nouveau poids = Ancien poids + Taux d'apprentissage × (Ce qui s'est réellement passé − Ce que j'ai prédit) × Caractéristique**

---

## Ce que Notre Code a Révélé

Lorsque vous lancez `linear_q_cartpole.py`, le robot :

- Commence par être très mauvais (la perche tombe en 10 à 30 étapes).
- Apprend progressivement de bons poids au fil de 3 000 essais.
- finit par maintenir la perche en équilibre pendant plus de 100 à 400 étapes !

Le graphique montre la **courbe d'apprentissage** — comment le score s'améliore avec le temps. Elle sera irrégulière (l'apprentissage n'est jamais fluide !), mais la tendance devrait être à la hausse.

---

## Pourquoi c'est Génial (et Limité !)

**Génial :** Une formule minuscule avec seulement 8 nombres (4 poids × 2 actions) peut équilibrer une perche ! Pas besoin de tableau géant.

**Limité :** La formule est trop simple pour des tâches complexes. Elle suppose que des nombres plus grands signifient toujours des effets plus importants (ce qui n'est pas toujours vrai). Pour des jeux plus difficiles comme Atari, nous avons besoin de **réseaux de neurones** — ce que fait le DQN !

---

## Vocabulaire Clé

| Mot | Signification |
|------|---------|
| **Caractéristique (Feature)** | Une chose mesurable sur le monde (ex: angle de la perche) |
| **Poids (Weight)** | L'importance d'une caractéristique dans la décision |
| **Linéaire** | La formule n'est faite que de multiplications et d'additions (pas de courbes complexes) |
| **Semi-gradient** | Mise à jour des poids en suivant la direction de l'erreur moindre |
| **Approximation de fonction** | Utiliser une formule au lieu d'un tableau |

---

## Quelle est la Suite ?

L'approximation linéaire est comme utiliser une règle droite pour dessiner une courbe — cela fonctionne bien pour les formes simples mais pas pour les complexes. Pour les jeux Atari avec des millions de situations possibles, nous avons besoin de **Deep Q-Networks (DQN)** — des réseaux de neurones capables d'apprendre des motifs beaucoup plus complexes. C'est ce que nous verrons dans le prochain fichier !
