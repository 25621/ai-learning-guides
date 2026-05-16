# Double DQN : Corriger le problème de surconfiance 🤔

## Le problème : DQN pense qu'il est meilleur qu'il ne l'est réellement

Imaginez qu'on vous demande : "Quel est le meilleur restaurant de la ville ?"

Vous pourriez répondre : "Pizza Palace est incroyable — c'est définitivement un 10/10 !" Mais vous n'y êtes allé que deux fois. Vous ne savez pas vraiment s'il s'agit *réellement* d'un 10/10. Vous surestimez peut-être parce que vous avez eu de la chance avec de bonnes pizzas lors de ces deux visites.

Ce même problème se produit avec DQN : l'agent **surestime les valeurs Q**.

---

## Pourquoi DQN surestime-t-il ?

Lorsque DQN calcule la cible (target), il fait :
> target = reward + γ × **max** Q(next_state)

C'est le `max` qui pose problème ! Lorsque vous choisissez le maximum parmi plusieurs estimations bruitées, vous choisissez presque toujours celle qui présente la plus grande erreur aléatoire (biais vers le haut).

**Exemple de la vie réelle :** Vous demandez à 5 amis de deviner la hauteur d'un bâtiment. Leurs estimations sont : 40m, 38m, 45m (coup de chance !), 39m, 41m. La hauteur réelle est de 40m.
Si vous utilisez `max(estimations)` = 45m, vous êtes loin de la réalité ! Le maximum d'estimations bruitées est presque toujours une surestimation.

Au fil de milliers de mises à jour, DQN continue de s'entraîner vers ces cibles surévaluées, apprenant que les choses sont meilleures qu'elles ne le sont en réalité. Cela peut ralentir l'apprentissage ou amener l'agent à prendre de mauvaises décisions par excès de confiance.

---

## La solution Double DQN

**Double DQN** (Hasselt et al., 2016) divise le `max` en deux étapes :

**Étape 1 — Quelle action ?** Utiliser le **réseau en ligne** (online network) pour choisir la meilleure action :
> best_action = argmax Q_online(next_state)

**Étape 2 — Quelle est sa valeur ?** Utiliser le **réseau cible** (target network) pour évaluer cette action :
> target = reward + γ × Q_target(next_state, best_action)

```
DQN classique :   target = r + γ × max_a Q_target(s', a)
                                   ↑ le même réseau choisit ET évalue → biaisé

Double DQN :      best_a = argmax_a Q_online(s', a)     ← le réseau en ligne choisit
                  target = r + γ × Q_target(s', best_a) ← le réseau cible évalue
                                   ↑ réseaux différents → moins biaisé
```

**Exemple de la vie réelle :** Lors d'un entretien d'embauche, vous ne laissez pas le candidat noter son propre test de performance (c'est le problème du DQN classique !). Au lieu de cela, le candidat *présente* son meilleur travail, et un examinateur *distinct* l'évalue.
Deux personnes différentes = évaluation plus juste !

---

## Pourquoi la séparation aide-t-elle ?

Les deux réseaux (en ligne et cible) ont des poids différents car le réseau cible est mis à jour moins fréquemment. Ils ont des "opinions" différentes sur l'action qui est la meilleure.

Lorsqu'ils ne sont pas d'accord :
- Le réseau en ligne dit : "L'action A a l'air géniale !"
- Le réseau cible dit : "En fait, l'action A est juste correcte — elle vaut environ 7, pas 10."

En utilisant l'estimation de VALEUR du réseau cible pour l'action CHOISIE par le réseau en ligne, nous obtenons un chiffre plus honnête et moins gonflé.

---

## Différence de code : une seule ligne !

Le seul changement de code entre le DQN classique et le Double DQN se situe dans le calcul de la cible :

```python
# DQN classique :
q_next = target_net(s_next).max(dim=1).values

# Double DQN :
best_actions = online_net(s_next).argmax(dim=1, keepdim=True)   # choix avec le réseau en ligne
q_next = target_net(s_next).gather(1, best_actions)              # évaluation avec le réseau cible
```

Seulement deux lignes changent — mais l'impact sur la stabilité et la précision est significatif !

---

## Ce que montre la comparaison

Lorsque vous lancez `double_dqn_cartpole.py`, vous verrez deux graphiques :

**Graphique 1 : Courbes d'apprentissage**
- Le DQN classique et le Double DQN devraient tous deux résoudre CartPole.
- Double DQN converge souvent plus rapidement et de manière plus stable.
- CartPole est assez simple pour que la différence soit modeste ; elle est beaucoup plus spectaculaire sur Atari.

**Graphique 2 : Estimations des valeurs Q**
- DQN classique : les valeurs Q dérivent vers le haut avec le temps (surestimation).
- Double DQN : les valeurs Q restent plus modestes et précises.

Le graphique de surestimation de la valeur Q est l'élément clé — il montre comment le DQN classique apprend des valeurs gonflées qui finissent par nuire aux performances.

---

## À quel point Double DQN est-il meilleur ?

| Métrique | DQN classique | Double DQN |
|----------|---------------|------------|
| Précision de la valeur Q | Surestime | Plus précise |
| Stabilité de l'apprentissage | Plus de variance | Moins de variance |
| Performance sur CartPole | Bonne | Légèrement meilleure |
| Performance sur Atari (50 jeux) | Référence | +2,6× plus de jeux proches du niveau humain |

Sur les jeux Atari complexes, Double DQN a fait une bien plus grande différence que sur CartPole (car Atari a des estimations de valeurs Q beaucoup plus bruitées).

---

## La famille des améliorations de DQN

Double DQN n'est qu'une des nombreuses améliorations du DQN classique. L'article "Rainbow" (2017) a combiné 6 améliorations :

1. **Double DQN** (corrige la surestimation) ← ce script !
2. **Prioritized Replay** (apprendre davantage des expériences surprenantes)
3. **Dueling Networks** (séparer "à quel point cet état est-il bon ?" de "quelle est la meilleure action ?")
4. **Multi-step returns** (regarder plus loin dans le futur)
5. **Distributional RL** (apprendre la distribution complète des retours, pas seulement la moyenne)
6. **NoisyNets** (exploration apprise au lieu de [ε-greedy](../foundations/multi_armed_bandit_explained.md#the-epsilon-greedy-strategy))

Rainbow a combiné TOUTES ces améliorations et a obtenu les meilleures performances sur Atari de son époque !

---

## Vocabulaire clé

| Mot | Signification |
|-----|---------------|
| **Surestimation** | Les valeurs Q sont plus élevées que les valeurs réelles (trop optimistes) |
| **Double DQN** | Utilise le réseau en ligne pour la sélection des actions, le réseau cible pour l'évaluation |
| **Découplage** | Séparer deux tâches qui étaient effectuées par le même réseau |
| **Biais** | Une erreur systématique dans une direction (toujours trop haut, ou toujours trop bas) |
| **Rainbow** | Une variante de DQN qui combine 6 améliorations pour une performance maximale |

---

## Résumé : Le parcours de la Phase 3

Vous avez maintenant terminé toute la progression de la Phase 3 :

| Algorithme | Ce qu'il ajoute | Pourquoi cela aide |
|------------|-----------------|-------------------|
| Q linéaire | Réseau neuronal → formule simple | Gère les états continus |
| Naive DQN | Réseau neuronal complet | Apprend des schémas complexes |
| + Replay buffer | Échantillonnage aléatoire de la mémoire | Casse les corrélations |
| + Target network | Copie figée pour les cibles | Stabilise la "cible" |
| Atari DQN | CNN + empilement de trames | Apprend à partir de pixels ! |
| Double DQN | Sélection/évaluation séparées | Réduit la surestimation |

Chaque étape a résolu un problème spécifique. C'est ainsi que fonctionne la recherche réelle — une amélioration minutieuse à la fois !
