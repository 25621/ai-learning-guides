# Le Problème du Bandit Manchot (Multi-Armed Bandit) 🎰

## Qu'est-ce que c'est ?

Imaginez que vous êtes à une fête d'anniversaire et qu'il y a **10 bocaux de bonbons différents**. Chaque bocal contient des bonbons, mais certains bocaux ont des bonbons *délicieux* et d'autres des bonbons *moins bons*. Vous ne savez pas quel bocal est le meilleur — vous devez les goûter !

Chaque fois que vous piochez dans un bocal, vous obtenez un bonbon. Votre mission est :

> **Obtenir autant de bonbons délicieux que possible !**

C'est le problème du Bandit Manchot ! Au lieu de bocaux de bonbons, les scientifiques les appellent des "bras" (comme les bras d'une machine à sous). Chaque bras vous donne un prix, mais les prix sont différents à chaque fois.

---

## La Grande Question : Essayer de Nouveaux Bocaux ou Rester sur mon Préféré ?

C'est la partie la plus difficile ! Disons que vous avez essayé le bocal n°3 et qu'il était plutôt bon. Vous avez maintenant un choix à faire :

- **Exploiter** : Continuer à choisir le bocal n°3 parce que vous savez déjà qu'il est bon.
- **Explorer** : Essayer un nouveau bocal — peut-être que le bocal n°7 est encore *meilleur* !

Si vous ne choisissez que le premier bocal que vous avez aimé, vous risquez de rater le bocal super-délicieux. Mais si vous essayez *toujours* de nouveaux bocaux, vous n'utilisez jamais ce que vous avez déjà appris !

**Exemple concret :** Pensez à votre restaurant préféré. Vous commandez toujours des nuggets de poulet (exploitation !), mais peut-être que la pizza est encore meilleure. Si vous n'essayez jamais rien de nouveau, vous ne le saurez jamais !

---

## La Stratégie Epsilon-Gloutonne (Epsilon-Greedy)

Une façon intelligente de résoudre ce problème s'appelle **epsilon-glouton** (epsilon est simplement la lettre grecque ε, prononcée "èp-si-lon") :

1. **La plupart du temps (disons 90%)** : Choisissez le bocal que vous *pensez* être le meilleur.
2. **Parfois (disons 10%)** : Choisissez un bocal *au hasard* pour explorer !

Les 10 % de trajets d'exploration vous aident à découvrir de meilleurs bocaux. Les 90 % de trajets d'exploitation vous permettent d'utiliser ce que vous avez déjà appris.

---

## Ce que Notre Code a Révélé

Nous avons testé 10 bras (bocaux de bonbons) avec 200 enfants différents, 1 000 choix chacun :

| Stratégie | % de temps à choisir le meilleur bocal |
|----------|----------------------------------|
| **Ne jamais explorer (ε=0)** | 14,5 % — est resté bloqué trop tôt, n'a jamais trouvé le meilleur ! |
| **Explorer 1 % du temps (ε=0,01)** | 37,6 % — a fini par trouver le meilleur bocal, mais lentement |
| **Explorer 10 % du temps (ε=0,10)** | **74,2 %** — a appris rapidement et a choisi le meilleur la plupart du temps ! |

**Leçon** : Un petit peu d'exploration fait toute la différence !

---

## Exemples Concrets

- **Recommandations Netflix** : Netflix doit-il vous montrer un film que vous aimerez probablement (exploitation) ou suggérer quelque chose de nouveau (exploration) ?
- **Médecin choisissant un traitement** : Utiliser le traitement qui fonctionne habituellement (exploitation) ou en essayer un nouveau qui pourrait être encore meilleur (exploration) ?
- **Une abeille cherchant des fleurs** : Doit-elle continuer à visiter les fleurs qu'elle connaît pour leur nectar, ou s'envoler vers un nouveau champ ?

---

## Mots Clés à Retenir

- **Bras (Arm)** : L'un des choix (comme un bocal de bonbons).
- **Récompense (Reward)** : Ce que vous obtenez en choisissant un bras (comme un bonbon).
- **Exploiter** : Utiliser ce que vous savez déjà être bon.
- **Explorer** : Essayer quelque chose de nouveau pour en apprendre davantage.
- **Epsilon (ε)** : La probabilité d'explorer au lieu d'exploiter.

L'idée maîtresse : **Vous devez équilibrer l'essai de nouvelles choses avec l'utilisation de ce que vous savez déjà !**
