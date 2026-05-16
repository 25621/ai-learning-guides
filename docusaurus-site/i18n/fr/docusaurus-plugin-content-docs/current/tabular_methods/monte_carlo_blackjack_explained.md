# Contrôle de Monte-Carlo pour le Blackjack 🃏

## Qu'est-ce que c'est ?

Avez-vous déjà joué à un jeu de cartes où vous deviez décider : **"Est-ce que je prends une autre carte, ou est-ce que je me contente de ce que j'ai ?"**

Le **Blackjack** (aussi appelé "21") c'est exactement ça ! Vous voulez que la somme de vos cartes se rapproche le plus possible de 21, sans la dépasser. Si vous dépassez 21, vous "sautez" (*bust*) et vous perdez !

Le **contrôle de Monte-Carlo** est la méthode utilisée par un robot pour apprendre à jouer au Blackjack — en jouant des *milliers de parties complètes* et en mémorisant ce qui a fonctionné ou non.

---

## L'Idée Maîtresse : Apprendre d'Histoires Complètes

Le terme "Monte-Carlo" vient du célèbre casino de Monaco. En mathématiques, cela signifie : **utiliser des expériences aléatoires pour apprendre quelque chose**.

Voici comment cela fonctionne :

1. **Jouer une partie complète** (un épisode complet) en utilisant votre stratégie actuelle.
2. **Observer ce qui s'est passé** : Avez-vous gagné ? Perdu ? Fait match nul ?
3. **Travailler à rebours** : Était-ce une bonne idée de tirer (*hit*) à 17 ? Et à 14 ?
4. **Mettre à jour votre mémoire** : Se souvenir si chaque décision a mené à la victoire ou à la défaite.

Faites cela pour **500 000 parties** et vous deviendrez très bon !

**Exemple concret :** Imaginez apprendre à cuisiner en préparant 500 000 repas. À chaque fois, vous vous rappelez exactement de ce que vous avez fait — et si le plat était bon. Après assez d'essais, vous saurez : "Ajouter trop de sel à cette étape rend toujours le plat mauvais." Monte-Carlo fonctionne de la même manière !

---

## Différence Clé avec SARSA et le Q-Learning

SARSA et le Q-Learning mettent à jour leurs connaissances **après chaque étape individuelle** (même en milieu d'épisode). Monte-Carlo attend que **l'épisode entier soit terminé**, puis analyse tout ce qui s'est passé.

| Méthode | Mise à jour quand ? | Nécessite un épisode complet ? |
|--------|---------------|------------------------|
| **TD (SARSA, Q-Learning)** | Après chaque étape | Non |
| **Monte-Carlo** | Après chaque épisode complet | Oui |

Cela rend Monte-Carlo plus simple à comprendre, mais il ne peut rien apprendre avant la fin de chaque épisode.

---

## L'État au Blackjack

Le robot examine 3 éléments à chaque tour :
1. **Le total de mes cartes** (de 12 à 21)
2. **Quelle carte le donneur (dealer) montre-t-il ?** (de l'As au 10)
3. **Ai-je un As utilisable ?** (Un As peut compter pour 1 ou 11)

À partir de ces 3 informations, il décide : **Tirer (*Hit*) ou Rester (*Stick*)** ?

---

## Ce que Notre Code a Révélé

Après **500 000 parties** de Blackjack :

| Résultat | Pourcentage |
|---------|------------|
| **Victoires** | **43,1%** |
| **Matchs nuls** | 8,9% |
| **Défaites** | 48,0% |

C'est proche de la "stratégie de base" mathématiquement optimale (environ 42-43% de victoires) ! Le robot a appris quand tirer et quand rester — simplement en jouant et en se souvenant.

La politique apprise montre :
- **Tirer** (*Hit*) quand votre total est bas (peu de chances de sauter).
- **Rester** (*Stick*) quand votre total est élevé (risque de sauter si vous prenez une autre carte).
- Avoir un **As utilisable** permet d'être plus agressif (il peut passer de 11 à 1 si besoin).

---

## Exemples Concrets

- **Prévisions météorologiques** : Les simulations de Monte-Carlo font tourner des milliers de scénarios "et si" pour prédire le temps de demain.
- **Modélisation boursière** : Les analystes simulent des milliers de futurs possibles pour estimer le risque.
- **Apprendre aux échecs** : Un joueur analyse des parties entières (pas seulement des coups isolés) pour comprendre quelle stratégie a mené à la victoire.

---

## Mots Clés à Retenir

- **Épisode** : Une partie complète du début à la fin.
- **Retour (G)** : Récompense totale collectée à partir d'un point du jeu jusqu'à la fin.
- **MC à chaque visite (Every-visit MC)** : Mettre à jour le score d'un état à chaque fois qu'il est rencontré dans un épisode.
- **Pas de bootstrapping** : Monte-Carlo n'utilise pas d'estimations de valeurs futures — il attend le résultat réel !
- **Politique ε-soft** (ε = epsilon) : Faire généralement la meilleure action connue, mais explorer parfois de manière aléatoire.

L'idée maîtresse : **Monte-Carlo apprend en jouant de nombreuses parties complètes. C'est comme apprendre par l'expérience — vous vous souvenez de tout ce qui s'est passé et vous déduisez ce qui a mené à la victoire !**
