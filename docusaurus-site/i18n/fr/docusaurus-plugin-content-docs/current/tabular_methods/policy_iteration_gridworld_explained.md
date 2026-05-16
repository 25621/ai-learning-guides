# Itération de politique pour GridWorld 🗺️

## Qu'est-ce que c'est ?

Imaginez que vous jouez à un jeu de plateau sur une **grille 4×4** (comme un minuscule échiquier). Vous commencez dans un coin et devez atteindre l'autre coin. Chaque étape coûte 1 point (vous ne voulez pas gaspiller d'étapes !), et atteindre l'objectif ne vous rapporte rien de plus — vous voulez simplement y arriver le plus vite possible.

**L'itération de politique** (Policy Iteration) est la manière dont un ordinateur détermine les **meilleurs mouvements pour chaque case** du plateau — tout en même temps !

---

## La grande idée : deux étapes, encore et encore

Pensez-y comme si vous rangiez votre chambre avec un assistant :

1. **Étape 1 — Déterminer la valeur de chaque case (Évaluation de politique)**
   Votre assistant parcourt chaque case et note : « Si je suis le plan actuel, combien d'étapes me faudra-t-il pour atteindre la sortie d'ici ? » Il fait cela encore et encore jusqu'à ce que les nombres ne changent plus.

2. **Étape 2 — Améliorer le plan (Amélioration de politique)**
   Maintenant, vous regardez chaque case et demandez : « Y a-t-il une meilleure direction que je pourrais prendre d'ici ? » Si oui, mettez à jour le plan !

Répétez les étapes 1 et 2 jusqu'à ce que le plan ne change plus — c'est la **politique optimale** !

**Exemple concret :** Imaginez trouver l'itinéraire le plus rapide pour aller à l'école. D'abord, vous devinez un itinéraire et vous le chronométrez (Étape 1). Ensuite, vous regardez chaque intersection et demandez « y a-t-il un raccourci à partir d'ici ? » (Étape 2). Vous mettez à jour votre itinéraire et répétez jusqu'à ce que vous ne puissiez plus trouver de raccourcis !

---

## Ce que notre code a trouvé

Notre GridWorld 4×4 possède deux états terminaux (coins), et l'agent paie -1 par étape. L'itération de politique a convergé en seulement **4 rounds** (139 balayages d'évaluation au total) :

```
Valeurs d'état V(s) :      Politique optimale :
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**Les valeurs sont parfaitement logiques !** Les cases adjacentes à un terminal ont une valeur de -1 (à une étape). Les cases à deux étapes ont une valeur de -1.9 (= -1 + 0.9 × -1), et ainsi de suite.

---

## Exemples concrets

- **Navigation GPS** : Déterminer le meilleur virage à *chaque* intersection sur la carte.
- **Contrôle d'ascenseur** : À quel étage l'ascenseur doit-il se rendre lorsqu'il a plusieurs demandes ?
- **Robot d'usine** : Planifier le chemin le plus efficace dans une grille d'entrepôt.

---

## Mots clés à retenir

- **Politique (Policy)** : Le plan — quelle action entreprendre dans chaque état.
- **Fonction de valeur V(s)** : À quel point il est bon d'être dans l'état s (plus c'est haut, plus on est proche du but).
- **Évaluation de politique** : Calculer la qualité du plan actuel.
- **Amélioration de politique** : Rendre le plan meilleur en utilisant la fonction de valeur.
- **Politique optimale** : Le meilleur plan possible — il ne peut plus être amélioré.

La grande idée : **Vous n'avez pas besoin d'essayer tous les plans possibles ! Améliorez simplement le plan actuel, et vous trouverez le meilleur plan en très peu de rounds.**
