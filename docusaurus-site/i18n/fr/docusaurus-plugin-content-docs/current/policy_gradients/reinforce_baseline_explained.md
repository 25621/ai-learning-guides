# REINFORCE avec Ligne de Base : Réduire le Bruit

## Le Problème de REINFORCE Standard

Imaginez que vous êtes un étudiant essayant de décider si votre réponse à un examen était bonne.

**Retour flou :** "Tu as eu 7 points !"

Est-ce que 7 est un bon score ? Si le maximum est 10, oui ! Si tous les autres ont eu 9, non ! Sans contexte, vous ne pouvez pas savoir si vous devez changer votre façon de répondre.

C'est exactement le problème de REINFORCE : il utilise les **retours bruts** (G_t) pour évaluer les actions. Un score de retour total de 200 points peut être incroyable ou terrible selon la situation.

---

## L'arrivée de la Ligne de Base (Baseline)

Une **ligne de base** b(s) est un point de référence : "Quelle récompense est-ce que j'**attends** dans cette situation ?"

Au lieu de demander "Est-ce que cette action était bonne ?", nous demandons :

> **"Est-ce que cette action était meilleure que ce à quoi je m'attendais normalement ?"**

```
Ancien signal : mise à jour ∝ G_t
Nouveau signal : mise à jour ∝ (G_t - b(s_t))
```

**Exemple concret :** Vous avez eu 85 à un test de maths.
- Si la moyenne de la classe est de 60 → votre réponse était **25 points au-dessus de la moyenne** → excellent !
- Si la moyenne de la classe est de 90 → votre réponse était **5 points en dessous de la moyenne** → à retravailler !

L'**avantage** (G_t - b(s)) est positif lorsque vous avez fait mieux que prévu et négatif lorsque vous avez fait pire. C'est un signal d'apprentissage beaucoup plus clair !

---

## Qu'est-ce que la Ligne de Base ?

La ligne de base naturelle est la **fonction de valeur V(s)** :

> V(s) = "Récompense totale attendue si je suis dans l'état s et que je joue ma politique actuelle"

Nous apprenons cela avec un **Réseau de Valeur** séparé (également appelé réseau de base ou critique) :

```
État  →  [128 neurones]  →  [128 neurones]  →  V(s)   (un seul nombre)
```

Pour chaque état visité par l'agent, V(s) prédit le retour attendu. Si le retour réel G_t est supérieur à V(s), l'action était meilleure que prévu !

---

## Deux Réseaux qui Apprennent Ensemble

```
L'épisode se déroule
     ↓
Calcul des retours réels G_t
     ↓
         ┌─────────────────────────────┐
         │ Avantage = G_t - V(s_t)     │
         │  + : l'action était meilleure│
         │  - : l'action était pire     │
         └─────────────────────────────┘
              ↓                  ↓
    Mise à jour du Réseau   Mise à jour du Réseau
    de Politique            de Valeur
    (rendre les bonnes      (rendre les prédictions
     actions plus/moins     plus précises pour la
     probables)             prochaine fois)
```

**Exemple concret :** Deux amis vont ensemble au restaurant.

- Ami 1 (Réseau de Valeur) : "Je prédis que ce plat vaudra 7/10"
- Ami 2 (Réseau de Politique) : Vous goûtez le plat et lui donnez 9/10
- Avantage = 9 - 7 = +2 → "C'était meilleur que prévu ! Commande-le encore !"

À la visite suivante, l'ami 1 met à jour sa prédiction pour qu'elle soit plus proche de 9/10.
L'ami 2 est plus susceptible de commander ce plat la prochaine fois.

---

## Pourquoi cela réduit-il la variance ?

**Preuve mathématique (intuition) :**

Sans ligne de base : `gradient ∝ ∇log π(a|s) × G_t`

Les valeurs de G_t varient beaucoup d'un épisode à l'autre :
```
Épisode 1 : G = [45, 44, 43, ..., 1]   (partie moyenne)
Épisode 2 : G = [500, 499, ..., 1]      (excellente partie !)
Épisode 3 : G = [12, 11, ..., 1]        (terrible partie)
```

Les estimations de gradient sautent énormément parce que G_t est grand et bruité.

Avec la ligne de base : `gradient ∝ ∇log π(a|s) × (G_t - V(s_t))`

L'avantage (G_t - V(s_t)) est beaucoup plus petit et centré autour de zéro :
```
Épisode 1 : avantage ≈ [-2, +1, -3, ..., 0]   (petit, centré)
Épisode 2 : avantage ≈ [+10, +8, ..., +3]      (cette partie ÉTAIT excellente)
Épisode 3 : avantage ≈ [-5, -6, ..., -2]       (cette partie ÉTAIT mauvaise)
```

**Exemple concret :** Mesurer votre vitesse de course.
- Sans ligne de base : "J'ai couru à 8 km/h" (insignifiant sans contexte)
- Avec ligne de base : "J'ai couru 2 km/h PLUS VITE que ma moyenne" (clairement bon !)

L'avantage est toujours une comparaison — il est naturellement plus petit et plus stable.

---

## Crucial : Pas de Biais !

La ligne de base ne change pas CE QUE l'algorithme apprend — seulement À QUELLE VITESSE et avec quelle STABILITÉ il apprend.

**Pourquoi ?** Parce que l'avantage attendu est toujours 0 en espérance :

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

N'importe quel b(s) qui ne dépend pas de l'action fonctionne comme une ligne de base valide !

**Exemple concret :** Noter avec une courbe de Gauss ne change pas qui a le mieux performé — cela rend simplement les scores plus faciles à interpréter. Le classement reste le même ; seule l'échelle change.

---

## Les Résultats

```
Sans ligne de base  — Moyenne finale 100-ep : 500.0, variance grad : 599.3
Avec ligne de base — Moyenne finale 100-ep : 491.4, variance grad : 578.8
```

Les deux méthodes atteignent une performance quasi parfaite sur CartPole, mais remarquez :
1. La **variance du gradient** est mesurable (le graphique de droite montre la variance pendant l'entraînement)
2. Avec la ligne de base, l'agent atteint des performances élevées de manière **plus fiable** — moins de chutes brutales de récompense pendant l'entraînement.

La réduction de la variance est plus spectaculaire dans des environnements plus difficiles (LunarLander, MuJoCo).

---

## Équations Clés

```
Valeur de la ligne de base : V(s) ← V(s) + α(G_t - V(s))   [minimisation de l'EQM]
Gradient de politique :      θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Avantage :                   A_t = G_t - V(s_t)
```

---

## Points Clés à Retenir

| Concept | Français Simple |
|---------|---------------|
| **Ligne de Base b(s)** | Récompense attendue dans l'état s — notre point de référence |
| **Avantage A_t** | "Est-ce que cette action était meilleure que prévu ?" |
| **Réseau de Valeur** | Un réseau de neurones qui apprend à prédire les retours attendus |
| **Réduction de variance** | Moins de bruit dans les estimations de gradient → apprentissage plus stable |
| **Impartial (Sans biais)** | La ligne de base ne change pas la politique cible en moyenne ; elle rend simplement le signal d'apprentissage moins bruité et plus stable |

---

## Prochaine Étape ?

La ligne de base n'est en fait que le début de quelque chose de bien plus puissant : les méthodes **Actor-Critic**.

Au lieu de calculer V(s) seulement à la fin d'un épisode, l'Actor-Critic met à jour V(s) à chaque étape en utilisant l'apprentissage par **Différence Temporelle**. Cela rend les mises à jour beaucoup plus rapides et permet à l'agent d'apprendre à partir d'épisodes incomplets !

Consultez `a2c_lunarlander.py` pour l'implémentation complète de l'Actor-Critic.
