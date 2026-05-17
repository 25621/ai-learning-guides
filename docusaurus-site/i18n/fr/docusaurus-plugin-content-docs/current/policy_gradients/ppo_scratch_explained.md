# PPO : Des mises à jour de politique sûres et régulières

## Le problème d'A2C

Imaginez que vous apprenez à faire tenir un balai en équilibre sur votre doigt. Après des semaines de pratique, vous pouvez le maintenir pendant 30 secondes !

Maintenant, votre entraîneur vous donne un conseil : « Penchez votre poignet un peu plus vers la gauche. »

**Bon conseil → changement prudent → équilibre toujours maintenu pendant 30 secondes ✓**

Mais que se passe-t-il si l'entraîneur surréagit et dit : « PENCHEZ-VOUS COMPLÈTEMENT VERS LA GAUCHE IMMÉDIATEMENT ! »
Vous sur-corrigez → le balai tombe → vous avez perdu des semaines de progrès.

C'est le problème d'A2C : **des mises à jour de gradient trop importantes peuvent détruire une bonne politique**.

**PPO (Proximal Policy Optimization)** est un système de sécurité qui empêche cela.

---

## L'idée centrale : Rester proche de ce qui fonctionnait

La contrainte clé de PPO :

> **« Ne changez pas trop la politique en une seule mise à jour. »**

Avant une mise à jour, nous avons l'« ancienne » politique π_old.
Après la mise à jour, nous avons la « nouvelle » politique π_new.

PPO mesure l'ampleur du changement de politique avec le **ratio de probabilité** :

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1.0 : politique inchangée.
- r = 1.5 : la nouvelle politique a 50 % de chances de plus de choisir cette action.
- r = 0.5 : la nouvelle politique a 50 % de chances de moins de choisir cette action.

**Exemple concret :** Vous êtes un chef qui ajuste une recette.
- r = 1.0 : même quantité de sel qu'avant.
- r = 2.0 : double de sel — trop extrême !
- r = 0.9 : 10 % de sel en moins — changement petit et sûr.

---

## L'astuce du clipping (tronquage) {#the-clipping-trick}

PPO tronque (clipping) le ratio pour qu'il reste dans l'intervalle [1-ε, 1+ε] (généralement ε = 0.2) :

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Analysons cela :

**Cas 1 : L'action était BONNE (A > 0)**

Nous voulons faire cette action plus souvent (r > 1). Mais nous limitons l'augmentation :
```
si r > 1.2 : on tronque à 1.2, plus d'incitation à pousser davantage.
```
Cela nous empêche de basculer TROP loin dans une direction.

**Cas 2 : L'action était MAUVAISE (A < 0)**

Nous voulons faire cette action moins souvent (r < 1). Mais là encore, nous limitons :
```
si r < 0.8 : on tronque à 0.8, plus de pénalité pour aller plus loin.
```

**Visualisation :**
```
ε = 0.2, donc la fenêtre de ratio sûre est de 0.8 à 1.2.

Action BONNE (A > 0) : augmente la probabilité de l'action, mais cesse de récompenser après 1.2.
ratio r :       0.6      0.8      1.0      1.2      1.4
incitation :     ↑        ↑        ↑        ↑        -
signification : trop bas   ok     anc.     max    tronqué

Action MAUVAISE (A < 0) : diminue la probabilité de l'action, mais cesse de punir en dessous de 0.8.
ratio r :       0.6      0.8      1.0      1.2      1.4
incitation :     -        ↓        ↓        ↓        ↓
signification : tronqué   max     anc.      ok    trop haut
```

Le signe `-` marque la région plate tronquée. Dans cette région, rendre le ratio de probabilité encore plus extrême n'améliore pas l'objectif, donc PPO n'a aucune incitation supplémentaire à pousser plus loin.

**Exemple concret :** Un limiteur de vitesse sur une voiture. Vous pouvez accélérer, mais une fois que vous atteignez 120 km/h, le limiteur s'active et ne vous laisse pas aller plus vite. Il vous maintient en sécurité sans vous empêcher de bouger.

---

## Pourquoi cela empêche les mises à jour catastrophiques

Une **mise à jour catastrophique** se produit lorsqu'un seul changement important de politique détruit complètement tout ce que l'agent a appris — des heures d'entraînement perdues en une seule étape de gradient.

Sans clipping : une seule grande étape de gradient pourrait changer radicalement la politique.
Avec clipping : le gradient est nul en dehors de [1-ε, 1+ε], donc la politique ne peut bouger que par petits pas.

**Exemple concret :** Un bon chirurgien pratique des incisions petites et précises — pas de grands mouvements amples. PPO est le « chirurgien prudent » des optimiseurs de RL.

---

## GAE : Des estimations d'avantage plus intelligentes {#gae-smarter-advantage-estimates}

PPO utilise le **Generalized Advantage Estimation (GAE)** pour calculer l'avantage :

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (erreur TD)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE possède un paramètre λ (lambda) :
- λ = 0 : utilise uniquement l'erreur TD à une étape (faible variance, biais élevé).
- λ = 1 : utilise les retours complets de Monte Carlo (variance élevée, faible biais).
- λ = 0.95 : un bon équilibre entre les deux !

**Exemple concret :** Planifier un voyage en voiture.
- λ=0 : ne regarder que les 10 prochains kilomètres (sûr, mais pourrait rater un raccourci plus loin).
- λ=1 : considérer l'ensemble du trajet de 800 km (plus d'infos, mais très incertain).
- λ=0.95 : regarder loin devant mais accorder plus de poids aux routes proches ← le meilleur équilibre !

---

## Époques multiples : Réutiliser les données efficacement

Après avoir collecté un lot d'expérience (rollout), REINFORCE le jette après UNE SEULE mise à jour.

PPO réutilise chaque lot pendant **K époques** (généralement 4 à 10 passages sur les mêmes données) :

```
Collecte de 512 étapes × 4 environnements = 2048 transitions
Époque 1 : 32 mini-lots (minibatches) × mise à jour de chacun
Époque 2 : mélange, 32 mini-lots supplémentaires × mise à jour de chacun
Époque 3 : ...
Époque 4 : ...
```

**Qu'est-ce qu'un « mini-lot » (minibatch) ?** Faire une mise à jour avec les 2048 transitions à la fois est lent et gourmand en mémoire ; mettre à jour une transition à la fois est bruité. Un **mini-lot** est un petit groupe intermédiaire — ici, 2048 ÷ 32 = **64 transitions par mini-lot**. Nous calculons une étape de gradient par mini-lot, donc chaque époque effectue 32 mises à jour stables au lieu d'une seule énorme.

Le clipping garantit que ces multiples passages ne dépassent pas la cible — sans clipping, des époques multiples détruiraient la politique en la poussant trop loin !

**Exemple concret :** Un étudiant a 30 problèmes d'entraînement.
- REINFORCE : faire chaque problème une fois, apprendre un peu, puis les jeter.
- PPO : faire chaque problème 4 fois (sous des angles différents à chaque fois), limiter vos changements pour ne pas mémoriser de mauvais schémas.

---

## La perte PPO complète (Loss)

```
L = L_CLIP - c₁ · L_entropie + c₂ · L_critique

L_CLIP    = gradient de politique tronqué (clipped policy gradient)
L_entropie = bonus d'entropie (encourage l'exploration)  
L_critique  = MSE entre V(s) et les retours (returns)
```

Coefficients typiques : c₁ = 0.01 (entropie), c₂ = 0.5 (critique).

**Deux termes à approfondir :**

- **Gradient de politique** — la moitié « acteur » de la perte. Elle utilise le signal de gradient pour pousser la politique vers des actions avec un avantage plus élevé. C'est la même idée centrale que dans REINFORCE — voir le [guide REINFORCE](./reinforce_cartpole_explained.md#the-old-way-vs-the-new-way) pour l'intuition. PPO y ajoute simplement l'enveloppe de clipping.
- **MSE (Mean Squared Error)** — la moitié « critique » de la perte. Le critique V(s) prédit le retour attendu d'un état ; nous comparons sa prédiction au retour réel et élevons la différence au carré : `MSE = moyenne((V(s) - retour)²)`. L'élévation au carré punit davantage les grandes erreurs que les petites et donne un signal lisse pour l'entraînement.

---

## Les résultats

```
Mise à jour  200 | Récompense moy. : ~120
Mise à jour  400 | Récompense moy. : ~280
Mise à jour  800 | Récompense moy. : ~280-300
```

PPO sur CartPole montre une amélioration régulière mais tend à stagner (plateau) autour de 280-300. (Un **plateau** signifie que la courbe d'apprentissage s'aplatit — la récompense cesse de s'améliorer même si l'entraînement continue.) C'est en fait attendu — PPO est conçu pour des environnements plus difficiles avec des épisodes plus longs.

Observation intéressante : **REINFORCE a résolu CartPole plus rapidement !** (moyenne de 500 contre 300).

Pourquoi ? Les épisodes de CartPole sont courts (≤ 500 étapes), les retours exacts de REINFORCE sont donc très précis. Les estimations « bootstrappées » de PPO ajoutent une complexité inutile. PPO brille vraiment sur des environnements où attendre des épisodes complets est impraticable (comme BipedalWalker).

**Qu'est-ce que « BipedalWalker » ?** BipedalWalker est un environnement de référence en RL : un robot à deux jambes qui doit apprendre à marcher vers l'avant sur un terrain accidenté sans tomber. Contrairement aux deux actions discrètes de CartPole (GAUCHE / DROITE), BipedalWalker a des actions **continues** — quatre valeurs de couple (torque), une pour chaque articulation de jambe. Les épisodes peuvent durer des milliers d'étapes, ce qui est exactement le régime où l'efficacité des données et la stabilité de PPO portent leurs fruits.

---

## Équations clés

```
Ratio :      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Perte Clip : L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE :        A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Points clés à retenir

| Concept | Français simple |
|---------|---------------|
| **Ratio r(θ)** | De combien la politique a changé sur cette action. |
| **Clip ε** | La limite de sécurité — ne pas changer la politique de plus de cette valeur. |
| **GAE** | Une façon intelligente d'estimer les avantages en regardant plusieurs étapes en avant. |
| **Efficacité des données** | Chaque rollout est collecté à partir de plusieurs environnements parallèles et réutilisé pour K époques de mises à jour — le clipping sécurise ces passages répétés. |

---

## Quelle est la suite ?

Jusqu'à présent, tous nos environnements avaient des actions **discrètes** (pousser à GAUCHE ou à DROITE).

Les vrais robots doivent contrôler des actions **continues** — comme « appliquer exactement 0,73 Newton de force ».

`ppo_continuous.py` étend PPO aux actions continues en utilisant une **politique gaussienne**, et le teste sur l'environnement BipedalWalker-v3, bien plus difficile !
