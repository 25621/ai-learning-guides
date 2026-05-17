# PPO pour le contrôle continu : Faire marcher BipedalWalker

## Actions discrètes vs continues

Jusqu'à présent, chaque environnement que nous avons résolu avait des actions **discrètes** :
- CartPole : pousser à GAUCHE ou à DROITE (2 choix)
- LunarLander : ne rien faire / gauche / moteur principal / droite (4 choix)

Mais les robots du monde réel ont besoin d'actions **continues** :
- Un robot humanoïde : « quelle force appliquer à chaque articulation » (n'importe quelle valeur de -1 à +1)
- Une voiture : « de quel angle exact tourner le volant » (n'importe quel angle de -30° à +30°)
- Un bras : « appliquer exactement 2,3 Newtons dans cette direction »

**Exemple concret :** Taper sur un clavier = discret (appuyer sur A, B, C...).
Écrire avec un crayon = continu (déplacer la main de 2,3 cm vers la droite, appliquer une force de 40g...).

---

## La politique gaussienne pour les actions continues

Pour les actions continues, au lieu d'une distribution catégorielle (choisir parmi N catégories), nous utilisons une **distribution normale (gaussienne)** :

```
Action ~ Normale(μ, σ)
```

Où :
- **μ (mu, moyenne)** : Le centre de la distribution — la valeur d'action que le réseau « vise ».
- **σ (sigma, écart-type)** : L'étalement — la quantité de hasard / d'exploration à ajouter.

```
        Probabilité
             │
        0.4 ─┤      ██████
             │    ████████████
        0.2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Valeur de l'action
           -1  -0.5   0   0.5   1
                      ↑
                   moyenne μ
```

**Exemple concret :** Un archer habile vise le centre de la cible (μ). Ses flèches n'atterrissent pas toutes exactement au même endroit — il y a une certaine dispersion (σ). À mesure qu'il s'entraîne, il devient plus précis (σ diminue) tout en restant centré sur le mille.

---

## Notre réseau Actor-Critic gaussien

```
État (24 nombres) → [256 neurones] → [256 neurones] →
    ├── Actor : 4 valeurs moyennes (μ₁, μ₂, μ₃, μ₄)
    │           + 4 paramètres log_std (partagés par tous les états !)
    └── Critic : 1 valeur (V(s))
```

Le `log_std` (logarithme de l'**écart-type** — une mesure de l'étalement ou de l'incertitude) est un **paramètre apprenable** — il ne dépend pas de l'état. Cela reste simple tout en permettant à l'exploration d'évoluer pendant l'entraînement.

**Pourquoi log_std au lieu de std ?** L'écart-type doit être positif. L'utilisation de `log_std` permet au réseau de produire n'importe quel nombre réel (positif ou négatif), puis nous appliquons `exp(log_std)` — la fonction exponentielle, qui est l'inverse du logarithme — pour retrouver un écart-type garanti positif. Cela empêche l'écart-type de devenir négatif ou nul.

---

## Calcul de la log-probabilité pour les actions continues

Pour les actions discrètes : `log_prob = log(P(action=GAUCHE))`

Pour les actions continues, la **distribution normale** décrit une courbe en cloche lisse autour de la moyenne. Une seule valeur exacte a une probabilité de zéro en mathématiques continues, nous utilisons donc la hauteur de la courbe à cette valeur, appelée la **fdp** (fonction de densité de probabilité) :
```
log_prob = Σᵢ log[Normale(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` désigne le logarithme népérien. Il transforme de minuscules valeurs de densité en nombres stables plus faciles à optimiser pour les réseaux de neurones. Nous sommons sur toutes les dimensions d'action (4 pour BipedalWalker), car l'action complète est un vecteur de 4 nombres.

**Exemple concret :** Quelle est la probabilité qu'il fasse exactement 5,732...°C demain ? Pour une météo continue, vous regarderiez la courbe de distribution normale et verriez sa hauteur à ce point précis. Les températures plus probables (proches de la moyenne) ont une probabilité plus élevée.

---

## BipedalWalker : Un défi de marche

BipedalWalker-v3 est un robot 2D qui doit apprendre à marcher sans tomber :

```
          O (tête)
         /│\
        / │ \
       /  │  \
      L   │   R   ← deux jambes, chacune avec une articulation de genou
     / \  │  / \
    ●   ● │ ●   ●  ← 4 moteurs (hanche/genou pour chaque jambe)
```

**Espace d'état (24 nombres) :**
- Coque (Hull) : angle, vitesse angulaire, vitesse horizontale, vitesse verticale (4 nombres)
- Articulations : 4 moteurs (2 hanches, 2 genoux) fournissant chacun l'angle et la vitesse, plus 2 capteurs de contact au sol (un pour chaque jambe) (10 nombres)
- 10 capteurs de distance LIDAR (lectures de distance qui voient le sol devant) (10 nombres)

**Espace d'action (4 valeurs continues, chacune dans [-1, 1]) :**
Les valeurs d'action contrôlent le **couple** (torque, la force de rotation appliquée par les moteurs) pour exactement 4 articulations (aucune action n'est appliquée directement sur la coque) :
- Couple hanche jambe 1, Couple genou jambe 1, Couple hanche jambe 2, Couple genou jambe 2.

**Récompenses :**
- +300 pour avoir atteint l'objectif (côté droit)
- -100 pour être tombé (le corps touche le sol)
- Petite récompense par étape de progression vers l'avant
- Petite pénalité pour chaque utilisation du moteur (récompense l'efficacité)

**Résolu quand :** Récompense moyenne > 300 sur 100 épisodes.

---

## Différence clé avec le PPO discret

Tout est identique SAUF :

| | PPO Discret | PPO Continu |
|---|---|---|
| **Politique** | Categorical(logits) | Normale(μ, σ) |
| **Échantillon** | action = échantillon parmi {0,1,...,N} | action = μ + σ × bruit |
| **log_prob** | log P(action=k) | Σ log Normale(μᵢ, σᵢ).pdf(aᵢ) |
| **Clamp** | Non requis | Clamper les actions dans [-1, 1] |

Les **logits** sont des scores bruts non normalisés pour des actions discrètes. Une politique catégorielle les convertit en probabilités avec **softmax** — une fonction qui prend n'importe quel ensemble de nombres et les écrase dans une distribution de probabilité valide (toutes les valeurs positives, somme égale à 1). Par exemple, les logits [2.0, 1.0, 0.5] deviennent des probabilités [0.59, 0.24, 0.17]. Le PPO continu n'utilise **pas** softmax pour l'action elle-même, car l'action n'est pas choisie dans un menu fixe. Au lieu de cela, la politique produit la moyenne et l'écart-type d'une distribution normale, puis en extrait des couples en nombres réels.

**Clamp** signifie forcer une valeur dans une plage valide. Le code utilise `action.clamp(-1, 1)` pour que l'environnement ne reçoive jamais une commande de moteur en dehors de ses limites autorisées.

Le **clipping** dans PPO signifie autre chose : PPO tronque (clipping) le ratio de probabilité à l'intérieur de la fonction de perte, comme expliqué dans la [section sur le clipping PPO](./ppo_scratch_explained.md#the-clipping-trick). Le clamping d'action protège l'interface de l'environnement ; le clipping PPO protège la mise à jour de la politique.

---

## Marcher en partant de zéro : ce que l'agent apprend

**Début de l'entraînement (récompenses négatives) :** Le robot s'agite au hasard, tombe immédiatement. Chaque épisode se termine par un crash en quelques secondes.

**Milieu de l'entraînement :** Le robot découvre que bouger les jambes alternativement permet de progresser vers l'avant. Il commence à faire de petits pas maladroits — la récompense devient moins négative.

**Fin de l'entraînement :** Une **démarche** (gait) fluide et efficace émerge. Une démarche est un motif de mouvement répété, comme l'alternance des pas gauche et droit. Le robot s'ajuste dynamiquement aux terrains accidentés en utilisant ses capteurs LIDAR pour adapter ses pas en temps réel.

**Exemple concret :** Un bébé qui apprend à marcher :
1. Tombe immédiatement (récompense négative)
2. Fait un pas, tombe (un peu moins négatif)
3. Fait quelques pas (petite récompense positive)
4. Traverse la pièce (grande récompense positive !)

---

## Pourquoi BipedalWalker a besoin de PPO (et pas de REINFORCE)

- Les épisodes de **BipedalWalker** peuvent durer jusqu'à 1600 étapes (bien plus long que CartPole !)
- Les **récompenses sont éparses** — les récompenses de progression vers l'avant sont minuscules par étape.
- **REINFORCE aurait besoin** de milliers d'épisodes complets pour obtenir un signal utile.

Les mises à jour n-étapes de PPO avec [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) permettent au robot d'apprendre à partir d'épisodes incomplets :
> « Même si je suis tombé après 50 étapes, ces étapes ont montré UNE CERTAINE progression. Utilisons une estimation de retour sur 50 étapes plutôt que d'attendre la fin de l'épisode. »

---

## Résultats

Après 500 mises à jour (≈ 1 million d'étapes d'environnement) :
- Le robot fait des progrès visibles, passant de l'agitation aléatoire à un mouvement vers l'avant.
- Amélioration constante de la courbe d'apprentissage.
- Une convergence totale vers une récompense > 300 nécessite plus d'entraînement (5 à 10 millions d'étapes).

La courbe d'apprentissage montre la « courbe en S » caractéristique du contrôle continu :
1. Progression initiale lente (stabilité de l'apprentissage)
2. Amélioration rapide (découverte de la démarche)
3. Affinement progressif (optimisation de la démarche)

---

## Points clés à retenir

| Concept | Français simple |
|---------|---------------|
| **Politique gaussienne** | Au lieu de choisir dans un menu, on lance une fléchette sur une plage de valeurs |
| **μ (moyenne)** | Là où la politique « vise » |
| **σ (écart-type)** | La quantité de hasard / d'exploration utilisée par la politique |
| **log_std comme paramètre apprenable** | Un taux d'exploration global mis à jour par optimisation basée sur le gradient (ascension de gradient sur la récompense, ou de manière équivalente descente de gradient sur la perte PPO) — comme n'importe quel autre poids du réseau |
| **Contrôle continu** | Contrôler des sorties en nombres réels (couples, forces, angles) |

---

## Quelle est la suite ?

PPO possède de nombreux **hyperparamètres** — des réglages que vous choisissez avant le début de l'entraînement (par opposition aux *paramètres* comme les poids du réseau, qui sont appris automatiquement). Les exemples incluent `clip_eps`, le taux d'apprentissage, le nombre d'époques et la taille du lot (batch size).

À quel point PPO est-il sensible à ces choix ? `ppo_hyperparams.py` mène des expériences en faisant varier systématiquement chaque hyperparameter et montre l'effet sur la vitesse et la stabilité de l'apprentissage.
