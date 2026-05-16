# A2C : L'Acteur et le Critique Travaillent Ensemble

## L'Idée Maîtresse

REINFORCE attend que la partie soit complètement terminée avant d'effectuer une mise à jour. C'est comme un entraîneur qui regarderait tout un match de football en silence, puis donnerait tous ses commentaires à la fin.

**A2C (Advantage Actor-Critic)** donne des commentaires PENDANT la partie — toutes les quelques étapes, l'entraîneur fait une pause pour dire : "Cette passe était excellente ! Ce tacle était mauvais !"

C'est beaucoup plus rapide et efficace.

---

## Rencontre avec les Deux Personnages

> **Qu'est-ce que LunarLander ?** Tout au long de ce document, nous utilisons l'environnement **LunarLander** — une simulation physique où vous contrôlez un petit vaisseau spatial et devez le poser en douceur sur une plateforme cible sur la lune en utilisant trois moteurs (gauche, principal et droit). Il s'agit d'un benchmark standard en apprentissage par renforcement, disponible dans la bibliothèque Gymnasium.

### L'Acteur 🎭
L'**Acteur** est la politique (policy) — il décide de l'action à entreprendre.

> "Je suis dans cet état. Dois-je allumer le moteur gauche ou le moteur droit ?"

**Exemple concret :** Le *conducteur* d'une voiture qui tourne le volant et appuie sur les pédales.

### Le Critique 🎬
Le **Critique** estime la qualité de la situation actuelle — la valeur V(s).

> "Être dans CET état vaut environ +150 points de récompense future totale."

**Exemple concret :** Le *navigateur* assis à côté du conducteur, disant : "Nous sommes sur une bonne route — nous devrions arriver dans 30 minutes" ou "Nous nous dirigeons vers un embouteillage — ça va être lent."

### Ils partagent un Cerveau
Dans notre implémentation, les deux utilisent la **même base de réseau de neurones** :

```
          État (8 nombres pour LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Couches Partagées      │
          │  [256 neurones] → ReLU   │
          │  [256 neurones] → ReLU   │
          └────────┬────────┬───────┘
                   ↓        ↓
          Tête de l'Acteur    Tête du Critique
          [4 sorties]         [1 sortie]
          (probabilités d'action) (V(s))
```

- **ReLU** (Rectified Linear Unit) : une fonction d'activation appliquée après chaque couche — elle renvoie `max(0, x)`, conservant les valeurs positives et mettant à zéro les négatives. Cela permet au réseau d'apprendre des motifs non linéaires.
- **probabilités d'action** : la probabilité de choisir chacune des 4 actions. L'Acteur échantillonne à partir de cette distribution pour choisir une action à chaque étape.

**Exemple concret :** Un seul cerveau, deux tâches — comme un chauffeur de taxi qui à la fois conduit (acteur) ET sait si l'itinéraire est bon (critique). Partager le cerveau permet d'apprendre plus vite !

---

## L'Avantage : Était-ce mieux que prévu ?

Tout comme REINFORCE avec ligne de base (baseline), A2C calcule l'**Avantage** :

> A(s, a) = "Résultat réel" − "Ce que nous attendions"

Mais ici, le "résultat réel" provient du **bootstrap à n étapes** du Critique (**bootstrapping** = utiliser la propre prédiction du Critique V(s) pour approximer la valeur des étapes futures, au lieu d'attendre la fin réelle de l'épisode — comme estimer votre note finale à l'examen à la mi-semestre en utilisant votre note actuelle) :

```
Retour TD réel : r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Avantage A_t = Retour TD - V(s_t)
```

**Exemple concret :** Vous vous attendez à marquer 3 buts ce match (V(s)). Si vous marquez 5 buts, votre avantage est de +2. Si vous marquez 1 but, votre avantage est de -2.

Avantage positif → "cette action a aidé plus que prévu → faites-la plus souvent !"
Avantage négatif → "cette action a aidé moins que prévu → faites-la moins souvent !"

---

## Pourquoi utiliser plusieurs environnements en parallèle ?

Notre A2C utilise **8 copies** de LunarLander fonctionnant en même temps !

**Pourquoi ?** Parce que les expériences d'un seul environnement sont **corrélées** — une étape suit de près l'étape précédente. Cette corrélation trompe le réseau de neurones en lui faisant croire que les motifs sont plus fiables qu'ils ne le sont réellement.

Avec 8 environnements, chaque étape donne 8 expériences indépendantes provenant de situations très différentes. Cela brise la corrélation et stabilise l'entraînement de manière spectaculaire.

**Exemple concret :** Pour en savoir plus sur la météo, qu'est-ce qui est préférable :
- Surveiller une seule ville pendant 8 heures consécutives (corrélé — s'il faisait beau à 14h, il fera probablement beau à 15h)
- Surveiller 8 villes simultanément (décorrélé — des modèles météorologiques différents, plus d'informations !)

```
Environnement 1 : [atterri sur la lune, moteur gauche, crash, réinitialisation...]
Environnement 2 : [chute trop rapide, les deux moteurs, vol stationnaire, atterrissage...]
Environnement 3 : [inclinaison à droite, moteur droit, stabilisation, atterrissage...]
...
Environnement 8 : [dérive à gauche, moteur gauche, stable, ...]
```

Les 8 mettent à jour le réseau simultanément — 8× plus d'expériences diversifiées par mise à jour !

---

## Mises à jour à N Étapes : N'attendez pas la fin du jeu

REINFORCE attend un épisode complet (qui pourrait durer 1000 étapes !).

A2C effectue une mise à jour toutes les **n_steps = 128 étapes** :

```
Jouer 128 étapes sur 8 environnements
    → Obtenir 128 × 8 = 1024 tuples d'expérience
    → Calculer les avantages et les retours
    → Mettre à jour l'Acteur et le Critique
    → Jouer 128 étapes de plus...
```

**Exemple concret :** Un étudiant révisant pour un examen.
- Style REINFORCE : Lire tout le manuel, PUIS faire des tests d'entraînement.
- Style A2C : Lire 10 pages, faire un quiz, lire 10 pages de plus, faire un quiz...

Des retours plus fréquents = un apprentissage plus rapide !

---

## Trois Pertes (Losses) Combinées

A2C s'entraîne avec trois termes de perte simultanément :

Une **perte (loss)** est le nombre que l'optimiseur essaie de minimiser. Une perte plus petite signifie que le comportement actuel du réseau est plus proche de l'objectif d'entraînement.

### 1. Perte de l'Acteur (Gradient de Politique)
Rendre les actions avantageuses plus probables :
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Si A > 0 : augmenter la probabilité de cette action
Si A < 0 : diminuer la probabilité de cette action

### 2. Perte du Critique (MSE de la Fonction de Valeur)
Rendre les prédictions de valeur plus précises (**MSE** = Mean Squared Error : mettre au carré l'erreur de prédiction et faire la moyenne — le carré pénalise plus lourdement les grandes erreurs que les petites) :
```
L_critic = E[(V(s) - return)²]
```
Comme l'entraînement de n'importe quel modèle de **régression** (régression = prédire un nombre continu, ici le retour attendu V(s)) — minimiser l'erreur de prédiction.

### 3. Bonus d'Entropie (Exploration)
Empêcher la politique de devenir trop confiante trop vite :
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Entropie élevée = choix d'actions diversifiés = exploration
Entropie faible = choix confiants et restreints = exploitation

**Exemple concret :** Le bonus d'entropie est comme un enseignant disant : "Ne vous contentez pas de deviner A à chaque question à choix multiples ! Essayez différentes réponses pour apprendre ce qui fonctionne."

```
Perte totale = L_actor + 0,5 × L_critic - 0,01 × entropie
```

---

## LunarLander : Un Défi plus Difficile

**LunarLander-v3** est un environnement Gymnasium (anciennement OpenAI Gym) — "v3" est le numéro de version indiquant la troisième révision de cet environnement. L'agent contrôle un petit vaisseau spatial qui doit se poser en toute sécurité sur une plateforme désignée sur la lune. C'est beaucoup plus difficile que CartPole :
- Espace d'états à 8 dimensions (position, vitesse, angle, contact des pattes, carburant)
- 4 actions discrètes (ne rien faire, moteur gauche, moteur principal, moteur droit)
- Récompense : +100 pour l'atterrissage, -100 pour le crash, petites pénalités de carburant

La courbe d'apprentissage montre une amélioration progressive, passant de récompenses très négatives à des récompenses positives. A2C sur LunarLander nécessite une expérience significative avant que l'atterrisseur n'apprenne la stabilité de base.

---

## Équations Clés

```
retour à n étapes :  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Avantage :           A_t = G_t - V(s_t)
Mise à jour Acteur : θ_π ← θ_π - α ∇ L_actor
Mise à jour Critique : θ_V ← θ_V - α ∇ L_critic
```

---

## Points Clés à Retenir

| Concept | Français simple |
|---------|---------------|
| **Acteur** | La politique — décide quoi faire |
| **Critique** | La fonction de valeur — juge la qualité de la situation |
| **Avantage** | "Était-ce mieux que prévu ?" (réel - attendu) |
| **Retour à n étapes** | Regarder n étapes dans le futur avant d'utiliser V(s) pour le bootstrap |
| **Env. parallèles** | Plusieurs environnements pour une expérience décorrélée et diversifiée |
| **Bonus d'entropie** | Encourage l'acteur à continuer d'essayer de nouvelles choses |

---

## Et après ?

A2C est excellent mais présente une faiblesse majeure : il met parfois à jour la politique de manière trop agressive. Une seule mauvaise mise à jour peut détruire tout le bon apprentissage d'une mise à jour précédente.

**PPO (Proximal Policy Optimization)** corrige cela avec un "clip de sécurité" astucieux qui empêche toute mise à jour unique de trop modifier la politique.

Consultez `ppo_scratch.py` pour l'implémentation de PPO !
