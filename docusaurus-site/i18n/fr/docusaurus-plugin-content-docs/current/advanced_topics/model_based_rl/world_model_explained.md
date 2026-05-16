# Entraîner un Modèle de Monde (World Model) : Apprendre à l'agent à rêver 🌍

## Qu'est-ce qu'un "Modèle de Monde" ?

Un **modèle de monde** est la *copie interne de l'univers* de l'agent. Donnez-lui un état et une action, et il prédit ce qui se passera ensuite :

```
(état, action)  ──►  Réseau de neurones  ──►  (état_suivant, récompense)
```

Ce n'est pas le monde réel — c'est un **simulateur que l'agent a construit pour lui-même** en observant la réalité et en apprenant à l'imiter.

Une fois entraîné, le modèle permet à l'agent de poser des questions de type "et si" sans entreprendre d'action réelle :

> *"Si je pousse à gauche maintenant puis deux fois à droite, où vais-je atterrir ? Est-ce que le mât va tomber ?"*

L'agent peut envisager une centaine de plans à l'intérieur de son modèle dans le temps qu'il lui faudrait pour effectuer un seul mouvement réel. C'est tout l'intérêt.

---

## Une analogie de la vie réelle

Pensez à la façon dont *vous* résolvez un puzzle. Vous ne déplacez pas physiquement chaque pièce dans chaque emplacement. Vous **imaginez** ce qui se passe si la pièce A va ici. Si cette simulation mentale semble erronée, vous la rejetez avant même de lever le petit doigt.

Votre cerveau possède un modèle de monde appris — construit à partir d'années d'observation du comportement des objets — qui vous permet de simuler les résultats avant de vous engager.

D'autres exemples :

- **Un joueur d'échecs** imagine les coups plusieurs tours à l'avance.
- **Un conducteur** pense : "Si je freine maintenant, la voiture derrière a assez de place."
- **Un enfant** empilant des blocs : "Si je mets le gros au-dessus, la tour va vaciller." (Il a appris ce modèle en renversant des tours auparavant.)

Dans chaque cas, **un modèle mental + l'imagination = de meilleures décisions avec moins de risques**.

---

## Comment l'agent construit-il son modèle ?

Il se contente d' **observer**. Plus précisément :

1. **Collecter des données.** Laisser n'importe quelle politique (même aléatoire) interagir avec l'environnement réel pendant un certain temps. Enregistrer chaque transition :
   ```
   (état, action, récompense, état_suivant)
   ```
2. **Entraîner un réseau de neurones** pour prédire l' `état_suivant` et la `récompense` à partir de `(état, action)`. Il s'agit d'un apprentissage supervisé : chaque transition enregistrée est un exemple étiqueté où l'entrée est "ce que l'agent a vu et fait" et l'étiquette est "ce qui s'est réellement passé ensuite".
3. **Valider.** Mettre de côté 10 % des données et vérifier les prédictions du modèle par rapport aux données réelles. Une erreur faible signifie que le modèle a capturé la **dynamique** de l'environnement : comment les états changent après les actions.

L'astuce que nous utilisons : au lieu de prédire directement l' `état_suivant`, on prédit le **delta** `état_suivant − état`. La plupart des phénomènes physiques sont incrémentaux ("le chariot a bougé d'un tout petit peu"), et les cibles de petite taille sont plus faciles à traiter pour les réseaux de neurones.

---

## Notre configuration

| Choix | Valeur | Pourquoi |
|-------|--------|-----|
| Environnement | `CartPole-v1` | État 4-D, 2 actions — facile à modéliser. |
| Données | 20 000 transitions d'une politique aléatoire | Large couverture de l'espace d'états. |
| Réseau | MLP, 2 × 128 ReLU cachés | MLP = Multi-Layer Perceptron (réseau de neurones standard). Deux couches cachées de 128 neurones utilisant des activations ReLU. Capacité suffisante, rapide à entraîner. |
| Perte (Loss) | MSE sur `(delta_state, reward)` | MSE = Mean Squared Error (moyenne des erreurs au carré). Perte de régression standard. |
| Optimiseur | Adam, lr = 1e-3, 30 époques | Adam = optimiseur adaptatif (ajuste les taux d'apprentissage par paramètre automatiquement). Standard, donc pas de réglage particulier nécessaire. |

L'ensemble de l'entraînement se termine en quelques secondes sur CPU.

---

## À quoi ressemble un "bon" résultat ?

Deux diagnostics sont importants :

### 1. Précision à une étape (MSE de validation)

Il s'agit de "dans quelle mesure le modèle prédit-il UNE étape dans le futur ?". Après 30 époques, vous devriez voir une MSE de validation dans la plage **1e-4 à 1e-3**. C'est minuscule — les angles du mât et les positions du chariot sont précis à quelques décimales près.

### 2. **Erreur cumulée** sur des rollouts de k étapes

C'est le *vrai* test. Prenez un état, injectez-le dans le modèle, puis prenez sa prédiction et réinjectez-la dans le modèle — pendant `k` étapes consécutives. L'erreur augmente car chaque étape ajoute un peu de bruit par-dessus la prédiction précédente.

```
Étape  1 :  Erreur L2 ≈ 0,01   (presque parfait)
Étape  5 :  Erreur L2 ≈ 0,05
Étape 10 :  Erreur L2 ≈ 0,15
Étape 20 :  Erreur L2 ≈ 0,40   (dérive visible)
```

*(Erreur L2 = distance euclidienne entre l'état suivant prédit et l'état réel — considérez cela comme "à quel point la supposition du modèle est-elle éloignée dans l'espace d'états 4-D ?")*

**Pourquoi est-ce important ?** Si nous planifions 15 étapes à l'avance avec le modèle, l'état *exact* à l'étape 15 sera faux — mais si le classement relatif des "bons plans vs mauvais plans" est préservé, la planification fonctionne toujours. (C'est ce qu'exploite `model_based_planning.py`).

Le graphique dans `outputs/world_model.png` montre les deux diagnostics côte à côte : la courbe de perte d'entraînement descend joliment sur une échelle logarithmique, et la courbe d'erreur de rollout monte joliment.

---

## Pourquoi prédire le *Delta* ?

Comparez deux façons de formuler le même problème au réseau :

| Cible | Magnitude typique | Facile ou difficile ? |
|--------|------------------:|--------------|
| `état_suivant` | 0–2,4 (pos. chariot) | Le réseau doit reproduire la position **et** le minuscule changement. |
| `état_suivant - état` | ~0,02 | Le réseau apprend juste le minuscule changement. |

Prédire le delta signifie également que si le réseau produit des zéros en sortie (comme le fait souvent un réseau débutant non entraîné), la prédiction est simplement "rien n'a bougé" — une valeur par défaut sensée et sûre pour un seul pas de temps. En revanche, prédire directement l' `état_suivant` absolu produirait initialement des valeurs totalement aléatoires, rendant le début de l'entraînement très instable.

---

## Ce que cela nous apporte

Un modèle de monde entraîné est la base pour :

- **La planification** — recherche sur des séquences d'actions imaginées (voir `model_based_planning.py`).
- **L'augmentation de type Dyna** — entraîner un réseau Q sur des données imaginées pour multiplier l'efficacité de l'échantillonnage.
- **La curiosité / l'exploration** — visiter des états que le modèle ne peut pas bien prédire.
- **Les articles Dreamer / World-Models** — entraîner une *politique* entièrement à l'intérieur du modèle avec zéro interaction avec le monde réel au-delà de la collecte initiale de données.

---

## Limites et précautions

- **Dérive hors-distribution.** Le modèle ne connaît que la partie du monde qu'il a vue. Si vous planifiez de manière trop agressive, vous finissez dans des régions que le modèle n'a jamais visitées — les prédictions y sont de la pure fantaisie.
- **Erreur cumulée.** La planification sur de longs **horizons** (beaucoup d'étapes dans le futur) n'est pas fiable en raison de l'accumulation des erreurs, comme le montre le graphique. Les systèmes modernes résolvent ce problème en utilisant des **ensembles probabilistes** (entraîner plusieurs modèles et vérifier s'ils sont d'accord, comme dans PETS ou Dreamer) afin que le planificateur sache exactement *à quel point* le modèle est incertain à chaque étape et puisse éviter les chemins risqués et inconnus.
- **Environnements stochastiques.** Un régresseur déterministe standard ne prédit que le résultat *moyen* et rate complètement l'éventail des résultats possibles. Les environnements complexes du monde réel nécessitent des modèles probabilistes (comme ceux avec des sorties gaussiennes, ou des **modèles stochastiques latents** — des réseaux qui encodent l'état du monde sous forme de distribution de probabilité dans un espace compressé, leur permettant de capturer le véritable caractère aléatoire plutôt que de l'effacer par une moyenne) pour représenter avec précision l'incertitude et l'aléa.

---

## Mots-clés

| Terme | Français simple |
|------|---------------|
| **Modèle de monde** | Un réseau de neurones qui imite l'environnement. |
| **Dynamique** | La fonction `(s, a) → s'` |
| **Modèle de récompense** | La fonction `(s, a) → r` (souvent incluse). |
| **Prédiction à une étape** | Ce que le modèle produit à partir d'un état réel. |
| **Rollout** | Prédictions répétées à une étape, en réinjectant les sorties. |
| **Erreur cumulée** | Petites erreurs qui s'amplifient au cours d'un rollout. |

---

## Résumé en une phrase

> **Un modèle de monde est une minuscule copie neuronale de l'univers que l'agent peut consulter — et dans laquelle il peut rêver — avant de risquer une action réelle.**

Ensuite : `model_based_planning.py` met ce modèle au travail pour la prise de décision réelle.
