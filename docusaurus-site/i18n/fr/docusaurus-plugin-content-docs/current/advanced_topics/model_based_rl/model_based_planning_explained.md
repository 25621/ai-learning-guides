# Utiliser un Modèle Appris pour la Planification (MPC) 🔮

## L'Idée Maîtresse

Vous disposez d'un **modèle du monde** (un réseau de neurones qui prédit le futur). Et maintenant ?

L'utilisation la plus directe est la **planification** : à chaque instant, demandez au modèle "que se passerait-il si j'essayais *ce* plan ? *cet autre* plan ? et *celui-là* ?" Choisissez ensuite le plan qui semble le meilleur — mais n'en exécutez que **la toute première étape**.

Comme le modèle n'est pas parfait, nous n'exécutons qu'une seule action, nous observons le nouvel état réel issu de l'environnement réel, puis nous recommençons la planification à partir de zéro.

Cette technique a un nom : le **Contrôle Prédictif par Modèle** (ou **MPC** pour *Model Predictive Control*).

---

## Une Analogie Concrète

Vous êtes au restaurant et vous regardez le menu. Vous ne vous engagez pas sur un menu à cinq plats dès le départ — vous commandez l'entrée, vous voyez si vous avez encore faim, puis vous décidez pour le dessert.

Ou encore : vous conduisez sur une route sinueuse. Vous ne fixez pas vos mouvements de volant pour les 30 prochaines secondes — vous regardez constamment devant vous, planifiez quelques secondes à l'avance, effectuez la prochaine action de direction, puis replanifiez.

Cette boucle **planifier loin / agir près / replanifier** est l'essence du MPC.

---

## Comment Fonctionne le "Random Shooting" (Tirage Aléatoire)

Il existe des planificateurs plus sophistiqués — par exemple :
- **CEM** (*Cross-Entropy Method*) : affine itérativement une distribution de plans en ne gardant que les meilleurs à chaque tour.
- **MCTS** (*Monte Carlo Tree Search*) : construit un arbre de recherche guidé par des statistiques de simulation, utilisé par AlphaGo et MuZero.
- **Planificateurs basés sur le gradient** : différencient les prédictions du modèle par rapport aux actions et suivent directement le gradient.

Nous utilisons le plus simple qui fonctionne : le **random shooting** (tirage aléatoire).

```
Étant donné l'état actuel s :
    1. Échantillonner N=200 séquences d'actions aléatoires de longueur H=15.
    2. Pour chaque séquence, la simuler à travers le modèle du monde à partir de s, en sommant une récompense "mise en forme" à chaque étape. (200 rêves en parallèle — très rapide !)
    3. Trouver la séquence avec la récompense totale prédite la plus élevée.
    4. Exécuter la PREMIÈRE action de cette séquence dans l'environnement réel.
    5. Observer le prochain état réel. Ignorer le reste du plan.
    6. Retour à l'étape 1 — replanifier à partir de zéro.
```

200 plans × 15 étapes = 3 000 transitions imaginées par étape réelle. Le modèle du monde les exécute toutes en une seule passe de réseau de neurones (batch) — généralement en quelques millisecondes.

---

## Pourquoi Replanifier à Chaque Étape ?

Parce que le modèle est imparfait. Les erreurs se cumulent au fil d'un déroulement (comme on le voit sur le graphique généré par `world_model.py`, enregistré dans `outputs/world_model.png`). Le plan établi à l'étape 0 n'est fiable que pour les premiers mouvements ; à l'étape 15, le modèle hallucine. Nous ne faisons donc confiance qu'au **premier mouvement**, puis nous rafraîchissons le plan avec le dernier état réel.

C'est la même raison pour laquelle les humains n'écrivent pas un plan de 100 coups aux échecs pour s'y tenir — les circonstances changent, et plus vous essayez de deviner loin, moins cela correspond à la réalité.

---

## Une Subtilité : La Récompense Doit Guider le Planificateur

Dans CartPole, la récompense réelle est `+1` à chaque étape jusqu'à ce que la perche tombe. Le modèle prédira fidèlement `+1, +1, +1, ...` pour presque tous les plans, car les plans aléatoires se terminent rarement rapidement à l'intérieur du modèle — et ainsi, chaque plan obtient le même score. Le planificateur ne sait pas quoi choisir.

La solution : remplacer la récompense réelle par un **proxy fluide** pendant la planification :

```python
reward_proxy(état) = 1
                    - |angle_perche| / 0.21          # perche verticale ? (1=oui)
                    - 0.1 * |position_chariot| / 2.4  # chariot centré ? (1=oui)
```

Désormais, les plans qui *pourraient* mener à la chute de la perche obtiennent des scores nettement inférieurs à ceux qui restent verticaux. Le planificateur peut les classer.

> **Leçon concrète.** Un signal de récompense binaire ou plat — "tu as survécu une seconde de plus" — est inutile pour la planification à court terme. Des signaux denses et mis en forme (*shaped*) sont indispensables.

---

## Ce que Fait Notre Code

`model_based_planning.py` :

1. **Charge** les poids du modèle du monde enregistrés par `world_model.py`. (S'ils manquent, il en réentraîne un à la volée.)
2. **Exécute 20 épisodes** de MPC sur le vrai CartPole-v1.
3. **Exécute également 20 épisodes** avec une politique uniformément aléatoire, comme base de comparaison (baseline).
4. **Trace** les deux côte à côte et affiche les moyennes.

### Ce que vous devriez voir lors de l'exécution

| Politique | Récompense moyenne (étapes survécues) |
|--------|-------------------------------:|
| Aléatoire (Random) | ~22 (typique pour CartPole — la perche tombe vite) |
| MPC (le nôtre) | ~150–500 (varie selon la graine ; beaucoup d'épisodes près de 500) |
| Maximum possible | 500 |

Cette **amélioration de 5 à 25 fois** est obtenue sans réseau de politique, sans fonction de valeur et sans entraînement supplémentaire. Juste un modèle du monde + 200 rêves par étape.

Le graphique `outputs/model_based_planning.png` montre deux barres colorées par épisode — le MPC est presque toujours plus haut que l'Aléatoire, avec de nombreux épisodes atteignant le plafond des 500 étapes.

---

## Points Forts de la Planification Basée sur un Modèle

- **Efficacité d'échantillonnage.** Tout l'apprentissage a été réalisé à partir d'un seul lot de transitions aléatoires. Aucune interaction supplémentaire avec l'environnement n'a été nécessaire pour dériver une politique utile.
- **Facile à réorienter.** Vous voulez contrôler l'agent différemment ? Changez le proxy de récompense — pas besoin de réentraînement. (Essayez de maximiser la vitesse du chariot pour vous amuser.)
- **Interprétable.** Vous pouvez inspecter les plans envisagés par l'agent, les trajectoires prédites et les scores.

## Faiblesses (et Comment y Remédier)

- **Le tirage aléatoire est basique.** Il échantillonne les plans aveuglément. Pour des dimensions plus élevées, on passe au **CEM** (voir plus haut), à l'**iLQR** (qui approxime le modèle comme localement linéaire) ou à des planificateurs basés sur le gradient.
- **Cumul d'erreurs du modèle.** L'horizon long dérive. On utilise des **ensembles probabilistes** (plusieurs modèles entraînés sur les mêmes données, comme dans PETS) pour que le planificateur puisse pénaliser les plans où les modèles ne sont pas d'accord.
- **C'est la récompense réelle qui compte, au final.** La mise en forme de la récompense aide, mais pour des tâches plus complexes, on apprend une **fonction de valeur** *à l'intérieur* du modèle du monde — un critique appris qui estime le retour à long terme. **Dreamer** et **MuZero** utilisent cette idée.

---

## Lien avec les Systèmes Modernes

La recette exacte que vous venez d'exécuter — **dynamique apprise + planification** — est la colonne vertébrale de certains des systèmes de RL les plus puissants :

- **MuZero** (DeepMind) : combine un modèle du monde appris avec la recherche arborescente Monte Carlo (MCTS). Il a maîtrisé le Go, les échecs et Atari sans connaître les règles au préalable.
- **Dreamer / DreamerV3** (Hafner et al.) : entraîne une politique *à l'intérieur* d'un modèle du monde en **espace latent** (le modèle compresse les images brutes en une représentation abstraite avant de prédire). Il atteint des performances de pointe sur plus de 100 benchmarks.
- **PETS / PlaNet / TD-MPC** : familles d'algorithmes qui adaptent cette idée à des tâches de contrôle continu complexes comme la robotique.

Vous venez de construire — en quelques centaines de lignes — le plus petit membre de cette famille.

---

## Mots Clés

| Terme | Définition simple |
|------|---------------|
| **MPC** | Contrôle Prédictif par Modèle — planifier, agir une fois, replanifier |
| **Random shooting** | Évaluer plein de plans aléatoires, garder le meilleur |
| **Horizon (H)** | Nombre d'étapes de prédiction du plan |
| **N échantillons** | Nombre de plans candidats examinés à chaque étape |
| **Horizon fuyant** | Replanifier à chaque étape plutôt que de s'en tenir à un plan long |
| **Proxy de récompense** | Récompense intermédiaire fluide pour guider le planificateur |

---

## Résumé en une phrase

> **Une fois que vous avez un modèle du monde, la planification consiste simplement à "rêver cent futurs, choisir la meilleure première étape, et recommencer."**

C'est tout le secret du RL basé sur un modèle.
