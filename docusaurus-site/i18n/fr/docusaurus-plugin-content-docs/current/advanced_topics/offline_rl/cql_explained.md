# Apprentissage Q Conservateur (Conservative Q-Learning - CQL) 🛡️

## Qu'est-ce que c'est ?

Imaginez que vous appreniez à investir de l'argent en lisant un registre géant des transactions boursières passées effectuées par d'autres personnes. Le registre contient des achats, des ventes et des conservations — mais **aucune trace de transaction que personne n'a réellement faite**.

Imaginez maintenant qu'un étudiant trop confiant regarde le registre et dise :
*"Et si quelqu'un avait acheté des tickets de loterie tous les lundis ? Ça aurait été une transaction incroyable !"*

Le problème : **le registre ne contient aucune donnée sur l'achat de loterie le lundi**, donc l'étudiant est en pleine hallucination. Pourtant, cette transaction hallucinante semble excellente sur le papier, donc la "politique" de l'étudiant veut absolument la réaliser.

Ce problème d'hallucination s'appelle le **décalage de distribution (distribution shift)** : un apprenant hors ligne (offline) adore les actions que l'ensemble de données n'a jamais testées, car il n'y a aucune donnée pour contredire son optimisme. CQL est le remède à ce problème.

---

## Pourquoi le Q-Learning échoue-t-il hors ligne ?

La cible (target) normale du Q-learning est :

```
cible(s, a) = r + γ · max_{a'} Q(s', a')
```

Ce `max_{a'}` est le danger. Lorsque l'ensemble de données n'a jamais enregistré l'action `a'` dans l'état `s'`, le réseau se contente de *deviner* une valeur Q — et les réseaux de neurones ont tendance à **surestimer** Q pour les entrées jamais vues. La cible hérite de cette surestimation, le réseau apprend à prédire ce nombre plus grand, et à l'étape suivante, nous **extrapolons** (projetons encore plus loin au-delà de ce que les données supportent) des valeurs encore plus élevées. La politique poursuit un fantôme.

Si vous pouviez continuer à collecter plus de données, cela s'auto-corrigerait (l'action fantôme s'avérerait mauvaise en réalité). Mais **en RL hors ligne, vous ne pouvez pas collecter plus de données.** Le fantôme est éternel.

---

## L'astuce de CQL

CQL (Kumar et al., 2020) ajoute un terme de pénalité à la perte (loss) :

```
perte_cql(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_ensemble_donnees)
```

Deux éléments :

1. **`log Σ_a exp Q(s, a)`** (lire : *"log-sum-exp sur toutes les actions"*) est un **maximum doux (soft maximum)** sur toutes les actions — une approximation lisse et dérivable de `max` qui prend en compte chaque action en même temps plutôt que d'en sélectionner une seule de manière rigide. Le pénaliser réduit les valeurs Q **de manière généralisée** (en poussant toutes les prédictions vers le bas uniformément) — surtout pour les actions ayant le Q le plus *élevé*, ce qui est exactement là où se trouvent les hallucinations.
2. **`- Q(s, a_ensemble_donnees)`** récompense un Q élevé pour l'action que l'ensemble de données a réellement enregistrée — protégeant les valeurs Q "dans la distribution" (in-distribution) de la réduction mentionnée ci-dessus.

Effet net : **Q est tiré vers le bas pour les actions non vues, et tiré vers le haut pour les actions vues.** Le Q appris devient une *borne inférieure* (lower bound) du vrai Q. La politique **`argmax`** (la règle qui choisit simplement l'action avec le Q le plus élevé) arrête de poursuivre des fantômes.

Perte totale :

```
L  =  Bellman_MSE   +   α · perte_cql
```

(Où **`Bellman_MSE`** est l'erreur standard du Q-learning normal, mesurant à quel point la supposition actuelle du réseau est en désaccord avec sa propre supposition future).

`α` est le bouton de réglage du conservatisme. Trop petit → le décalage de distribution réapparaît. Trop grand → l'agent est si conservateur qu'il ne s'améliore jamais au-delà des données.

---

## Exemples concrets

- **Un entraîneur d'échecs conservateur.** Vous ne pouvez apprendre qu'à partir de parties déjà jouées. Un entraîneur imprudent dirait : "ce coup hypothétique sans précédent pourrait être brillant !" CQL est l'entraîneur qui dit : "nous n'avons aucune donnée là-dessus — tenons-nous en aux coups que les vrais joueurs ont essayés."
- **Choix de menu au restaurant.** Les avis Yelp ne couvrent jamais les plats hors menu. Une politique naïve recommanderait les plats hors menu sur la base de notes cinq étoiles hallucinées. CQL recommande uniquement ce qui a été commandé assez souvent pour être fiable.
- **Saisie de robot à partir de journaux.** Le robot dispose de vidéos de saisie de tasses, de bouteilles et de livres — mais jamais d'un couteau. CQL refuse de recommander avec assurance "saisir le couteau par la lame".

---

## Ce que fait notre code

Le script `cql.py` :

1. **Charge les quatre ensembles de données** créés par `d4rl_dataset.py`.
2. **Choisit `medium-replay`** comme ensemble d'entraînement — c'est le plus réaliste (qualité mixte) et le plus dommageable pour les méthodes naïves.
3. **Entraîne trois agents purement hors ligne**, dans des conditions identiques sauf pour `α` :
   - `α = 0`   →  DQN hors ligne naïf (pas de pénalité — la ligne de base défaillante)
   - `α = 1.0` →  CQL léger
   - `α = 5.0` →  CQL fort
4. **Évalue chacun toutes les 2 500 étapes de gradient** en testant l'agent dans le véritable environnement (10 épisodes). C'est le *seul* contact avec l'environnement ; l'entraînement lui-même ne voit jamais l'environnement.
5. **Trace les courbes d'apprentissage** dans `outputs/cql.png`.

---

## Ce que vous devriez voir

Une exécution typique affiche quelque chose comme :

```
Final evaluation returns (avg over 10 episodes, greedy):
  naive offline DQN (alpha=0)         ->  ~30-150  (instable ; échoue souvent)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

Dans le graphique des courbes d'apprentissage :

- La **courbe rouge** (`α = 0`) grimpe tôt puis **s'effondre souvent** une fois que les hallucinations dues au décalage de distribution infectent la **cible de Bellman (Bellman target)** (le nombre que nous utilisons comme "réponse correcte" lors de l'entraînement du réseau Q : `r + γ · max Q(s', ·)`). Lorsque des valeurs Q fantômes polluent cette cible, chaque étape de gradient aggrave les choses. La **perte de Bellman (Bellman loss)** (l'erreur quadratique moyenne entre la prédiction du réseau Q et la cible de Bellman) semble correcte — c'est là toute la **traîtrise** du problème : le réseau est parfaitement cohérent avec ses propres croyances erronées, de sorte que la perte ne donne aucun avertissement.
- La **courbe orange** (`α = 1.0`) grimpe plus lentement mais **reste stable**.
- La **courbe verte** (`α = 5.0`) est la plus stable et généralement la meilleure.

Le panneau de la perte de Bellman montre un autre signe révélateur : la perte de DQN naïf peut rester faible alors que sa politique est terrible, car le réseau est intérieurement cohérent avec ses propres hallucinations.

---

## La place de CQL dans le domaine

CQL a été une étape *majeure* car il a apporté une solution simple et fondée au décalage de distribution. La lignée :

```
DQN (en ligne)
   │
   ▼
DQN hors ligne naïf ── échoue à cause du décalage de distribution
   │
   ▼
CQL (Kumar 2020)     ── ajoute une pénalité conservatrice : Q est une borne inférieure
   │
   ▼
IQL (Kostrikov 2021) ── ne questionne jamais Q sur des actions non vues au départ
   │
   ▼
Decision Transformer (Chen 2021) ── ignore complètement Q, traite le RL comme une modélisation de séquence
                                     (prédit l'*action suivante* compte tenu des états passés et
                                      d'un retour total souhaité, exactement comme un LLM
                                      prédit le mot suivant)
```

Chaque étape de cette lignée est une réponse différente à la même question : **comment éviter de demander à mon réseau Q des choses qu'il n'a jamais vues ?**

---

## Mots-clés à retenir

| Mot | Signification |
|------|---------|
| **Décalage de distribution (Distribution shift)** | La politique entraînée veut des actions en dehors des données |
| **Hors-distribution (Out-of-distribution - OOD)** | Une paire (s, a) que l'ensemble de données n'a jamais enregistrée |
| **Vrai Q** | Le retour futur attendu réel pour l'action `a` dans l'état `s`, si on pouvait le mesurer parfaitement |
| **Q conservateur** | Une fonction Q apprise qui essaie de rester sous le vrai Q au lieu de sur-promettre |
| **Logsumexp** | Une approximation lisse et dérivable de `max` |
| **Alpha (α)** | Le bouton de conservatisme de CQL — force avec laquelle on pousse Q vers le bas sur les actions OOD |

---

## Résumé en une phrase

> **CQL ajoute une "pénalité de pessimisme" qui punit les valeurs Q élevées sur les actions que l'ensemble de données n'a jamais essayées — afin que la politique ne tombe pas amoureuse d'hallucinations.**
