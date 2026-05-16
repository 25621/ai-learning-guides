# SARSA vs Q-Learning : Chemins sûrs vs Chemins optimaux 🐢 vs 🐇

## Qu'est-ce que c'est ?

Deux robots doivent tous deux marcher le long d'un **bord de falaise** pour atteindre l'objectif. Les deux robots sont encore en phase d' *apprentissage* et font parfois des mouvements aléatoires (oups !).

- 🐢 **Robot SARSA** : "Je sais que je vacille parfois... donc je vais marcher loin de la falaise pour être en sécurité, même si cela prend plus de temps."
- 🐇 **Robot Q-Learning** : "Le chemin le plus court rase la falaise — allons-y ! (Tombe parfois pendant l'apprentissage, mais finit par apprendre le meilleur itinéraire.)"

Les deux robots sont intelligents, mais ils font un **compromis différent** : sûr-mais-plus-lent vs optimal-mais-risqué-pendant-l'apprentissage.

---

## La différence clé : Quelle "Action Suivante" utilisez-vous ?

Lors de la mise à jour des scores après chaque étape, les deux algorithmes demandent :
> "Quelle est la valeur de l' *état suivant* ?"

| Algorithme | Utilise l'action suivante... | On-policy ? |
|------------|-----------------------------|-------------|
| **SARSA** | ...que je vais *réellement prendre* (peut-être aléatoire !) | Oui |
| **Q-Learning** | ...qui est *théoriquement la meilleure* (toujours gourmande/greedy) | Non |

**Exemple de la vie réelle :** Deux enfants apprenant à faire du vélo.

- **Enfant SARSA** : Reste près de l'herbe parce qu' *il sait* qu'il vacille parfois de manière aléatoire. Il planifie en fonction de son moi vacillant actuel.
- **Enfant Q-Learning** : S'entraîne au milieu du chemin parce qu'il imagine un futur moi parfait qui ne vacille jamais. Il tombe parfois maintenant, mais apprend le meilleur chemin plus rapidement.

Les deux enfants finissent par apprendre — mais pendant l'entraînement, l'enfant SARSA tombe moins !

---

## Ce que notre code a trouvé

Les deux algorithmes ont fonctionné pendant **500 épisodes** sur Cliff Walking avec ε=0,1 (ε = epsilon ; ici, cela signifie 10 % de chances de faire un mouvement aléatoire) :

| Métrique | SARSA | Q-Learning |
|----------|-------|------------|
| Récompense moyenne pendant l'entraînement (50 derniers ép.) | **-19,7** | **-51,0** |
| Évaluation gourmande (sans exploration) | -17 | **-13** |

- **Pendant l'entraînement** : SARSA obtient de **bien meilleures récompenses** car il évite la falaise (en tenant compte de ses propres mouvements aléatoires).
- **Après l'entraînement** (purement gourmand) : Q-Learning trouve le **chemin optimal plus court** (-13) !

À mesure que ε tend vers 0, les deux algorithmes convergent vers la même politique optimale.

---

## Résumé visuel

```
Chemin SARSA (pendant l'entraînement) :   Chemin Q-Learning (gourmand, après l'entraînement) :
[ ][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]   [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][C][C][C][C][C][C][C][C][C][C][G]   [S][→][→][→][→][→][→][→][→][→][→][G]
     (détour sûr, rangées du haut)           (optimal, rase la falaise)
```

---

## Exemples de la vie réelle

- **Chirurgien débutant vs chirurgien expérimenté** : Le chirurgien débutant (SARSA) reste à l'écart des techniques risquées pendant son apprentissage. Le chirurgien expérimenté (Q-Learning gourmand) utilise la technique la plus efficace après l'avoir maîtrisée.
- **Conduite en ville vs itinéraire d'autoroute** : Une planification de type SARSA emprunte des rues résidentielles plus sûres ; Q-Learning trouve l'autoroute optimale mais étroite.
- **Étudiant qui étudie** : L'étudiant-SARSA s'en tient aux sujets bien compris pendant la pratique. L'étudiant-Q-Learning s'attaque aux problèmes les plus difficiles (échoue davantage) mais apprend la stratégie optimale.

---

## Mots-clés à retenir

- **On-policy** (SARSA) : Apprend de ce que vous *faites réellement*, y compris l'exploration aléatoire.
- **Off-policy** (Q-Learning) : Apprend le *meilleur comportement possible* séparément de ce que vous faites réellement.
- **Chemin sûr** : Itinéraire plus long qui évite le danger, utilisé lorsque l'exploration provoque des accidents.
- **Chemin optimal** : Itinéraire le plus court/le plus récompensé, trouvé lorsqu'il n'y a plus d'exploration.
- **Compromis exploration-exploitation** : L'équilibre entre essayer de nouvelles choses et utiliser ce que vous savez.

La grande idée : **SARSA est plus sûr pendant l'entraînement (on-policy), Q-Learning trouve le chemin optimal plus rapidement (off-policy). Lequel est le meilleur dépend de l'importance de tomber de la falaise !**
