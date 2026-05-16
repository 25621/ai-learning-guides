# Dyna-Q : Apprendre plus vite en imaginant 🧠

## De quoi s'agit-il ?

Imaginez une enfant nommée Mia qui apprend à se repérer dans sa nouvelle école. Chaque jour, elle parcourt les couloirs et découvre de nouvelles choses : "La bibliothèque est après la cafétéria", "La salle de M. Smith est à l'étage, près de la cage d'escalier".

Un élève pratiquant le **Q-learning classique** n'apprend que de ce qu'il fait *aujourd'hui*. Si aujourd'hui il est simplement allé de la classe à la cafétéria, il ne met à jour sa mémoire que pour ce trajet spécifique.

Une élève pratiquant le **Dyna-Q** est différente. Après chaque trajet réel, elle s'assoit une minute et **rejoue dans sa tête** plusieurs trajets passés dont elle se souvient. Chaque rejeu renforce sa carte mentale. Après quelques semaines, elle connaît l'école par cœur — non pas parce qu'elle a marché davantage, mais parce qu'elle a **davantage réfléchi à ce qu'elle avait déjà vu**.

C'est exactement ce que Dyna-Q fait pour un agent de RL : il apprend de l'expérience réelle **et** de l'expérience imaginée tirée d'un modèle qu'il construit au fur et à mesure.

---

## Les trois ingrédients

Dyna-Q, c'est "Q-learning + modèle + planification". Une seule étape réelle remplit **trois** fonctions :

1. **RL direct** — la mise à jour habituelle du Q-learning à partir de `(s, a, r, s')`.
2. **Apprentissage du modèle** — noter : "Quand j'ai fait *a* dans l'état *s*, j'ai obtenu *r* et je suis arrivé dans *s'*."
3. **Planification** — choisir *n* paires `(s, a)` au hasard dans la mémoire du modèle et effectuer *n* mises à jour supplémentaires du Q-learning, en **faisant comme si** ces étapes venaient d'avoir lieu.

Cette troisième étape est la clé. Avec `n = 50`, chaque étape réelle dans le monde provoque **51 mises à jour** de la table Q. L'agent apprend environ 50 fois plus vite — en termes d'étapes réelles — qu'un agent pratiquant uniquement le Q-learning.

---

## Schéma de la boucle

```
                   ┌────────────────────────────────────┐
                   │                                    │
   monde réel  ──► faire l'action a ──► observer (r, s')│
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
        Mise à jour     Modèle[s,a] ← (r,s') Planification
        Q-learning                                 ┘
                                     (n mises à jour imaginées)
```

Le modèle est simplement une table de correspondance :
`(état, action) → (récompense, état_suivant)`. Peu coûteux à construire, peu coûteux à interroger.

---

## Exemples de la vie réelle

- **Étude d'échecs.** Les grands maîtres passent des heures à rejouer leurs propres parties et des parties de maîtres dans leur tête. Chaque rejeu est une "planification" — un apprentissage supplémentaire à partir d'expériences déjà vécues.
- **Un musicien pratiquant ses gammes.** Après avoir joué une mesure difficile une fois, il la répète mentalement dix fois de plus avant de continuer. Les doigts ne bougent pas, mais le cerveau se met à jour.
- **Une voiture autonome.** À l'arrêt à un feu rouge, elle rejoue les cent derniers changements de voie en simulation pour affiner sa politique sans user ses pneus.

---

## Ce que fait notre code

Nous utilisons le classique **Dyna Maze** ([Sutton & Barto, Figure 8.2](http://incompleteideas.net/book/the-book.html)) : une grille de 6×9 avec quelques murs, un point de départ `S` au milieu à gauche, et un objectif `G` en haut à droite.

Nous lançons trois variantes, chacune moyennée sur 30 graines aléatoires (seeds) :

| Réglage | Étapes de planification par étape réelle | Signification |
|---------|------------------------------------------|---------------|
| `n = 0` | 0 | Q-learning classique |
| `n = 5` | 5 | un peu de pratique imaginée |
| `n = 50`| 50 | beaucoup de pratique imaginée |

Le script indique le **nombre moyen d'étapes réelles par épisode** au fur et à mesure de l'entraînement. Moins d'étapes signifie que l'agent a appris un chemin plus direct vers l'objectif.

### Ce que vous devriez voir en l'exécutant

Le chemin le plus court dans ce labyrinthe est d'environ 9 étapes ; avec l'exploration ε-greedy, un agent bien entraîné atteint en moyenne 10 étapes par épisode. Au bout de 50 épisodes, les trois réglages convergent vers ce résultat — la différence réside dans la **vitesse de convergence** :

| Réglage | Étapes par épisode (10 derniers eps) | Ce que cela signifie |
|---------|--------------------------------------|----------------------|
| `n = 0` | ~10 | Convergé — mais il a fallu ~30-50 épisodes d'errance pour y arriver |
| `n = 5` | ~10 | Convergé en ~10 épisodes |
| `n = 50`| ~10 | Convergé en ~3-5 épisodes |

Le signal intéressant est la *courbe d'apprentissage*, pas le chiffre final. Le graphique enregistré dans `outputs/dyna_q.png` montre trois courbes plongeant vers le bas à des rythmes très différents : `n = 50` l'atteint en quelques épisodes, tandis que `n = 0` descend beaucoup plus lentement. (Sur un petit labyrinthe déterministe comme celui-ci, le Q-learning classique finit par y arriver — Dyna-Q a simplement besoin de beaucoup moins d'épisodes réels, ce qui est tout l'intérêt dans les environnements où les étapes réelles sont coûteuses.)

---

## Pourquoi cela fonctionne-t-il si bien sur ce labyrinthe ?

Deux raisons :

1. **L'environnement est déterministe.** Chaque couple `(s, a)` donne toujours le même `(r, s')`, donc le modèle est exact après une seule visite. L'expérience imaginée est aussi bonne que l'expérience réelle.
2. **Les étapes réelles sont coûteuses, les étapes imaginées sont gratuites.** Chaque mise à jour imaginée n'est qu'une simple consultation de table, alors qu'une étape réelle oblige l'agent à se déplacer. Lorsque les interactions réelles sont coûteuses (pensez à un vrai robot, un vrai jeu), Dyna-Q est extrêmement efficace en termes d'échantillonnage (sample-efficient).

---

## Les limites de Dyna-Q

- **Environnements stochastiques.** Si un couple `(s, a)` peut mener à de nombreux états `s'` différents, un modèle qui se contente de "se souvenir du dernier résultat" vous mentira. Solution : stocker le nombre de visites ou entraîner un modèle probabiliste.
- **Environnements non stationnaires.** Si le monde change — par exemple, un passage ouvert qui se ferme soudainement ou un raccourci qui apparaît — le modèle devient obsolète et donne de mauvaises prédictions. **Dyna-Q+** résout ce problème en ajoutant un *bonus d'exploration* : les états qui n'ont pas été revisités depuis longtemps reçoivent une petite récompense supplémentaire, incitant l'agent à vérifier si le monde a changé.
- **Grands espaces d'états.** Un dictionnaire utilisant `(s, a)` comme clé ne passe pas à l'échelle pour des millions d'états ou des états continus. C'est exactement le vide que comblent les **modèles du monde appris par réseaux de neurones (learned world models)** — voir `world_model.py` ensuite.

---

## Mots-clés à retenir

| Mot | Signification |
|-----|---------------|
| **Modèle** | Mémoire de `(état, action) → (récompense, état_suivant)` |
| **Étape de planification** | Effectuer une mise à jour Q en utilisant des données imaginées par le modèle |
| **RL direct** | Une mise à jour Q utilisant des données réelles |
| **Efficacité d'échantillonnage (Sample efficiency)** | Mesure l'efficacité avec laquelle un modèle ou un algorithme d'IA utilise les données d'entraînement pour atteindre un certain niveau de performance |
| **Dyna** | L'architecture de Sutton qui entrelace l'apprentissage et la planification |

---

## Résumé en une phrase

> **Dyna-Q apprend en agissant ET en imaginant — et imaginer est gratuit.**

Cette idée, sous sa forme neuronale moderne, alimente certains des agents de RL les plus puissants jamais construits (MuZero, Dreamer, World Models).
