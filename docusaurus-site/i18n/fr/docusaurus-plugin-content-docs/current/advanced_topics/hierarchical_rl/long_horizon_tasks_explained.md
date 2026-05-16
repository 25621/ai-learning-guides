# Tâches à Horizon Long

## L'Idée Maîtresse : Quand la Récompense est Très Lointaine

Imaginez que vous êtes un chef cuisinier essayant d'apprendre une nouvelle recette uniquement en goûtant le plat final. Vous suivez 40 étapes — couper, faire revenir, assaisonner, mijoter, dresser — mais vous ne recevez de feedback qu'à la toute fin : "Trop salé." Laquelle des 40 étapes a causé le problème ? Vous n'en avez aucune idée.

C'est le **problème de l'horizon long** : lorsque le signal de récompense est séparé des décisions qui l'ont provoqué par des dizaines (ou des centaines) d'étapes, l'apprentissage devient très difficile.

---

## Pourquoi les Agents "Plats" Échouent

Un agent RL "plat" (comme les agents DQN de la Phase 3) essaie d'apprendre la valeur de chaque étape individuelle d'un seul coup. Dans les tâches courtes — équilibrer une perche, éviter un mur — cela fonctionne bien. La récompense arrive rapidement, et l'agent peut relier la cause à l'effet.

Mais dans une tâche longue — ramasser une clé, puis l'utiliser pour ouvrir une porte, puis sortir du labyrinthe — l'agent doit :

1. Tomber sur la clé par hasard (quelle chance !)
2. Se souvenir que ramasser des clés est utile
3. Tomber sur la porte par hasard (encore de la chance !)
4. Relier toute la séquence à la récompense unique à la sortie

Avec une exploration aléatoire, la probabilité de compléter accidentellement toute cette séquence diminue de manière exponentielle avec chaque nouvelle étape requise. Le DQN plat doit essentiellement avoir de la chance de très nombreuses fois avant de voir une seule récompense positive dont il peut tirer un enseignement.

---

## La Solution Hiérarchique : Diviser pour Régner

Le RL Hiérarchique décompose la tâche longue en une **structure à deux niveaux** :

| Niveau | Nom | Rôle |
|-------|--------|-----|
| Haut | **Gestionnaire (Manager)** | Choisit le prochain sous-objectif |
| Bas  | **Ouvrier (Worker)** | Navigue vers ce sous-objectif |

C'est exactement ainsi que les humains abordent les tâches complexes. Vous ne planifiez pas votre voyage en voiture virage par virage avant de partir. Au lieu de cela :

- **Gestionnaire (vous, à la maison) :** "Premier arrêt : la station-service. Prochain arrêt : l'entrée de l'autoroute. Ensuite : sortie 42."
- **Ouvrier (vous, au volant) :** Gère toutes les décisions de direction individuelles pour atteindre chaque arrêt.

Le gestionnaire pense en *points de passage*. L'ouvrier pense en *coups de volant*.

---

## Pourquoi cela l'emporte sur l'apprentissage plat pour les tâches longues

L'ouvrier n'a besoin d'atteindre que le *prochain sous-objectif* — une tâche courte avec une récompense claire et proche. Il reçoit un feedback rapidement et apprend efficacement.

Le gestionnaire n'a besoin que de décider de l'*ordre des sous-objectifs* — un problème beaucoup plus simple que de planifier chaque étape individuelle.

Ensemble, les deux niveaux divisent le problème difficile de l'horizon long en deux problèmes faciles de l'horizon court.

---

## L'Expérience de la Grille Clé-Porte

Notre script teste les deux approches sur une **grille ouverte de 9x9** avec deux objets :

- Une **CLÉ** à un coin (doit être ramassée en premier).
- Une **PORTE** au coin opposé (ne compte que si vous avez la clé).

La seule récompense réelle est +1 lorsque l'agent atteint la porte *après* avoir ramassé la clé. Cette récompense unique nécessite que deux sous-tâches séquentielles soient enchaînées correctement.

Deux agents s'affrontent :

**DQN Plat :** Doit tomber par hasard sur les deux sous-tâches dans le bon ordre, puis propager un signal à travers les deux. Parce que le succès nécessite deux découvertes chanceuses dans un seul épisode, le DQN apprend rarement quoi que ce soit d'utile.

**Agent Hiérarchique :**
- Règle du Gestionnaire : "Aller à la clé d'abord, puis aller à la porte."
- L'ouvrier reçoit **+1 chaque fois qu'il atteint un sous-objectif** — qu'il s'agisse de la clé ou de la porte.
- Deux tâches courtes distinctes, chacune avec une récompense claire à proximité.

---

## Ce que Montrent les Graphiques

![Résultats de la Tâche à Horizon Long](outputs/long_horizon_tasks.png)

**À gauche — Taux de réussite au fil du temps :** L'agent hiérarchique (bleu) apprend à résoudre le labyrinthe bien avant le DQN plat (rouge). L'agent plat peut finir par apprendre aussi — avec suffisamment d'épisodes — mais l'agent hiérarchique y parvient plus vite car son signal d'apprentissage est dense et local.

**À droite — Performance finale :** L'histogramme montre le taux de réussite moyen sur les 500 derniers épisodes. L'avantage de l'agent hiérarchique est clair : la décomposition du problème en sous-objectifs le rend gérable.

---

## Où se manifeste la pensée à horizon long

| Domaine | Exemple d'horizon long |
|--------|---------------------|
| Robotique | Assembler un appareil avec 30 pièces dans l'ordre |
| Jeux | Gagner une partie d'échecs (beaucoup de coups, un seul gagnant) |
| Langage | Écrire un document de recherche complet (beaucoup de décisions d'écriture, un seul score de qualité) |
| Science | Mener une expérience de plusieurs mois et évaluer les résultats |

C'est précisément pourquoi les *Feudal Networks* (une architecture où les gestionnaires fixent des objectifs directionnels pour les ouvriers de niveau inférieur) et HIRO (*Hierarchical RL with subgoals*) ont été inventés — alors que le RL plat se heurtait à un mur sur ces problèmes, la décomposition hiérarchique est devenue la stratégie dominante.

---

## Le Lien avec les Politiques Conditionnées par des Objectifs

Remarquez que l'**ouvrier** dans notre agent hiérarchique est essentiellement une **politique conditionnée par des objectifs** — il reçoit un sous-objectif et navigue vers lui. C'est la conception standard dans HIRO et les articles connexes : le gestionnaire fixe des objectifs, l'ouvrier est une politique conditionnée par des objectifs qui les poursuit.

Les deux idées — politiques conditionnées par des objectifs et structure hiérarchique — sont donc les deux faces d'une même pièce, c'est pourquoi elles apparaissent ensemble dans ce module.

---

## Résumé en une phrase

> **Les tâches à horizon long sont difficiles parce que la récompense arrive trop tard pour enseigner les décisions individuelles — le RL hiérarchique résout ce problème en insérant des sous-objectifs proches qui permettent à l'ouvrier d'apprendre rapidement tandis que le gestionnaire s'occupe de la séquence globale.**
