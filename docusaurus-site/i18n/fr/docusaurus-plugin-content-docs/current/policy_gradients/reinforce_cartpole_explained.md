# REINFORCE : Apprendre à un Robot à Faire de Meilleurs Choix

## Qu'essayons-nous de faire ?

Imaginez que vous avez un robot qui joue à un jeu vidéo. Chaque seconde, le robot doit choisir :
**"Dois-je appuyer sur le bouton ou non ?"**

Au lieu de mémoriser chaque situation dans un tableau (comme pour le Q-learning), nous voulons que le robot apprenne une **recette** — un ensemble de règles qui dit directement : "Dans cette situation, fais cette action."

Cette recette est appelée une **politique** (π, pi). En apprentissage par renforcement, π signifie "la règle de choix des actions".

---

## L'Ancienne vs la Nouvelle Méthode {#the-old-way-vs-the-new-way}

**Ancienne méthode (Q-learning / DQN) :** Apprendre à quel point chaque action est BONNE (valeurs Q), puis choisir la meilleure.
> "Pousser à GAUCHE a un score de 7, pousser à DROITE a un score de 5 → pousse à GAUCHE !"

**Nouvelle méthode (Gradient de Politique) :** Apprendre directement quelle action CHOISIR.
> "Quand la perche penche à droite, pousse à DROITE avec 80% de chance, pousse à GAUCHE avec 20% de chance."
*(Le mot **Gradient** fait référence au "pas" mathématique que nous faisons pour ajuster lentement ces probabilités dans la bonne direction.)*

**Exemple concret :** Apprendre à faire du vélo.
- L'ancienne méthode : calculer le *score exact* pour "se pencher à gauche de 5 degrés" vs "se pencher à gauche de 7 degrés".
- La nouvelle méthode : s'entraîner jusqu'à ce que votre **corps** apprenne — appuyez sur le pied comme vous le sentez !

---

## Comment fonctionne REINFORCE ?

REINFORCE observe le robot jouer une partie complète du début à la fin (un **épisode**), puis demande : "Quelles actions ont mené à un bon score ? Faisons-en davantage !"

### Étape par Étape

**1. Jouer un épisode**

Le robot fait des choix et accumule de l'expérience :
```
Étape 1 : État = [perche penche à droite] → Action = pousser à DROITE → Récompense = +1
Étape 2 : État = [perche presque en équilibre] → Action = pousser à DROITE → Récompense = +1
Étape 3 : État = [perche penche à gauche] → Action = pousser à GAUCHE → Récompense = +1
...
Étape 47 : État = [la perche est tombée !] → Épisode terminé
```

**2. Calculer les retours**

Pour chaque étape, calculer G_t — la **récompense totale à partir de ce moment-là** :
```
G_à_l_étape_47 = 1
G_à_l_étape_46 = 1 + 0,99 × 1 = 1,99
G_à_l_étape_45 = 1 + 0,99 × 1,99 = 2,97
...
G_à_l_étape_1  = 47 (environ — retour plus élevé car calculé depuis le début)
```

Le **facteur d'actualisation** γ = 0,99 signifie que les récompenses lointaines comptent un peu moins.

**Exemple concret :** Recevoir une étoile dorée le premier jour d'école est plus excitant que de savoir que vous *pourriez* en recevoir une le 100e jour. Les récompenses futures sont légèrement "actualisées".

**3. Mettre à jour la politique**

Pour chaque action entreprise :
> Si G_t était ÉLEVÉ (cette action a mené à un excellent résultat) : **fais-le davantage !**
> Si G_t était BAS (cette action a mené à un mauvais résultat) : **fais-le moins !**

Le calcul : `perte = -log_prob(action) × G_t`

Calculer le gradient et mettre à jour la politique revient à dire au robot :
*"Cette action que tu as faite à l'étape 20 ? Tu devrais la faire 5% plus souvent la prochaine fois !"*

---

## Qu'est-ce qu'un Réseau de Politique ?

Au lieu d'un tableau, nous utilisons un **réseau de neurones** pour représenter la politique.

```
Observation          Réseau de Politique       Probabilités d'Action
[pos chariot]    →   [128 neurones]  →  →  [pousser à GAUCHE : 30%]
[vitesse chariot] →  [128 neurones]        [pousser à DROITE : 70%]
[angle perche]   →
[vitesse perche] →
```

Le réseau produit des **probabilités** pour chaque action. Nous échantillonnons ensuite :
> Lancer un dé → 1-30 : pousser à GAUCHE, 31-100 : pousser à DROITE

**Exemple concret :** Une application météo annonce "70% de chance de pluie". Vous ne SAVEZ PAS s'il va pleuvoir — vous décidez en fonction de la probabilité. Le robot fait la même chose !

---

## Normalisation : Pourquoi soustraire la moyenne

Avant d'utiliser G_t pour la mise à jour, nous normalisons :
```
G_normalisé = (G - moyenne(G)) / ecart_type(G)
```

**Pourquoi ?** Imaginez que toutes les récompenses soient positives (ce qui est le cas dans CartPole — toujours +1 par étape). Sans normalisation, CHAQUE action semble "bonne" et le signal de mise à jour est confus.

Après normalisation, certains retours sont positifs (au-dessus de la moyenne → pousser plus), et d'autres sont négatifs (en dessous de la moyenne → pousser moins). Le signal devient beaucoup plus clair !

**Exemple concret :** Votre professeur note avec une moyenne de classe. Si la moyenne est de 70 et que vous avez eu 85, c'est super ! Mais si la moyenne est de 90 et que vous avez eu 85, c'est en dessous de la moyenne. Le score brut seul ne dit pas tout.

---

## Le Problème : Haute Variance

REINFORCE a une grande faiblesse : la **variance**. Les retours G_t sont très bruités.

**Exemple concret :** Imaginez juger un chef en ne goûtant qu'UN SEUL plat de chaque restaurant. Parfois le chef a passé une mauvaise journée, parfois les ingrédients n'étaient pas frais. Un seul repas ne suffit pas pour savoir de manière fiable si le restaurant est bon !

REINFORCE attend un épisode COMPLET avant de se mettre à jour. Un épisode peut être très chanceux, un autre très malchanceux. Les gradients sautent dans tous les sens.

C'est pourquoi la courbe d'apprentissage (sur le graphique) est en dents de scie :
- Certaines sessions atteignent 500 (génial !)
- D'autres chutent à 50 (terrible !)

Malgré le bruit, REINFORCE finit par apprendre — mais cela demande de la patience.

---

## Les Résultats

```
Épisode  100 | Récompense moy (100 derniers) :  43,1
Épisode  200 | Récompense moy (100 derniers) : 193,9
Épisode  500 | Récompense moy (100 derniers) : 408,4
Épisode 1000 | Récompense moy (100 derniers) : 500,0  ← Résolu !
```

Le robot apprend à maintenir la perche en équilibre pendant le maximum de 500 étapes — RÉSOLU !

Malgré ses problèmes de variance, REINFORCE sur CartPole est efficace car :
1. Les épisodes sont courts (on en obtient donc beaucoup par session d'entraînement)
2. La politique optimale est simple (pousser principalement dans la direction où la perche penche)

---

## Points Clés à Retenir

| Concept | Français Simple |
|---------|---------------|
| **Politique** | La recette du robot pour choisir ses actions |
| **Épisode** | Une partie complète du début à la fin |
| **Retour G_t** | Récompense future totale à partir de ce moment |
| **Remise (Discount) γ** | Les récompenses futures comptent un peu moins que les immédiates |
| **Normalisation** | Soustraire la moyenne pour que le signal soit plus clair |
| **Variance** | L'amplitude des sauts des estimations de gradient |

---

## Prochaine Étape ?

La grande faiblesse de REINFORCE est la **variance**. Dans le script suivant (`reinforce_baseline.py`), nous ajoutons une **ligne de base** (baseline) qui réduit considérablement ce bruit — sans changer ce que l'algorithme apprend en moyenne.

L'idée clé : au lieu de demander "est-ce que cette action était bonne ?", nous demandons "est-ce que cette action était **meilleure que prévu** ?". Ce petit changement rend l'apprentissage beaucoup plus stable.
