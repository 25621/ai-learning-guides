# Fonctions de Valeur d'État (State-Value) 🗺️

## Qu'est-ce qu'un "État" (State) ?

Pensez à un jeu de société. À tout moment, vous vous trouvez sur *une* case du plateau. Cette case est votre **état** — c'est l'endroit où vous vous trouvez en ce moment.

Dans notre jeu de grille 4×4, il y a 16 cases (états). Chaque case est un endroit où l'agent peut se tenir.

---

## Qu'est-ce qu'une "Valeur" (Value) ?

Voici maintenant la question magique : **"Si je me tiens sur cette case en ce moment, quel trésor puis-je espérer collecter avant la fin de la partie ?"**

La réponse est la **valeur** de cet état !

Une case avec une **valeur élevée** signifie : "C'est un excellent endroit — je vais probablement collecter beaucoup de trésors d'ici !"

Une case avec une **valeur faible** signifie : "Oh oh — à partir d'ici, les choses tournent généralement mal."

**Exemple de la vie réelle :** Imaginez que vous jouez à cache-cache. Si vous vous cachez derrière un gros arbre (un excellent endroit), votre chance de gagner est élevée — c'est un état de haute valeur ! Si vous vous cachez au milieu d'une pièce vide, vous serez probablement trouvé — c'est un état de faible valeur.

---

## Notre monde de grille (Grid World)

Voici le plateau de jeu que nous avons utilisé :

```
S  .  .  .      S = Départ (Start)
.  H  .  H      H = Trou (Hole) (récompense -1, fin de partie)
.  .  .  H      G = Objectif (Goal) (récompense +1, fin de partie)
H  .  .  G      . = Case vide sûre
```

- Si vous atteignez **G** (Objectif) : vous obtenez **+1 point** 🎉
- Si vous marchez sur **H** (Trou) : vous obtenez **-1 point** 😢
- Autres pas : **0 point**

Nous avons utilisé γ (gamma) = 0,99, ce qui signifie que les récompenses futures comptent presque autant que les récompenses immédiates. (Un bonbon demain est presque aussi bon qu'un bonbon aujourd'hui !)

---

## Deux plans différents (Politiques)

Nous avons testé deux politiques et calculé la valeur de chaque case pour chacune :

### Politique 1 : Aléatoire uniforme
Choisir au hasard haut, bas, gauche ou droite avec une chance égale.

```
Valeurs (Politique aléatoire uniforme) :
-0,912  -0,932  -0,912  -0,942
-0,929   (H)   -0,898   (H)
-0,901  -0,801  -0,696   (H)
 (H)   -0,630  -0,104   (G)
```

Presque partout est **négatif** — la politique aléatoire tombe si souvent dans les trous qu'être n'importe où est assez mauvais !

---

### Politique 2 : Orientée vers l'objectif
Préférer se déplacer vers la droite et vers le bas (vers l'objectif), tout en allant parfois dans d'autres directions.

```
Valeurs (Politique orientée vers l'objectif) :
-0,838  -0,895  -0,814  -0,961
-0,798   (H)   -0,665   (H)
-0,595  -0,143  -0,213   (H)
 (H)    0,254   0,673   (G)
```

Maintenant, les cases proches de l' **Objectif** ont des **valeurs positives** (0,254 et 0,673) ! La politique intelligente fait de ces cases de bons endroits où se trouver.

---

## Ce que signifient les couleurs sur notre image

Dans notre visualisation :
- **Cases vertes** = haute valeur (excellents endroits où se trouver)
- **Cases rouges** = faible valeur (évitez-les !)
- **Cases jaunes** = quelque part entre les deux

Vous pouvez voir le **gradient** — les valeurs deviennent plus vertes à mesure que vous vous rapprochez de l'objectif et plus rouges près des trous.

---

## Pourquoi nous soucions-nous des valeurs ?

Les valeurs sont le *fondement* de l'apprentissage par renforcement ! Une fois que vous connaissez la valeur de chaque état, vous pouvez prendre des décisions intelligentes :

> "Je suis à la case A. Je peux aller à la case B (valeur = 0,5) ou à la case C (valeur = -0,3).
> J'irai en B — sa valeur est plus élevée !"

C'est exactement ainsi que de nombreux algorithmes de RL (comme le Q-learning) apprennent à prendre de bonnes décisions sans qu'on leur dise les règles.

**Exemple de la vie réelle :** Imaginez que vous choisissez dans quelle file d'attente vous mettre à l'épicerie. Chaque file est un "état". La valeur de cet état est la rapidité avec laquelle vous passerez à la caisse. Vous regardez les files (observez les états) et choisissez celle qui a la valeur la plus élevée (attente la plus courte + moins d'articles).

---

## Comment nous avons calculé les valeurs

Nous avons utilisé l' **Évaluation de politique itérative** (Iterative Policy Evaluation), qui fonctionne comme ceci :

1. Départ : on suppose que toutes les valeurs sont à 0.
2. Mise à jour : pour chaque case, on calcule ce que la valeur *devrait* être en fonction de l'endroit où la politique vous mène ensuite.
3. Répéter jusqu'à ce que les valeurs cessent de changer (convergence).

Mathématiquement : **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(next_state)]**

En français simple : "La valeur de cette case = la récompense moyenne que je vais obtenir tout de suite + une petite partie de la valeur de l'endroit où je vais atterrir."

---

## Mots-clés à retenir

- **État (State)** : Où vous êtes en ce moment (une case sur le plateau).
- **Valeur V(s)** : Récompense totale attendue en partant de l'état s.
- **Politique (Policy)** : Votre plan pour ce qu'il faut faire dans chaque état.
- **Facteur de remise γ (gamma)** : À quel point vous vous souciez des récompenses futures (0,99 = beaucoup !).
- **Évaluation de politique** : Calculer les valeurs pour chaque état sous une politique donnée.

La grande idée : **Certains endroits sont meilleurs que d'others — et la fonction de valeur vous indique exactement à quel point chaque endroit est bon !**
