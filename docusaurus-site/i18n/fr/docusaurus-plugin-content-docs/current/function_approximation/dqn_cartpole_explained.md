# Deep Q-Network (DQN) de zéro 🧠

## Le problème de l'approche linéaire

Vous souvenez-vous de notre formule linéaire précédente ?

> Score = w₁ × position_chariot + w₂ × vitesse_chariot + w₃ × angle_perche + w₄ × vitesse_perche

Cela fonctionne correctement pour CartPole, mais qu'en est-il d'un jeu vidéo où l'on voit des milliers de pixels ? On ne peut pas écrire une recette simple pour cela !

Nous avons besoin de quelque chose capable d'analyser des situations complexes et de déterminer la meilleure action. Ce quelque chose, c'est un **réseau de neurones**.

---

## Qu'est-ce qu'un réseau de neurones ?

Pensez à votre cerveau. Des millions de petites cellules appelées neurones communiquent entre elles. Lorsque vous touchez quelque chose de chaud, les neurones envoient des signaux : "CHAUD ! → Retire la main TOUT DE SUITE !" Chaque neurone transmet l'information, et ensemble ils prennent une décision intelligente.

Un **réseau de neurones informatique** fonctionne de la même manière :

```
Couche d'entrée    Couche cachée 1   Couche cachée 2   Couche de sortie
[pos chariot]   →  [128 neurones] →  [128 neurones] →  [score pousser GAUCHE]
[vitesse char]  →  [  ...       ]    [  ...       ]    [score pousser DROITE]
[angle perche]  →
[vitesse perch] →
```

Chaque flèche possède un **poids** (la force de cette connexion). Il existe des milliers de ces poids — et le réseau les apprend TOUS !

**Exemple de la vie réelle :** Un chef de restaurant goûte votre plat et ajuste des centaines d'ingrédients à la fois. Chaque papille gustative est comme un neurone, et ensemble elles disent au chef "ajoute plus de sel" ou "moins de poivre". L'entraînement du réseau est comme le chef qui apprend au fil de milliers de repas.

---

## DQN = Deep Q-Network

Le **DQN** (Deep Q-Network) a été inventé par DeepMind en 2013. Ils ont repris la vieille formule du Q-learning et ont remplacé la table Q (Q-table) par un réseau de neurones !

Au lieu de :
> Q-table[état][action] = score

Nous avons :
> Q-network(état) → [score_pour_gauche, score_pour_droite]

Le réseau prend l'état en entrée et produit les valeurs Q pour TOUTES les actions à la fois. C'est beaucoup plus efficace que de les calculer séparément !

---

## Ce script : La version "naïve"

Ce script montre le DQN **sans** aucune astuce particulière. Il se contente de :
1. Voir l'état
2. Demander au réseau "est-ce que gauche est bien ? est-ce que droite est bien ?"
3. Effectuer l'action avec le score le plus élevé
4. Recevoir une récompense, mettre à jour le réseau

**C'est volontairement instable !** C'est un peu comme un étudiant qui oublierait immédiatement ses leçons précédentes chaque fois qu'il apprend quelque chose de nouveau. Le réseau se met à jour après chaque étape, ce qui provoque du chaos.

**Exemple de la vie réelle :** Imaginez apprendre à cuisiner en changeant toute votre recette après chaque bouchée. Vous pourriez passer de "trop salé" à "pas de sel du tout" puis à "beaucoup trop salé" sans jamais trouver le bon dosage. C'est ce qui se passe ici !

---

## Ce que vous allez voir

Lorsque vous lancez `dqn_cartpole.py` :
- Les scores peuvent fluctuer énormément (apprentissage instable).
- Parfois, l'agent devient très bon, puis oublie tout d'un coup.
- Le graphique de la perte (loss) montre des variations brutales.

**C'est tout à fait normal !** Cela montre POURQUOI nous avons besoin d'améliorations — comme l'expérience replay et les réseaux cibles (target networks). Ce sont les prochaines étapes !

---

## L'astuce ε-Greedy 🎲

Le robot ne choisit pas toujours la meilleure action. Parfois, il choisit au hasard !

Pourquoi ? Parce que s'il choisit toujours ce qui semble être le mieux, il pourrait ne jamais découvrir de meilleures options.

> Avec une probabilité ε (epsilon) : choisir une action ALÉATOIRE (explorer !)
> Avec une probabilité 1-ε : choisir la MEILLEURE action connue (exploiter !)

Nous commençons avec ε = 1,0 (100 % au hasard) et diminuons lentement jusqu'à ε = 0,01 (1 % au hasard). De cette façon, le robot explore beaucoup au début, puis se concentre sur ce qu'il a appris.

**Exemple de la vie réelle :** En visitant une nouvelle ville, vous pourriez essayer des restaurants au hasard au début (explorer). Après un certain temps, vous retournez dans vos préférés (exploiter). Mais de temps en temps, vous essayez quand même quelque chose de nouveau, juste au cas où il y aurait un trésor caché !

---

## Vocabulaire clé

| Mot | Signification |
|-----|---------------|
| **Réseau de neurones** | Couches de neurones mathématiques connectés qui apprennent à partir de données |
| **Deep** | Plus d'une couche cachée (d'où le terme "deep learning" ou apprentissage profond) |
| **DQN** | Deep Q-Network — utilise un réseau de neurones au lieu d'une Q-table |
| **ε-Greedy** | Stratégie : explorer parfois au hasard, exploiter les meilleures connaissances le reste du temps |
| **Instabilité** | Le réseau "oublie" sans cesse car les mises à jour interfèrent les unes avec les autres |

---

## Ce qui manque (et pourquoi c'est important)

Ce DQN naïf présente deux gros problèmes :

1. **Mises à jour corrélées** : Chaque expérience arrive dans l'ordre (étape 1, étape 2, étape 3...). Si l'étape 5 était mauvaise, TOUTES les mises à jour proches se retrouvent confuses.
   
2. **Cible mouvante** : Après chaque mise à jour, le réseau change. Mais la mise à jour suivante utilise ce MÊME réseau pour calculer ce que la cible devrait être. C'est comme essayer d'atteindre le centre d'une cible qui bouge sans cesse !

Ces problèmes sont résolus par l'**Experience Replay** et les **Réseaux Cibles** dans les scripts suivants. Ensemble, ils transforment le DQN d'un débutant instable en un champion de jeu vidéo !
