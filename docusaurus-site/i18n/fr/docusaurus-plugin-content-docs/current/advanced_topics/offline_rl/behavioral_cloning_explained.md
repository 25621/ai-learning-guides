# Clonage Comportemental (Behavioral Cloning - BC) 🐒

## Qu'est-ce que c'est ?

Imaginez que vous vouliez apprendre à jouer au tennis. Vous regardez des centaines d'heures de matchs enregistrés de Wimbledon et vous vous contentez de **copier ce que font les joueurs**. Vous ne vous demandez pas si leur coup était le *meilleur* possible — vous adaptez simplement la position de votre corps à la leur et balancez la raquette de la même manière.

C'est le clonage comportemental. **Pas de récompense. Pas de planification. Juste de l'imitation.**

En termes d'apprentissage par renforcement (RL) : prenez l'ensemble de données composé de paires `(état, action)` et entraînez un réseau de neurones à prédire l'action à partir de l'état, exactement comme un modèle de classification d'images prédit s'il s'agit d'un chat ou d'un chien. L'"étiquette" (label) est simplement l'action entreprise par le collecteur de données.

---

## En quoi cela diffère-t-il du "vrai" RL hors ligne (Offline RL) ?

| Approche | Utilise des récompenses ? | Peut dépasser les données ? |
|----------|---------------|---------------------|
| **BC**   | ❌ non         | ❌ non — au mieux, il égale la qualité moyenne des données |
| **CQL** (et consorts) | ✅ oui | ✅ oui — peut assembler de bonnes transitions à partir de données mixtes |

Le BC est la vision "apprentissage supervisé" du RL. C'est incroyablement simple, souvent étonnamment efficace, et c'est la ligne de base (baseline) universelle. **Si un algorithme de RL hors ligne ne peut pas battre le BC sur le même ensemble de données, c'est qu'il n'a rien accompli.**

---

## Exemples concrets

- **Apprendre à conduire à partir d'images de caméra de tableau de bord (dashcam).** Regarder la route, prédire l'angle du volant utilisé par l'humain. Deux exemples historiques :
  - **ALVINN (1989)** — le tout premier conducteur par réseau de neurones ; un minuscule réseau à 3 couches entraîné sur des entrées de caméra + laser pour diriger une camionnette sur l'autoroute.
  - **NVIDIA PilotNet (2016)** — un CNN profond moderne entraîné de bout en bout sur des images de dashcam ; il a appris le maintien de la trajectoire et les manœuvres de base uniquement en imitant des conducteurs humains, sans aucune règle programmée à la main.
- **Un apprenti copiant un chef cuisinier.** "Tout ce que fait le chef, je le fais." Cela fonctionne très bien si le chef est excellent ; cela produit un mauvais chef si le chef est médiocre.
- **GitHub Copilot.** L'autocomplétion est entraînée à prédire "quel code un humain taperait-il ensuite ?" — de l'imitation pure de journaux de codes sources.
- **Imiter son grand frère ou sa grande sœur.** Les enfants font cela pendant des années avant de commencer à raisonner sur le *pourquoi* de ces actions.

---

## Les mathématiques (en une ligne)

Pour chaque `(s, a)` dans l'ensemble de données, minimisez :

```
perte = -log π(a | s)        (entropie croisée)
```

C'est tout. La politique `π` est juste un MLP (Multi-Layer Perceptron) qui produit des logits d'action ; l'entraînement est identique à MNIST. Décomposons le jargon :
- **`π` (Pi) :** Le symbole standard pour la "politique" (policy) — la règle ou le réseau de neurones qui décide quoi faire.
- **MLP (Multi-Layer Perceptron) :** Un réseau de neurones standard de base.
- **Logits :** Les scores bruts et non normalisés que le réseau produit avant de les transformer en probabilités.
- **Entropie croisée (Cross-entropy) :** La formule standard pour pénaliser un modèle lorsqu'il attribue une faible probabilité à la réponse correcte.
- **MNIST :** Le célèbre ensemble de données pour débutants composé de chiffres écrits à la main.

Entraîner un agent à jouer à un jeu via le BC est littéralement identique à entraîner un réseau à reconnaître des chiffres manuscrits dans MNIST. Dans MNIST, l'entrée est une image et la sortie est un chiffre (0-9). Dans le BC, l'entrée est l'état du jeu et la sortie est l'action (ex: "aller à gauche").

---

## Ce que fait notre code

Le script `behavioral_cloning.py` :

1. **Charge les quatre ensembles de données** créés par `d4rl_dataset.py`
   (`random`, `medium`, `expert`, `medium-replay`).
2. Pour chaque ensemble de données, **entraîne une politique de BC distincte** pendant 10 000 étapes de gradient d'entropie croisée. La colonne des récompenses est totalement ignorée.
3. Toutes les 2 500 étapes, **évalue** la politique actuelle en la testant dans le véritable environnement CartPole-v1 (moyenne sur 20 épisodes).
4. Génère des graphiques :
   - Un diagramme en barres : score final du BC par ensemble de données.
   - Une courbe d'apprentissage : progression de chaque variante de BC au cours de l'entraînement.

---

## Ce que vous devriez voir

Une exécution typique affiche :

```
Final evaluation returns:
  BC on random          ->    ~20  ± quelques-uns   (≈ jeu aléatoire)
  BC on medium          ->   ~150  ± grand écart   (≈ la politique medium)
  BC on expert          ->   ~480  ± petit écart   (≈ la politique expert)
  BC on medium-replay   ->    ~60  ± grand écart   (≈ la MOYENNE des données mixtes)
```

Le graphique en barres rend l'histoire évidente : **le score du BC suit le score moyen de l'ensemble de données.** Il ne peut pas dépasser ce plafond car il n'a aucun moyen de préférer les "bonnes" parties d'un ensemble de données mixte aux "mauvaises" — les deux sont des cibles d'imitation également valables.

C'est la conclusion : **le BC hérite du plafond des données.**

---

## BC vs CQL — La comparaison la plus claire

Sur l'ensemble de données **medium-replay** (le cas le plus réaliste, de qualité mixte) :

| Méthode | Score final approx. | Pourquoi ? |
|--------|--------------------:|------|
| BC     | ~60   | Imite la *moyenne* des premières tentatives ratées + des bonnes tentatives ultérieures |
| CQL    | ~400+ | Utilise les récompenses pour préférer les transitions à haut Q ; assemble une bonne politique à partir de données mixtes |

Ainsi, CQL **bat les données**, tandis que le BC **égale les données**. C'est toute la raison pour laquelle le RL hors ligne est un domaine de recherche et ne se résume pas à "faire de l'apprentissage par imitation". Lorsque les données sont de qualité mixte (ce qui est toujours le cas pour les journaux réels), les méthodes tenant compte des récompenses récupèrent plus de valeur.

Sur des données **expert**, la comparaison s'inverse : le BC égale l'expert (~480). Vous pourriez vous demander pourquoi CQL fait "match nul" ici plutôt que de perdre. Comme CQL est conçu pour être *conservateur* et pénaliser les actions non vues dans l'ensemble de données, il finit par faire exactement ce que l'expert a fait. Il ne peut pas battre l'expert (car le score maximum possible est déjà atteint), mais il ne casse pas activement la stratégie de l'expert non plus. Il fait simplement jeu égal avec les performances du BC.

C'est le célèbre compromis "qualité des données vs algorithme" :

```
                                    Données EXPERT → BC gagne, CQL fait jeu égal
   Sophistication de l'algorithme ↑         
                                    Données MIXTES → CQL bat clairement le BC
                            
                                    Données ALÉATOIRES → Tout le monde échoue ; besoin d'exploration
```

---

## La place du BC dans le RL moderne

- **Pré-entraînement pour le RL en ligne.** De nombreux systèmes modernes (RT-1, Voyager, bots de jeux) commencent par du BC sur des démonstrations, puis s'affinent (fine-tuning) en ligne avec PPO/SAC.
- **RLHF.** L'étape 1 d'InstructGPT est un réglage fin supervisé — du BC pur sur des réponses écrites par des humains. PPO + le modèle de récompense viennent plus tard.
- **DAgger (Ross et al., 2011).** Une extension intelligente pour corriger le problème de l'**erreur composée** (compounding error).
  *Pourquoi l'erreur composée est-elle un problème si le BC clone parfaitement ?* Même si un modèle de BC est précis à 99 %, cette erreur de 1 % finit par se produire. Quand c'est le cas, l'agent entre dans un état qu'il n'a jamais vu dans l'ensemble de données parfaitement piloté. Parce qu'il est confus, il fait une erreur plus grande, s'éloignant encore plus des données connues, ce qui s'aggrave jusqu'à un échec total (comme tomber d'une falaise).
  *La solution :* Nous pourrions demander à l'expert de conduire indéfiniment, mais le temps d'un expert est cher. À la place, DAgger laisse la politique BC conduire. Quand la politique fait une erreur et dérive vers un état bizarre, on fait une pause, on demande à l'expert "que ferais-tu *juste ici* ?", et on ajoute cela à l'ensemble de données. On ne "sollicite à nouveau l'expert que sur les états visités par la politique BC" car nous avons seulement besoin que l'expert nous apprenne comment récupérer de nos propres erreurs spécifiques.
- **Decision Transformer (Chen et al., 2021).** Un BC "intelligent" qui conditionne la prédiction de l'action sur un *retour à venir* (return-to-go) souhaité, transformant essentiellement le RL hors ligne en une prédiction du prochain jeton (token).

---

## Mots-clés à retenir

| Mot | Signification |
|------|---------|
| **Apprentissage par imitation** | Terme générique pour "copier le démonstrateur" ; le BC en est le membre le plus simple |
| **Erreur composée** | Une petite erreur de BC vous emmène vers des états jamais vus, où les erreurs s'accumulent |
| **Données de démonstration** | Trajectoires produites par un expert, utilisées comme ensemble d'entraînement pour le BC |
| **Plafond des données** | Le score du BC est limité par le score moyen de l'ensemble de données |
| **DAgger** | Une solution interactive pour l'erreur composée |

---

## Résumé en une phrase

> **Le clonage comportemental n'est que de l'apprentissage supervisé sur des paires (état, action) — puissant quand les données sont bonnes, impuissant quand les données sont mixtes.**
