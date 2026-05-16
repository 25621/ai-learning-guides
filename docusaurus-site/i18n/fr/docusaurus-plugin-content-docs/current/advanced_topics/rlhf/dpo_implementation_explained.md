# DPO : Sauter le juge et aller directement à la source

## L'idée maîtresse

Le RLHF classique comporte deux étapes : d'abord entraîner un modèle de récompense (reward model), puis utiliser PPO pour maximiser ses scores. Le DPO (Direct Preference Optimization) pose une question ingénieuse :

*Si le modèle de récompense n'est qu'une étape intermédiaire, pouvons-nous l'ignorer ?*

Il s'avère que oui. Le DPO entraîne le modèle de langage directement à partir de paires de préférences, sans juge séparé, sans boucle d'échantillonnage PPO et sans coefficient KL à ajuster. Il utilise une formule élégante et se comporte comme un apprentissage supervisé.

Cela rend le DPO plus simple à exécuter, plus stable et plus rapide — c'est pourquoi il est rapidement devenu le choix par défaut pour de nombreux modèles alignés en open-source.

## Une analogie de la vie réelle

Supposons que vous coachez un étudiant pour rédiger des essais.

L'approche PPO consiste à : embaucher un professeur pour noter les essais, puis demander à l'étudiant d'écrire essai après essai et de s'ajuster en fonction des notes du professeur.

L'approche DPO consiste à : montrer à l'étudiant deux essais à la fois et lui dire :
"celui-ci est meilleur — essaie d'écrire davantage comme celui-là, et moins comme l'autre." Pas de professeur intermédiaire. L'étudiant s'ajuste directement à partir des comparaisons.

Les deux méthodes peuvent fonctionner. Le DPO finit généralement plus vite car personne n'a à former et à entretenir un professeur séparé.

## Comment fonctionne l'apprentissage (intuition uniquement)

Le DPO utilise les mêmes paires de préférences que la modélisation de récompense : prompt, réponse choisie, réponse rejetée. Pour chaque paire, il pose deux questions :

1. Le modèle est-il devenu **plus susceptible** de produire la réponse choisie que ne l'aurait été le modèle de référence ?
2. Le modèle est-il devenu **moins susceptible** de produire la réponse rejetée que ne l'aurait été le modèle de référence ?

L'entraînement pousse ces deux chiffres dans la bonne direction en même temps. Crucialement, le modèle de référence est toujours présent dans la comparaison — il joue le même rôle que la pénalité KL dans PPO. Le modèle est autorisé à changer, mais les changements sont toujours *relatifs* au point de départ.

Un résultat subtil et magnifique de l'article sur le DPO est que cette fonction de perte unique est mathématiquement équivalente à "entraîner un modèle de récompense, puis exécuter PPO avec une pénalité KL". Même destination, voyage plus simple.

## Ce que montre l'expérience

Nous avons entraîné une politique directement sur 2 000 paires de préférences pendant 300 époques.

![DPO training](outputs/dpo_implementation.png)

- **À gauche** — la perte DPO diminue à mesure que le modèle apprend à préférer les réponses choisies aux réponses rejetées.
- **Au milieu** — la précision des préférences (la fréquence à laquelle la politique attribue une récompense implicite plus élevée à la réponse choisie) grimpe à environ 99 %.
- **À droite** — la marge de récompense implicite augmente. Le DPO ne nomme jamais de "récompense", mais l'écart entre les log-probabilités des réponses choisies et rejetées, pondéré par bêta, peut être lu comme tel. Il s'élargit régulièrement, ce qui signifie que le modèle devient plus confiant dans ses préférences.

Remarquez à quel point cela semble propre par rapport à PPO. Il n'y a pas de boucle d'échantillonnage, pas de bruit d'exploration et pas de modèle de récompense séparé en cours d'exécution. Chaque époque est une pure mise à jour de type supervisé sur l'ensemble de données de préférences.

## Place du DPO dans le pipeline RLHF

Le DPO est une *alternative* à l'étape deux du pipeline classique :

- **Classique :** préférences → modèle de récompense → PPO → modèle aligné.
- **DPO :** préférences → modèle aligné. (Terminé.)

Le bémol est que le DPO s'entraîne sur un ensemble de données de préférences fixe. PPO, parce qu'il échantillonne de nouvelles réponses à chaque cycle, peut en principe explorer davantage. En pratique, le DPO gagne pour la plupart des cas d'utilisation d'alignement sur un ensemble de données de préférences organisé.

## Pourquoi c'est important en dehors du laboratoire

Le schéma "sauter la mesure intermédiaire" se retrouve partout :

- Un coach corrigeant la posture d'un nageur en lui montrant des démonstrations côte à côte plutôt qu'en chronométrant chaque longueur.
- Un photographe éditant deux photos à la fois, choisissant la meilleure, au lieu de construire une grille de notation pour une "bonne photo".
- Un responsable du recrutement comparant deux CV plutôt que de noter chacun d'eux par rapport à une liste de contrôle de 30 points.

Lorsque vous n'avez besoin que de *classer*, vous n'avez pas besoin d'une échelle absolue. Le DPO est cette intuition appliquée aux modèles de langage.

## Résumé en une phrase

**Le DPO transforme directement les paires de préférences en un meilleur modèle, sans modèle de récompense intermédiaire — plus simple que PPO, et souvent tout aussi efficace.**
