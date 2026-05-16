# SARSA pour Cliff Walking (Marche au bord de la falaise) 🏔️

## Qu'est-ce que c'est ?

Imaginez un **très long couloir** avec une **falaise terrible** le long d'un bord. Si vous tombez de la falaise, vous devez retourner tout au début ! Votre objectif est de marcher d'un bout à l'autre le plus rapidement possible, sans tomber.

**SARSA** est un robot qui apprend à parcourir ce couloir en pratiquant. Il apprend à emprunter un *chemin sûr* qui évite la falaise — même s'il est un peu plus long — car il sait qu'il pourrait accidentellement glisser près du bord lors de son exploration !

---

## La grande idée : Apprendre de ce que l'on fait réellement

SARSA signifie : **S**tate (État) → **A**ction → **R**eward (Récompense) → **S**tate (État) → **A**ction

Ce sont les cinq informations que SARSA utilise pour apprendre :

1. **S** — Où suis-je en ce moment ? (état actuel)
2. **A** — Quelle action ai-je réellement entreprise ?
3. **R** — Quelle récompense ai-je obtenue ?
4. **S** — Où ai-je atterri ?
5. **A** — Quelle action vais-je *réellement entreprendre ensuite* ?

Le dernier "A" est ce qui rend SARSA spécial ! Il se met à jour en utilisant l'action qu'il va *réellement entreprendre ensuite* (même s'il s'agit d'un mouvement d'exploration aléatoire), et non l'action idéale parfaite.

**Exemple de la vie réelle :** Pensez à l'apprentissage du vélo. Si vous savez que vous vacillez parfois de manière aléatoire (exploration), vous restez un peu plus loin des voitures garées — parce que vous savez que votre moi vacillant pourrait faire un écart ! SARSA fait cela : il apprend un chemin sûr car il tient compte de ses propres erreurs aléatoires.

---

## La carte de Cliff Walking

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← FALAISE ← ← ← ← ←
```

- **S** = Départ (en bas à gauche)
- **G** = Objectif (en bas à droite)
- **C** = Falaise — marcher ici = -100 de récompense, redémarrage !
- Chaque autre pas = -1 de récompense

---

## Ce que notre code a trouvé

Après avoir entraîné SARSA pendant 500 épisodes :

| Résultat | Valeur |
|----------|--------|
| Récompense moyenne finale sur 50 épisodes | **-21,6** |
| Récompense du chemin optimal (risqué) | -13 |

La politique apprise par SARSA passe **par le haut de la grille** — un détour sûr ! Cela coûte quelques pas supplémentaires (-21 au lieu de -13), mais il ne tombe presque jamais de la falaise pendant l'entraînement.

---

## Exemples de la vie réelle

- **Infirmier administrant un médicament** : Suit le protocole de sécurité prouvé (chemin sûr) même s'il existe une méthode légèrement plus rapide, car de petites erreurs (exploration) pourraient être dangereuses.
- **Pilotes de ligne** : Suivent des listes de contrôle strictes (chemins sûrs) même lorsque des raccourcis pourraient sembler plus rapides, en tenant compte de l'erreur humaine.
- **Apprendre à cuisiner** : Commencez par des recettes bien testées (sûres), pas par des raccourcis risqués.

---

## Mots-clés à retenir

- **On-policy** (En politique) : Apprend sur la politique qu'il utilise réellement (y compris ses erreurs aléatoires).
- **Mise à jour SARSA** : Utilise l'action suivante *réelle*, et non celle théoriquement la meilleure.
- **Chemin sûr** : Un chemin plus long qui évite le danger, en tenant compte des erreurs d'exploration.
- **Contrôle TD (Différence Temporelle)** : Mise à jour des valeurs après chaque étape (sans attendre la fin de l'épisode).

La grande idée : **SARSA est honnête — il apprend de ce qu'il fait réellement, pas de ce qu'il aimerait faire. Cela le rend prudent et sûr à proximité du danger !**
