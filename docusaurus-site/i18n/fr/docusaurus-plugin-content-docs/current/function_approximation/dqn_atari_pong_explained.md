# DQN sur Atari Pong 🏓

## Qu'est-ce qu'Atari Pong ?

Pong est l'un des plus anciens jeux vidéo jamais créés — c'est comme du tennis de table numérique ! Deux raquettes font rebondir une balle d'un côté à l'autre. Vous gagnez un point si l'adversaire rate la balle. Le jeu se termine lorsqu'un joueur atteint 21 points.

Dans notre version, l'IA contrôle une raquette. L'adversaire (l'ordinateur) contrôle l'autre. Le jeu commence toujours avec un score de -21 (le pire score possible). Un bon agent atteint 0 ou +21.

---

## Pourquoi Pong est-il difficile pour une IA ?

Dans CartPole, le robot pouvait "voir" les nombres directement (angle de la perche, vitesse du chariot...). Dans Pong, tout ce qu'il voit sont des **pixels bruts** — des milliers de petits points colorés sur un écran !

```
Entrée CartPole : [0.02, −0.14, 0.01, −0.23]   ← 4 nombres, facile !
Entrée Pong :     [grille de pixels : 210×160×3] ← 100 800 nombres, BEAUCOUP plus difficile !
```

Le robot doit déduire à partir des pixels :
- Où est ma raquette ?
- Où est la balle ?
- La balle se déplace-t-elle vers la gauche ou vers la droite ?
- À quelle vitesse ?

Les humains font cela automatiquement (nous avons une vision incroyable !). Pour une IA, c'est un défi immense.

---

## Voir le mouvement : Frame Stacking (Empilement de trames) 🎬

Une seule image (capture d'écran) ne vous dit pas si la balle se déplace vers la gauche ou vers la droite. Vous devez voir PLUSIEURS images pour comprendre le mouvement — tout comme un folioscope (flip book) ne fonctionne que lorsque vous faites défiler plusieurs pages.

**Frame Stacking :** On envoie simultanément les 4 dernières images au réseau.

```
Image 1 : balle à la position 40
Image 2 : balle à la position 43    → Empilement de ces 4 images → Le réseau voit le MOUVEMENT !
Image 3 : balle à la position 46
Image 4 : balle à la position 49
```

Le réseau peut maintenant déduire : "la balle se déplace vers la droite à la vitesse 3".

**Exemple de la vie réelle :** Regarder un film par rapport à une seule image. Une image fixe d'une course de voitures n'est qu'une image floue. Regardez 4 images, et vous pouvez dire quelle voiture est la plus rapide !

---

## Voir avec un CNN 🔍

Pour les entrées de type pixels, nous utilisons un réseau de neurones spécial appelé **Réseau de Neurones Convolutif (CNN - Convolutional Neural Network)**. Au lieu de regarder tous les pixels à la fois, un CNN utilise des fenêtres glissantes pour détecter des motifs — comme des yeux balayant une image.

```
Pixels bruts (84×84×4 images)
       ↓
Couche Conv 1 (filtre 8×8, foulée 4) → trouve les bords et les formes
       ↓
Couche Conv 2 (filtre 4×4, foulée 2) → trouve les objets (raquettes, balle)
       ↓
Couche Conv 3 (filtre 3×3, foulée 1) → trouve les relations
       ↓
Aplatissement (Flatten) → 512 neurones → Valeurs Q (une par action)
```

**Exemple de la vie réelle :** Lorsque vous cherchez un ami dans une foule, votre cerveau remarque d'abord des formes (une personne), puis des caractéristiques (couleur de cheveux), puis des détails (son visage). Les CNN fonctionnent de la même manière — des motifs simples aux plus complexes !

---

## Prétraitement : Réduire le monde

Les images de Pong font 210×160 pixels en couleur. C'est trop grand ! Nous prétraitons chaque image :

1. **Niveaux de gris** — la couleur n'a pas d'importance pour Pong (la balle est de toute façon toujours blanche).
2. **Redimensionnement à 84×84** — plus petit = entraînement plus rapide, mais toujours assez clair pour voir.
3. **Normalisation à [0,1]** — on divise les valeurs des pixels par 255 pour obtenir de petits nombres.

**Exemple de la vie réelle :** C'est comme faire une photocopie à 50 %. Les détails importants (balle, raquettes) sont toujours visibles, mais plus petits. Le photocopieur ne se soucie pas non plus des couleurs !

---

## Reward Clipping : Traiter tous les jeux équitablement ✂️

Dans Pong, vous obtenez +1 pour un point marqué, -1 pour un point encaissé. Dans certains autres jeux Atari, les scores peuvent s'élever à des milliers !

Nous effectuons un **reward clipping** (écrêtage des récompenses) entre [-1, +1] afin que le réseau ne se soucie pas de l'échelle des récompenses. Ce même code peut s'entraîner sur N'IMPORTE QUEL jeu Atari sans ajuster l'échelle des récompenses.

---

## Combien de temps prend l'entraînement ?

| Durée de l'entraînement | Ce que l'agent apprend |
|---|---|
| 100 000 étapes | Surtout aléatoire, réagit à peine |
| 1 million d'étapes | Commence parfois à se déplacer vers la balle |
| 5 millions d'étapes | Renvoie quelques coups |
| 10 millions d'étapes | Jeu compétitif, peut gagner quelques manches |
| 20 millions+ étapes | Bat souvent l'ordinateur adverse |

Notre démonstration dure **300 000 étapes** — assez pour voir que l'architecture d'entraînement fonctionne et observer les débuts de l'apprentissage, mais pas assez pour maîtriser le jeu.

**Exemple de la vie réelle :** Apprendre le piano prend des mois. Une séance de pratique de 10 minutes montre que vous faites les choses correctement, mais ne vous attendez pas encore à donner des concerts !

---

## Ce que notre code a révélé

Après 300 000 étapes sur Pong :
- L'agent commence avec des scores autour de -20 (il renvoie à peine la balle).
- À la fin, il s'améliore généralement pour atteindre environ -15 à -10.
- La courbe d'apprentissage montre une amélioration progressive par rapport au jeu aléatoire.

Pour voir de réelles performances compétitives sur Pong, vous devriez lancer l'entraînement pendant environ 10 millions d'étapes ou plus avec un GPU. L'implémentation est complète et correcte — elle a juste besoin de plus de temps !

---

## Vocabulaire clé

| Mot | Signification |
|-----|---------------|
| **CNN** | Convolutional Neural Network (Réseau de Neurones Convolutif) — spécialisé pour les entrées d'images |
| **Frame Stacking** | Envoi de plusieurs images consécutives pour capturer le mouvement |
| **Prétraitement** | Transformation des images brutes (niveaux de gris, redimensionnement, normalisation) avant de les envoyer au réseau |
| **Reward Clipping** | Limitation des récompenses à [-1, +1] pour fonctionner sur différents jeux |
| **ALE** | Arcade Learning Environment — la bibliothèque qui fait fonctionner les jeux Atari |

---

## Un exploit historique

Lorsque DeepMind a publié DQN en 2015, le monde a été stupéfait. Un SEUL algorithme, avec la MÊME architecture, a appris à jouer à 49 jeux Atari différents — dont beaucoup à un niveau surhumain — uniquement à partir des pixels bruts et d'un score !

Avant DQN, on pensait qu'il fallait coder manuellement la stratégie pour chaque jeu. DQN a montré qu'un système d'apprentissage polyvalent pouvait tout comprendre par lui-même. Ce fut un moment historique pour l'IA !
