# Experience Replay : Apprendre au robot à se souvenir 🎒

## Le problème : L'oubli (et la confusion)

Vous vous souvenez que le DQN "naïf" était instable ? La raison principale est l'**apprentissage corrélé**.

Lorsque le robot joue, il vit les choses dans l'ordre :
> Étape 1 → Étape 2 → Étape 3 → Étape 4 → ...

Ces étapes sont liées ! Si le robot penche à gauche à l'étape 10, il penchera probablement aussi à gauche à l'étape 11. Elles ne sont pas indépendantes — elles dépendent les unes des autres.

Lorsque nous mettons à jour le réseau en utilisant ces étapes corrélées, c'est comme essayer d'apprendre l'histoire en lisant le même chapitre encore et encore. On finit par devenir très bon sur un chapitre, mais on oublie tout le reste !

**Exemple de la vie réelle :** Imaginez que vous révisiez un examen en ne pratiquant que les devoirs d'hier. Vous deviendriez excellent sur ces problèmes précis, mais l'examen comportera des questions différentes ! Vous avez besoin de pratiquer un MÉLANGE de différents problèmes.

---

## La solution : Une boîte à souvenirs 📦

L'**Experience Replay** (rejeu d'expérience) ajoute une grande boîte à souvenirs (le **replay buffer**) au robot.

Au lieu d'apprendre uniquement de sa toute dernière expérience, le robot :
1. **Stocke** chaque expérience dans la boîte : (état, action, récompense, état suivant)
2. **Pioche au hasard** une poignée de souvenirs dans la boîte
3. **Apprend de ce mélange aléatoire** au lieu de se contenter de la dernière étape

```
Étape 1 → [stockage dans la boîte]
Étape 2 → [stockage dans la boîte]
Étape 3 → [stockage dans la boîte]
...
Étape 50 → [stockage] → pioche 64 souvenirs au hasard → mise à jour du réseau
Étape 51 → [stockage] → pioche 64 souvenirs au hasard → mise à jour du réseau
```

**Exemple de la vie réelle :** Pensez à un album photo. Vous ne comprenez pas votre vie en regardant seulement les photos d'aujourd'hui. Vous feuilletez aussi les ANCIENNES photos — un mélange de bons souvenirs et de moments difficiles. Cela vous aide à comprendre les schémas sur toute votre vie, pas seulement sur la journée actuelle.

---

## Pourquoi l'échantillonnage aléatoire aide-t-il ?

En choisissant les souvenirs au hasard, nous cassons les corrélations. Le robot peut apprendre de :
- Un souvenir où la perche était parfaitement droite (il y a 500 étapes)
- Un souvenir où la perche était sur le point de tomber (il y a 20 étapes)
- Un souvenir où il a eu de la chance (à l'étape 3)

Ce mélange aléatoire permet :
✅ Au robot d'apprendre d'une grande variété de situations.
✅ À chaque souvenir d'être "rejoué" plusieurs fois (utilisation efficace de l'expérience).
✅ Au réseau de ne pas se focaliser uniquement sur les événements récents (surapprentissage).

---

## Apprentissage par mini-lots (Mini-Batch Learning)

Au lieu de faire une mise à jour sur UNE seule expérience à la fois, nous faisons une mise à jour sur **64 expériences simultanément** (un "mini-lot" ou "mini-batch"). C'est un peu comme :
- Ancienne méthode : Lire une fiche de révision, s'interroger.
- Nouvelle méthode : Lire 64 fiches différentes, puis s'interroger sur l'ensemble.

Les mini-lots rendent le signal d'apprentissage beaucoup plus fiable et moins bruité.

---

## Période d'échauffement (Warmup Period)

Nous ne commençons pas l'apprentissage tout de suite ! Le replay buffer a d'abord besoin de quelques souvenirs. Nous attendons d'avoir au moins **500 expériences** dans la boîte avant que l'entraînement ne commence réellement.

**Exemple de la vie réelle :** Vous n'essaieriez pas de cuisiner un plat avant d'avoir rassemblé vos ingrédients. La période d'échauffement, c'est comme faire les courses avant de cuisiner !

---

## Ce que montre la comparaison

Lorsque vous lancez `dqn_experience_replay.py`, vous verrez deux courbes d'apprentissage :

| DQN Naïf | DQN + Replay |
|----------|--------------|
| Très irrégulier | Plus fluide |
| Chutes fréquentes (oublie tout) | Amélioration plus constante |
| Forte variance | Faible variance |

La version avec replay :
- Atteint des scores élevés de manière plus fiable.
- Ne rechute pas aussi souvent de 500 à 30.
- Affiche une progression d'apprentissage plus stable.

---

## Le Replay Buffer en code

```
ReplayBuffer:
  - capacité : 10 000 souvenirs (les plus anciens sont oubliés quand c'est plein)
  - push(état, action, récompense, état_suivant, terminé)
  - sample(taille_lot=64) → lot aléatoire
```

Imaginez un carnet de 10 000 lignes. Quand il est plein, vous effacez la ligne la plus ancienne pour écrire la nouvelle. Et vous étudiez toujours à partir d'une page choisie au hasard !

---

## Vocabulaire clé

| Mot | Signification |
|-----|---------------|
| **Experience Replay** | Stocker et réutiliser aléatoirement les expériences passées pour l'entraînement |
| **Replay Buffer** | La boîte à souvenirs qui stocke les tuples (état, action, récompense, état suivant) |
| **Mises à jour corrélées** | Quand les données d'entraînement dépendent d'elles-mêmes (mauvais pour l'apprentissage !) |
| **Mini-lot (Mini-batch)** | Un petit échantillon aléatoire de souvenirs utilisé pour une étape de mise à jour |
| **Décorrélation** | Casser les liens entre des expériences consécutives |

---

## Que manque-t-il encore ?

Même avec un replay buffer, il reste un autre problème : la **cible mouvante**.

Chaque fois que nous mettons à jour le réseau, les valeurs Q changent. Mais ces valeurs Q mises à jour sont AUSSI utilisées pour calculer la cible de la PROCHAINE mise à jour. C'est un cercle de confusion !

Ce problème est résolu par le **Réseau Cible (Target Network)** — une copie figée du réseau qui ne se met à jour que toutes les 100 étapes. Cela permet à la "cible" de rester immobile un moment pour que le robot puisse viser de manière fiable !
