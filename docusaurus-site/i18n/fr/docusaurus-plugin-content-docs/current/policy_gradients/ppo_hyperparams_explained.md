# Sensibilité des hyperparamètres de PPO : Qu'est-ce qui compte le plus ?

## Pourquoi les hyperparamètres sont-ils importants ?

Imaginez que vous préparez un gâteau au chocolat. La recette demande :
- 2 œufs
- 200g de farine
- 1 cuillère à café de levure chimique
- 35 minutes à 180°C

Si vous utilisez 10 œufs, le gâteau explose. Si vous utilisez 0,1 cuillère à café de levure, il ne lève pas. Si vous le faites cuire à 300°C pendant 10 minutes, il brûle à l'extérieur et reste cru à l'intérieur.

**Les hyperparamètres de PPO sont comme les ingrédients et les réglages du four.** La bonne combinaison fonctionne merveilleusement bien ; de mauvais réglages peuvent empêcher tout apprentissage.

Ce script teste systématiquement 3 hyperparamètres clés en n'en changeant qu'UN SEUL à la fois, en exécutant chaque réglage avec 3 graines aléatoires (seeds) différentes et en comparant les résultats.

---

## Les trois expériences

### Expérience 1 : Clip Epsilon (ε)

```
ε = 0.05   (très conservateur — seules de minuscules modifications de politique sont autorisées)
ε = 0.2    (standard — équilibre entre sécurité et vitesse)
ε = 0.4    (agressif — permet des changements de politique importants)
```

**Que contrôle ε ?**

ε est la taille de la « fenêtre de sécurité » autour de l'ancienne politique :
```
le ratio doit rester dans [1 - ε,  1 + ε]
ε=0.05 : ratio dans [0.95, 1.05]  ← changements minuscules
ε=0.2 :  ratio dans [0.80, 1.20]  ← standard  
ε=0.4 :  ratio dans [0.60, 1.40]  ← changements importants
```

**Exemple concret :** Pensez à ε comme à « l'angle maximum dont vous avez le droit de tourner le volant en une seule fois ».
- ε=0.05 : Comme conduire sur la glace — uniquement de minuscules ajustements.
- ε=0.2 : Conduite normale — virages raisonnables.
- ε=0.4 : Pilote de course — direction agressive, risque de **tête-à-queue** (perte de contrôle parce que le changement est trop radical).

**Résultats attendus :**
- ε=0.05 : Apprentissage lent mais stable (trop prudent).
- ε=0.2 : Bon équilibre (la valeur idéale « Boucle d'or » — ni trop petite, ni trop grande, juste ce qu'il faut).
- ε=0.4 : Peut apprendre vite mais risque de **dépasser la cible et d'osciller** (dépasser = aller au-delà de la politique optimale ; osciller = rebondir autour de celle-ci sans se stabiliser).

---

### Expérience 2 : Taux d'apprentissage (Learning Rate)

```
lr = 1e-4  (lent mais stable)
lr = 3e-4  (standard)
lr = 1e-3  (rapide mais risqué)
```

**Que contrôle le taux d'apprentissage ?**

Le taux d'apprentissage est comme la « taille du pas » lors de l'ascension d'une colline (chaque pas = une mise à jour des poids du réseau de neurones) :
- Trop petit : Prend une éternité pour atteindre le sommet (converge lentement).
- Trop grand : Vous dépassez le sommet et tombez de l'autre côté (**diverge** — la récompense s'effondre ou fluctue violemment).
- Juste ce qu'il faut : Progression régulière vers le sommet.

**Exemple concret :** Accorder une corde de guitare.
- lr=1e-4 : Minuscules rotations de la **cheville** (le bouton que vous tournez) — prend une éternité mais est précis.
- lr=3e-4 : Accordage normal — trouve la bonne note en quelques tours.
- lr=1e-3 : Grands **coups secs** sur la cheville — risque de **casser** la corde !

**Résultats attendus :**
- lr=1e-4 : Finit par être bon mais très lent.
- lr=3e-4 : Meilleure performance globale.
- lr=1e-3 : Progrès initiaux rapides, puis instabilité.

---

### Expérience 3 : Époques de mise à jour (K)

```
K = 3   (conservateur — peu de passages sur chaque lot)
K = 10  (standard)
K = 20  (agressif — de nombreux passages sur chaque lot)
```

**Que contrôlent les époques de mise à jour ?**

Après avoir collecté un **rollout** (= jouer pendant un certain temps pour accumuler de l'expérience), PPO regroupe cette expérience dans un **lot** (batch). Il effectue ensuite K **passages** sur ces mêmes données.
Plus d'époques = extraire plus d'apprentissage de chaque lot, mais au risque de **sur-apprentissage sur des données périmées** (= mémoriser des schémas qui étaient vrais sous l'ancienne politique mais qui ne sont plus valides une fois la politique mise à jour).

**Exemple concret :** Un étudiant s'exerçant sur une série de 20 problèmes de maths.
- K=3 : Faire chaque problème 3 fois → apprend toujours, ne sur-apprend pas sur l'exercice.
- K=10 : Faire chaque problème 10 fois → maîtrise solide de ces problèmes spécifiques.
- K=20 : Faire chaque problème 20 fois → **mémorise les solutions sans vraiment comprendre les maths** (le modèle s'adapte parfaitement au lot spécifique mais perd sa capacité à généraliser) !

> ⚠️ **« Mais les résultats pour K=20 semblent corrects — pourquoi s'en soucier ? »**
> L'astuce de clipping de PPO limite la modification de la politique par passage, donc K=20 ne causera pas d'effondrement soudain. Cependant, l'agent sur-adapte discrètement à des données qui ne reflètent plus ce que la politique actuelle rencontrerait réellement. Cela **ralentit l'apprentissage à long terme** : chaque rollout apprend moins à l'agent qu'il ne le devrait, car les passages ultérieurs recyclent des informations de plus en plus périmées. Les dommages sont graduels, pas spectaculaires — c'est précisément pourquoi il est facile de les ignorer lors d'expériences courtes.

Le clipping empêche le sur-apprentissage catastrophique, mais trop d'époques peuvent tout de même ralentir l'apprentissage global.

**Résultats attendus :**
- K=3 : Moins efficace (potentiel d'apprentissage gaspillé par lot).
- K=10 : Bon équilibre.
- K=20 : Risque que la politique devienne **trop confiante sur des données périmées**.

---

## Comment lire les résultats

Le graphique montre trois courbes, chacune faisant varier un hyperparamètre :

```
Graphique de gauche :  Clip Epsilon — quel ε apprend le plus vite ?
Graphique du milieu :  Learning Rate — quel lr est le plus stable ?
Graphique de droite :  Update Epochs — quel K trouve la meilleure politique ?
```

Chaque ligne représente la **récompense moyenne sur 3 graines (seeds)** (pour réduire l'aléa).

**Ce qu'il faut regarder :**
1. **Vitesse d'apprentissage :** Quelle ligne atteint une récompense élevée le plus vite ?
2. **Performance finale :** Quelle ligne atteint la récompense finale la plus élevée ?
3. **Stabilité :** Quelle ligne présente le moins d'oscillations ?

Un bon hyperparamètre équilibre les trois !

---

## Méthodologie : Expérimentation scientifique

Cette expérience utilise une conception d'**étude d'ablation** (= une méthode où l'on retire ou fait varier un seul composant à la fois pour mesurer son impact individuel) :
1. Choisir des valeurs par défaut : ε=0.2, lr=3e-4, K=10.
2. Changer UN SEUL paramètre à la fois.
3. Garder tout le reste fixe.
4. Comparer les résultats.

Cela nous indique l'effet de CHAQUE paramètre isolément.

**Exemple concret :** Tester si un nouvel engrais aide les plantes :
- Changer d'engrais, garder tout le reste identique (même sol, eau, soleil).
- Si les plantes poussent mieux → l'engrais a aidé !

---

## Constatations courantes en pratique

| Hyperparamètre | Trop petit | Juste milieu | Trop grand |
|----------------|-----------|------------|-----------|
| **ε (clip)** | Convergence lente | ε ≈ 0.2 | Instabilité |
| **lr** | Trop lent | 2.5e-4 à 3e-4 | Divergence |
| **K (époques)** | **Gaspillage de données** | K = 4-10 | Sur-apprentissage sur données périmées |
| **n_steps** | Trop de bruit | 128-2048 | **Erreurs de mémoire (OOM)** |
| **batch_size** | Trop de bruit | 32-256 | **Erreurs de mémoire (OOM)** |

Ces « points idéaux » peuvent varier selon l'environnement !

---

## L'idée clé : PPO est relativement robuste

Comparé aux algorithmes précédents (comme DQN sans réseaux cibles), PPO est relativement robuste aux choix d'hyperparamètres. Le mécanisme de clipping fournit un filet de sécurité naturel.

**Exemple concret :** Une voiture avec **ABS** (système antiblocage des roues) contre une voiture sans :
- Sans ABS (DQN) : Un mauvais virage (mauvais hyperparamètre) et vous partez en tête-à-queue.
- Avec ABS (PPO) : La voiture se corrige d'elle-même — les hyperparamètres raisonnables fonctionnent tous à peu près bien.

Cette robustesse est l'une des raisons majeures pour lesquelles PPO est l'algorithme de RL le plus populaire en pratique !

---

## Points clés à retenir

| Concept | Français simple |
|---------|---------------|
| **Étude d'ablation** | Changer une chose à la fois pour voir son effet. |
| **Clip epsilon ε** | Limite de sécurité — 0.2 est généralement le mieux. |
| **Taux d'apprentissage** | **Taille du pas** — de combien les poids du réseau sont ajustés. **2.5e-4 à 3e-4** est la notation scientifique pour 0.00025 à 0.0003. |
| **Époques de mise à jour K** | Combien de fois réutiliser chaque lot — 4-10 est standard. |
| **Graines aléatoires (Seeds)** | Chaque expérience est répétée avec différentes graines pour vérifier si les résultats sont cohérents ou s'il s'agit d'un coup de chance. |

---

## Résumé : Les méthodes de gradient de politique en un coup d'œil

```
REINFORCE                 A2C                       PPO
    │                      │                         │
Épisodes complets     Mises à jour n-étapes     N-étapes + clipping
Simple mais bruité    Plus rapide mais instable Stable + efficace
Idéal pour env.       Environnements de         Environnements difficiles
faciles               difficulté moyenne        (standard industriel)
```

**Si vous ne devez apprendre qu'UN SEUL algorithme, apprenez PPO.** C'est le fondement de :
- L'entraînement de ChatGPT d'OpenAI (RLHF utilise PPO).
- Les suites d'AlphaGo de DeepMind.
- La plupart des recherches modernes en robotique.
- Les IA jouant aux jeux vidéo.
