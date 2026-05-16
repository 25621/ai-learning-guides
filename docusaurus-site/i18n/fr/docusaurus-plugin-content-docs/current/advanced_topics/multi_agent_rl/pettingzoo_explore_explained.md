# Explorer les Environnements PettingZoo 🦓

## Qu'est-ce que PettingZoo ?

Si vous avez déjà fait du RL mono-agent, vous avez probablement utilisé **Gymnasium** (le successeur d'OpenAI Gym). Chaque environnement s'y ressemble : `env.reset()`, `env.step(action) → obs, reward, done, info` — une nouvelle *observation* du monde, un signal de *récompense* scalaire, un indicateur (*flag*) *done* disant "partie terminée", et un dictionnaire *info* pour le débogage. Cette uniformité est ce qui permet aux bibliothèques de RL de fonctionner.

**PettingZoo** est exactement la même idée, mais pour *plusieurs agents*. C'est un "zoo" d'environnements multi-agents, tous derrière une API bien définie :
- **Problèmes classiques** : des environnements simples comme Pierre-Feuille-Ciseaux pour tester les algorithmes de base.
- **Mondes de grille coopératifs** : des agents naviguant dans une grille pour atteindre un objectif commun.
- **Atari multijoueur** : des jeux compétitifs classiques comme Pong.
- **MPE (Multi-Particle Environment)** : des environnements physiques en espace continu pour la coordination et la compétition complexes.

Si vous savez écrire du code qui fonctionne sur un environnement PettingZoo, vous pouvez l'utiliser sur n'importe quel autre avec presque aucun changement.

---

## Les Deux Styles d'API

Les configurations multi-agents sont plus complexes que les mono-agents car deux agents peuvent agir en même temps, ou tour à tour, ou même dans des ordres arbitraires. PettingZoo résout cela avec deux API parallèles :

### 1) AEC (Agent-Environment-Cycle)

Un seul agent agit à la fois. L'environnement boucle sur les agents dans un certain ordre, et chacun reçoit :
- une **observation** — ce qu'il voit *à cet instant*,
- une **récompense** — le gain généré par l'action *jointe* du dernier tour complet (c'est-à-dire ce qui s'est passé suite aux actions de *tous* les agents ; au échecs, par exemple, votre récompense reflète l'état du plateau après le dernier coup de votre adversaire, pas seulement le vôtre),
- un **flag de terminaison** — `True` quand l'épisode se termine *naturellement* (ex : échec et mat),
- un **flag de troncature** — `True` quand l'épisode est *interrompu* par une limite de temps avant d'avoir atteint une fin naturelle.

C'est le style naturel pour les **jeux au tour par tour** comme les échecs, le Go ou le poker.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = ma_politique(obs, agent)
    env.step(action)
```

### 2) Parallèle (Parallel)

Tous les agents observent et agissent simultanément à chaque étape. `step()` prend un *dictionnaire* d'actions et renvoie des dictionnaires d'observations et de récompenses.

C'est le style naturel pour les **jeux en temps réel** comme le MPE (où tous les agents-points bougent simultanément) ou les mondes de grille multi-agents.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: ma_politique(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

Les deux styles sont **isomorphes** — structurellement équivalents et interchangeables : tout environnement AEC peut être automatiquement enveloppé (*wrapped*) pour ressembler à un environnement Parallèle, et vice versa. PettingZoo fournit les outils de conversion pour que vous n'ayez qu'à écrire votre code pour un seul style.

---

## Analogie avec la Vie Réelle

- **AEC = une soirée jeux de société.** "C'est au tour d'Alice. Puis Bob. Puis Carol. Retour à Alice." Celui qui joue voit l'état le plus récent du plateau.
- **Parallèle = un jeu vidéo multijoueur.** Les quatre joueurs appuient sur les boutons simultanément ; le jeu met à jour le monde 60 fois par seconde.
- **Pourquoi les API uniformes sont importantes.** Imaginez si chaque jeu vidéo multijoueur nécessitait sa propre manette. PettingZoo est la "manette universelle" du MARL (Multi-Agent RL).

---

## Ce que Fait Notre Code

Nous construisons un environnement **style PettingZoo** à partir de zéro : le **Jeu de Coordination Itéré**. Deux agents choisissent à plusieurs reprises le canal `0` ou `1` :

- Même choix → les deux reçoivent +1
- Choix différent → les deux reçoivent -1

L'**observation** reçue par chaque agent est l'action *jointe* précédente — ce que les deux agents ont choisi au tour dernier, encodé en un seul entier. Concrètement : la dernière action de chaque agent est l'une de `{start, 0, 1}` (3 états), la paire est donc encodée comme `3 × état_agent_1 + état_agent_2`, donnant 9 entiers possibles (0 – 8). L'entier 0 est l'état "départ" (pas d'action encore prise). Un épisode dure 25 étapes, le retour total maximum est donc de +25 par agent. **Un jeu aléatoire donne un score ≈ 0** car les agents correspondent 50 % du temps (+1) et diffèrent 50 % du temps (-1).

Nous allons ensuite :

1. **Démontrer l'interface AEC** avec un déroulement aléatoire — cela confirme la boucle AEC de base : `agent_iter()` donne l'agent dont c'est le tour, `last()` lit son observation et sa récompense, et `step()` renvoie son action.
2. **Entraîner deux apprenants Q indépendants via l'interface Parallèle**. Chaque agent possède sa propre table Q indexée par l'**observation de l'action jointe** (ce que *les deux* agents ont fait au tour précédent), lui permettant d'apprendre : "quand nous avons tous deux choisi 0 la dernière fois, je devrais encore choisir 0".
3. **Tenter d'importer la vraie bibliothèque `pettingzoo`** et lancer l'un de ses environnements intégrés (Pierre-Feuille-Ciseaux). Si PettingZoo n'est pas installé, nous passons cette étape avec un message explicatif.

### Ce que vous devriez voir

| Étape | Résultat attendu |
|-------|----------|
| Déroulement aléatoire (AEC) | Retour moyen par épisode proche de **0** — les agents aléatoires ne se coordonnent pas. |
| Apprenants Q (Parallèle) — 100 prem. épisodes | Environ **0** — toujours principalement aléatoire pendant l'exploration. |
| Apprenants Q — 100 derniers épisodes | Fortement positif, **+20 à +25** — **la coordination a émergé** : les deux agents ont appris à choisir le même canal de manière fiable. |

Le graphique `outputs/pettingzoo_coordination.png` montre les retours individuels des épisodes (gris) et une courbe de **Moyenne** mobile (bleu). La moyenne lisse le bruit pour montrer la tendance : les agents passent d'un jeu aléatoire vers une **coordination** stable. La ligne verte pointillée marque le plafond de coordination parfaite.

Si `pettingzoo` est installé, le script lance également `pettingzoo.classic.rps_v2` pour prouver que notre code fonctionne aussi bien avec la vraie bibliothèque qu'avec notre environnement fait main.

---

## Pourquoi construire d'abord un environnement personnalisé ?

Parce que **l'API est la leçon**. (Comprendre comment structurer l'interaction entre plusieurs agents et l'environnement est plus important que les règles spécifiques du jeu.) Le RL multi-agent a plusieurs variantes (tour par tour, temps réel, coopératif, compétitif, mixte), et elles rentrent toutes dans le modèle AEC / Parallèle. Une fois que vous avez implémenté ces deux boucles, utiliser n'importe quel environnement PettingZoo n'est qu'une question de changer de constructeur — le code d'entraînement reste le même.

C'est exactement ainsi que Gymnasium a changé le RL mono-agent : en faisant de l'environnement une boîte noire derrière une interface uniforme.

---

## Où le Q-learning indépendant aide et nuit

Les jeux de coordination sont *indulgents* — les agents partagent le signe de la récompense, leurs intérêts sont alignés. Des apprenants indépendants peuvent résoudre cela facilement.

Dans les jeux **adversaires** (RPS), le Q-learning indépendant oscille éternellement (quand un agent s'adapte, l'autre change de stratégie pour le contrer). Dans les jeux à **observabilité partielle**, il ne peut rien apprendre du tout car l'"observation" n'est qu'une partie de l'état réel. PettingZoo inclut ces deux types d'environnements pour que vous puissiez constater ces modes d'échec par vous-même.

---

## Mots Clés à Retenir

| Mot | Signification |
|------|---------|
| **PettingZoo** | Le Gymnasium du RL multi-agent — une bibliothèque d'environnements MARL standardisés |
| **AEC** | Agent-Environment-Cycle : un seul agent agit par étape (tour par tour) |
| **API Parallèle** | Tous les agents agissent simultanément à chaque étape |
| **MPE** | Multi-Particle Environment, un banc d'essai coopératif/compétitif populaire inclus dans PettingZoo |
| **CTDE** | Centralised Training, Decentralised Execution — s'entraîner avec une vue globale, mais agir avec seulement des observations locales |
| **Q-learning indépendant** | Chaque agent lance un Q-learning classique, ignorant l'existence des autres apprenants |

---

## Résumé en une phrase

> **PettingZoo donne à chaque environnement multi-agent la même forme — ainsi, le code que vous écrivez aujourd'hui fonctionnera encore demain sur un jeu totalement différent.**

Une fois ces deux styles d'API maîtrisés, vous pourrez passer au MADDPG (critique centralisé), QMIX (mélange de valeurs pour équipes coopératives), MAPPO (PPO multi-agent) ou tout autre algorithme MARL moderne — le côté environnement de votre code n'aura plus besoin de changer.
