# Comparaison des Stratégies d'Exploration 🔦

## Le problème en une phrase

Un agent d'apprentissage par renforcement (RL) doit faire deux choses qui tirent dans des directions opposées :

- **Exploiter (Exploit)** : faire ce qui a le mieux fonctionné jusqu'à présent.
- **Explorer (Explore)** : essayer quelque chose de nouveau, au cas où ce serait encore mieux.

Penchez trop vers l'exploitation et vous vous contenterez joyeusement d'une routine médiocre pour toujours. Penchez trop vers l'exploration et vous ne récolterez jamais les fruits de votre apprentissage. C'est la *manière* dont vous explorez — et pas seulement le *fait* d'explorer — qui différencie un agent capable de résoudre Montezuma's Revenge d'un agent qui obtient un score de zéro.

Ce script met **cinq** stratégies d'exploration en concurrence sur les deux mêmes tâches difficiles, afin que vous puissiez observer leurs comportements respectifs.

## Analogie concrète : Choisir un endroit pour déjeuner

Vous venez d'emménager dans une nouvelle ville qui compte 200 restaurants.

- **ε-greedy** = "Aller à mon restaurant préféré actuel, mais une fois tous les dix jours, lancer un dé et choisir un restaurant *totalement au hasard*." Vous testerez beaucoup d'endroits mais *sans but précis* — et vous continuerez à retourner dans des endroits que vous détestiez déjà.
- **Initialisation optimiste (Optimistic initialisation)** = "Partir du principe que *chaque* restaurant que je n'ai pas essayé est le meilleur de la ville jusqu'à preuve du contraire." Vous passerez méthodiquement en revue les 200, en rayant chacun d'eux à mesure que la réalité vous déçoit — et vous trouverez rapidement les restaurants vraiment excellents.
- **UCB (Upper Confidence Bound)** = "Préférer mon restaurant préféré, mais accorder un *bonus* aux endroits que j'ai à peine essayés — moins j'en sais sur un endroit, plus le bonus est grand." C'est une approche intelligente pour choisir *quel* endroit inconnu essayer aujourd'hui, mais chaque décision est locale : elle choisit la meilleure option *à l'instant présent* sans planifier d'itinéraire à travers des quartiers entiers inexplorés. Elle ne se dira pas "je devrais traverser la ville pour aller à l'est, car il y a vingt endroits non essayés regroupés là-bas" — chaque restaurant est évalué isolément, étape par étape.
- **Bonus de récompense basé sur le comptage (Count-based reward bonus)** = comme UCB, mais vous *appréciez également la nouveauté elle-même* — un repas dans un endroit tout nouveau est intrinsèquement satisfaisant, et cette satisfaction façonne votre plan à long terme sur les quartiers dans lesquels s'aventurer.
- **Bonus de récompense basé sur l'erreur de prédiction (Prediction-error reward bonus)** = "Je ressens une excitation lors d'un repas qui m'a *surpris* — quelque chose que je n'aurais pas pu prédire." Un nouvel endroit qui s'avère exactement comme prévu ? Bof. Un endroit radicalement différent de votre modèle mental ? Fascinant, et vous mettez à jour votre plan pour en chercher d'autres semblables.

## Les cinq stratégies (toutes dans `compare_exploration.py`)

### 1. ε-greedy — le choix par défaut, et c'est de l'hésitation (dithering), pas de l'exploration

Agir avec gourmandise (greedy), mais avec une probabilité ε, choisir une action uniformément aléatoire. C'est la ligne de base standard dans DQN et ses dérivés. Son défaut fatal sur les tâches difficiles : **chaque étape est un lancer de pièce indépendant.** Pour réussir une chaîne de `N` bons mouvements, il faut que la pièce tombe du bon côté `N` fois de suite — ce qui est exponentiellement improbable. ε-greedy, c'est du *tâtonnement*, pas de l'*exploration*.

### 2. Initialisation optimiste — "innocent jusqu'à preuve d'ennui"

Commencer *chaque* valeur Q au plus grand retour possible, `R_max / (1 − γ)`. Ainsi, une action que l'agent n'a jamais essayée semble être la meilleure chose au monde, donc la politique **greedy** est forcée d'aller l'essayer ; ce n'est qu'après l'avoir visitée que la valeur chute vers la vérité. L'optimisme concernant les régions *non* explorées se **propage automatiquement à travers la fonction de valeur** (via le bootstrap du Q-learning), de sorte que l'agent est attiré, étape par étape, vers les parties du monde qu'il n'a pas vues. Presque gratuit, pas de comptabilité supplémentaire — et, comme vous le verrez, c'est l'explorateur *profond* le plus robuste dans un petit monde tabulaire.

### 3. Sélection d'action de type UCB — bonus dans le *choix*, pas dans la *récompense*

Choisir `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]` : préférer les actions de grande valeur, mais gonfler artificiellement celles que vous avez rarement essayées. Célèbre dans le cadre des bandits manchots (multi-armed bandits). Le piège : le bonus ne réside que dans la **règle de sélection d'action**, jamais dans la récompense — il ne se transmet donc *pas* à travers la fonction de valeur. UCB est excellent pour "s'assurer d'avoir essayé chaque action dans *cet* état" mais faible pour "planifier un itinéraire vers une région inexplorée éloignée".

### 4. Bonus de **récompense** basé sur le comptage — la curiosité, version classique

Ajouter `1/√(N(s,a))` à la **récompense** (avec un poids `beta` qui diminue). Puisque c'est dans la récompense, le Q-learning le propage : les états qui mènent vers des régions nouvelles deviennent précieux. C'est l'idée du MBIE-EB / "exploration bonus" classique.

### 5. Bonus de **récompense** basé sur l'erreur de prédiction — la curiosité, version ICM/RND

Ajouter `−log P(s'|s,a)` provenant d'un minuscule modèle prévisionnel appris à la récompense (toujours avec un `beta` décroissant). Le signal de nouveauté le plus net des cinq : dans un monde déterministe, la surprise d'une transition tombe à ~0 dès que vous l'avez vue une fois, au lieu de s'estomper lentement comme `1/√N`. Le cousin tabulaire d'ICM / RND.

## Les deux tâches de test

- **Tâche A — MiniMontezuma** : un monde en grille de type clé→porte→trésor, avec une récompense uniquement au trésor (~15 mouvements parfaits de distance). Teste la capacité à survivre à une longue chaîne de récompenses éparses (sparse-reward).
- **Tâche B — DeepSea(N)** : la chaîne d'exploration profonde de référence, exécutée avec des longueurs `N = 5, 8, 11, 14`. La récompense se cache derrière `N` bons mouvements, chacun ayant un petit coût immédiat — un agent myope apprend donc à éviter le coût et ne trouve jamais le prix. Teste si la stratégie fonctionne toujours à mesure que la chaîne s'allonge.

## Ce qui se passe réellement (lancez le script pour voir)

**Tâche A — MiniMontezuma :**

| Stratégie | Premier trésor | Taux de réussite final |
|----------|---------------:|-----------------:|
| ε-greedy | jamais | 0,00 |
| initialisation optimiste | ~épisode 1 | 1,00 |
| sélection d'action UCB | ~épisode 3 | ~0,95 |
| bonus récompense comptage | ~épisode 82 | ~0,41 |
| bonus récompense prédiction | ~épisode 23 | 1,00 |

**Tâche B — DeepSea, fraction de graines (seeds) ayant trouvé la récompense :**

| Stratégie | N=5 | N=8 | N=11 | N=14 |
|----------|----:|----:|-----:|-----:|
| ε-greedy | 0 | 0 | 0 | 0 |
| initialisation optimiste | 1,0 | 1,0 | 1,0 | 1,0 |
| sélection d'action UCB | 1,0 | 1,0 | 0,0 | 0,0 |
| bonus récompense comptage | 1,0 | 1,0 | ~0,1 | 0,0 |
| bonus récompense prédiction | ~0,9 | ~0,8 | ~0,9 | ~0,2 |

*(Les chiffres varient légèrement selon les graines aléatoires, mais la tendance générale est constante.)*

## Les leçons à retenir

1. **ε-greedy n'est pas de l'exploration.** Il ne résout jamais l'une ou l'autre des tâches difficiles. Les tâtonnements aléatoires ne permettent tout simplement pas de suivre de longues séquences correctes. (Pourtant, c'est toujours le choix par défaut dans beaucoup de codes — car sur des tâches *faciles*, c'est suffisant et extrêmement simple.)

2. **La vraie exploration signifie être optimiste face à l'inconnu — d'une manière ou d'une autre.** Que vous intégriez l'optimisme dans les *valeurs initiales* (stratégie 2), dans le *choix de l'action* (stratégie 3) ou dans une *récompense auto-générée* (stratégies 4-5), le fil conducteur est le suivant : *rendre l'inexploré attractif*, puis laisser l'apprentissage vous y mener.

3. **Sur une grille à récompenses éparses, les quatre "vraies" stratégies fonctionnent — et le bonus d'erreur de prédiction y arrive le plus vite**, car il produit le signal "ceci est nouveau" le plus net.

4. **Sur une chaîne *profonde*, où l'optimisme doit voyager loin, le champion incontesté est l'initialisation optimiste.** Elle propage l'optimisme à travers la fonction de valeur gratuitement. UCB s'effondre en premier (son bonus n'entre jamais dans la fonction de valeur, il ne peut donc pas *planifier* en profondeur). Les bonus de récompense s'en sortent mieux — ils se propagent — mais le Q-learning tabulaire classique est trop lent pour pousser cet optimisme jusqu'au bout d'une longue chaîne avant que le bonus ne s'estompe.

5. **Ce dernier point est exactement la raison pour laquelle le passage de l'exploration profonde aux pixels a nécessité plus de puissance** — DQN bootstrapé, RND avec un vrai réseau de neurones (pour que l'optimisme se *généralise* aux états similaires au lieu de se propager cellule par cellule), Go-Explore (se souvenir littéralement et retourner aux états prometteurs). Ces "jouets" tabulaires vous montrent les *principes* ; les systèmes réels sont ces mêmes principes couplés à un réseau qui généralise.

## Mots-clés à retenir

| Mot | Signification |
|------|---------|
| **Compromis exploration–exploitation** | Essayer de nouvelles choses vs capitaliser sur ce que l'on sait — la tension centrale du RL |
| **Dithering (Hésitation/Tâtonnement)** | "Exploration" en ajoutant un bruit aléatoire aux actions (ε-greedy, bruit gaussien) — inefficace sur les tâches ardues |
| **Optimisme face à l'incertitude** | Le principe directeur : traiter l'inconnu comme s'il était excellent jusqu'à vérification |
| **Initialisation optimiste** | Mettre en œuvre ce principe en commençant toutes les valeurs au retour maximum possible |
| **UCB** | Upper Confidence Bound : choisir `argmax (valeur + bonus qui diminue avec le nombre de visites)` |
| **Exploration profonde** | Exploration nécessitant une longue séquence *cohérente* d'actions "inhabituelles", pas seulement une seule |
| **Recuit (annealing) du `beta`** | Diminuer le poids de la curiosité au fil du temps pour que l'agent finisse par arrêter d'explorer et exploite |

## Résumé en une phrase

> **ε-greedy n'est que du bruit ; toute véritable stratégie d'exploration consiste à rendre l'inexploré attractif — via des valeurs optimistes, un bonus de choix d'action ou une récompense de nouveauté auto-générée — et le bon choix dépend de si votre récompense est simplement *éparse* (comme trouver un trésor caché dans un champ plat) ou véritablement *profonde* (comme un code de coffre-fort nécessitant une longue séquence précise de choix spécifiques).**
