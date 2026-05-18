# Addestramento su Montezuma's Revenge 🏛️🔑

## Perché questo gioco è famoso (nei circoli RL) {#why-this-game-is-famous-in-rl-circles}

Nel 2015, il DQN di DeepMind imparò a giocare a dozzine di giochi Atari a un livello
sovrumano partendo dai pixel grezzi. Fu un evento da prima pagina. Ma sepolto nella
tabella dei risultati c'era un gioco in cui il DQN ottenne il punteggio di **0** — lo stesso punteggio che si ottiene
non facendo nulla: **Montezuma's Revenge**.

Perché? Guarda cosa ti chiede il gioco nella primissima stanza:

1. Scendi una scala.
2. Cammina lungo una sporgenza.
3. Salta sopra un teschio che rotola (sbaglia il tempo → muori).
4. Sali un'altra scala.
5. Afferra la chiave.

Sono circa **100 pressioni precise di pulsanti**, e il gioco non ti dà
**nemmeno un punto** finché non hai la chiave in mano. Il segnale di ricompensa è un
piatto e uniforme **zero** per tutta la sequenza iniziale.

Un normale agente RL impara adattandosi alle ricompense che effettivamente riceve.
Se la ricompensa è zero ovunque possa arrivare, non c'è *nulla da cui imparare* —
è come cercare di trovare il fondo di una valle perfettamente piatta cercando di
sentire la direzione della discesa. Così il DQN ha continuato a muoversi a vuoto sulla
piattaforma di partenza per sempre. Montezuma è diventato *il* benchmark per l'**esplorazione difficile (hard exploration)**: il gioco che puoi vincere solo se esplori in modo
*intelligente*, non casuale.

La svolta è arrivata nel 2018 con **Random Network Distillation (RND)** —
e il trucco è stato esattamente l'oggetto del punto 1: aggiungere un **bonus di curiosità intrinseca**
in modo che l'agente premi *se stesso* per aver raggiunto schermate nuove,
e improvvisamente ha un segnale denso che lo trascina più in profondità nel livello. RND
ha ottenuto un punteggio sovrumano su Montezuma. (Successivamente: Go-Explore, Agent57, …)

## Esempi di vita reale di ricompensa sparsa "stile Montezuma" {#real-life-examples-of-montezuma-style-sparse-reward}

- **Una serratura a combinazione / una caccia al tesoro con indizi criptici.** Nessun credito
  parziale. Sei a zero finché non trovi improvvisamente il premio.
- **Far pubblicare un articolo scientifico o portare una startup alla redditività.** Mesi di
  nessuna ricompensa esterna, poi (forse) una grande.
- **Un percorso per una speedrun di un videogioco.** Dozzine di input precisi al fotogramma
  in fila senza alcun feedback finché il trucco non funziona o fallisce.
- **Escape room.** La stanza non ti dice quasi nulla finché non hai concatenato
  diverse scoperte tra loro.

In tutti questi casi, "provare roba a caso" è inutile. Devi
esplorare *sistematicamente* — e un segnale interno del tipo "ooh, questo è nuovo, continua così"
è ciò che ti mantiene sistematico.

## Perché qui non ci addestriamo sul Montezuma originale a pixel {#why-we-dont-actually-train-on-pixel-montezuma-here}

Fare la cosa *reale* in modo appropriato significa:

- una rete convoluzionale per vedere lo schermo RGB 210×160,
- frame-stacking (così l'agente può capire in che direzione si muove il teschio),
- un modulo RND (altre due reti: un "target" casuale fisso e un
  "predictor" addestrato),
- e **decine di milioni di fotogrammi (frame) dell'ambiente** — molte ore di GPU.

Questo è un progetto di ricerca, non uno script didattico. Quindi `montezuma_revenge.py`
fa invece due cose oneste:

### 1. "Tocca" il gioco reale (se `ale-py` è installato) {#1-it-touches-the-real-game-if-ale-py-is-installed}

Carica `ALE/MontezumaRevenge-v5` tramite Gymnasium, esegue un **agente
casuale uniforme per 2000 passi** e riporta la ricompensa totale del gioco. Il
numero che stampa è quasi sempre **0.0** — la frase astratta "ricompensa sparsa"
trasformata in un fatto concreto che puoi verificare tu stesso. Se il pacchetto
Atari non è installato, stampa il comando `pip install` e prosegue.

### 2. Addestra un agente tabellare su un *modello in scala*: `MiniMontezumaEnv` {#2-it-trains-a-tabular-agent-on-a-scale-model-minimontezumaenv}

Questo è un minuscolo mondo a griglia con lo stesso *scheletro* della prima stanza di
Montezuma:

```
###############
#S....#.......#
#.....#.......#
#.....#...T...#     S = inizio (start)
#.....D.......#     K = chiave (key)     D = porta (door - attraversabile solo con la chiave)
#..K..#.......#     T = tesoro (treasure - l'UNICA casella che dà una ricompensa)
###############
```

Per vincere devi: camminare fino alla **chiave** (~6 mosse), raccoglierla; camminare fino alla
**porta** (~4 mosse) — che ora si apre; attraversarla e raggiungere il
**tesoro** (~5 mosse). Circa **15 mosse perfette**, con **zero feedback
fino al tesoro**. Il flag `has_key` fa parte dello stato dell'agente, quindi
una volta afferrata la chiave c'è un'intera seconda stanza di *nuovi* stati da
scoprire — proprio come nuove schermate che si aprono nel gioco reale.

Addestriamo quindi un semplice **Q-learner tabellare** due volte:

| Agente | Risultato su MiniMontezuma |
|-------|--------------------------|
| **senza curiosità (epsilon-greedy)** | Il ritorno rimane a **0** per tutti i 1.500 episodi. Non raggiunge mai nemmeno la chiave. (Ti suona familiare? È il DQN sul gioco reale.) |
| **con un bonus di curiosità basato sull'errore di previsione** | Raggiunge il tesoro entro ~20–25 episodi e poi impara il **percorso ottimale di 15 passi**. (Questa è l'idea di RND, rimpicciolita per adattarsi a una tabella Q.) |

La figura mostra le due curve di apprendimento affiancate, oltre al percorso
effettivo appreso dall'agente curioso, disegnato sulla griglia (inizio → chiave → porta →
tesoro). Lo script stampa anche quel percorso come fotogrammi ASCII.

## La lezione {#the-lesson}

> **La "ricompensa sparsa" non è una bizzarria di un unico strano gioco Atari — è l'impostazione
> predefinita in qualsiasi mondo in cui il successo richiede una sequenza lunga e specifica di
> azioni.** Un agente basato solo sulla ricompensa (DQN standard) letteralmente non può
> iniziare: non c'è un gradiente da seguire. Un bonus di curiosità ne crea uno —
> un segnale denso, auto-generato, del tipo "questo è nuovo, continua così" — ed
> è quel segnale che trasporta l'agente attraverso il deserto degli zeri fino alla
> prima vera ricompensa. Tutto ciò che viene dopo (RND, Go-Explore, Agent57) è una
> versione in scala reale, basata su reti neurali, della stessa mossa.

## Parole chiave da ricordare {#key-words-to-remember}

| Parola | Significato |
|------|---------|
| **Esplorazione difficile (Hard exploration)** | Problemi in cui si ha successo solo esplorando in modo intelligente; l'esplorazione casuale fallisce |
| **Ricompensa sparsa** | La ricompensa è zero quasi ovunque; si ottiene solo dopo una lunga sequenza corretta |
| **Montezuma's Revenge** | Il gioco Atari in cui i classici agenti deep-RL (DQN, A3C) hanno ottenuto 0 — il benchmark canonico per l'esplorazione difficile |
| **RND (Random Network Distillation)** | Il metodo del 2018 che ha battuto Montezuma usando un bonus di curiosità basato sull'errore di previsione |
| **Go-Explore** | "Ricorda gli stati promettenti, torna ad essi, poi esplora da lì" — un altro sistema che ha battuto Montezuma |
| **Modello in scala** | Un ambiente piccolo ed economico che mantiene la *struttura* di un problema difficile in modo da poterlo studiare rapidamente |

## Riassunto in una frase {#one-sentence-summary}

> **Montezuma's Revenge è il gioco che ha insegnato all'RL che "le ricompense che non ricevi mai
> non possono insegnarti nulla" — e la soluzione, allora come oggi, è un
> bonus di curiosità che permette all'agente di premiarsi da solo per l'esplorazione finché non
> trova il vero premio.**
