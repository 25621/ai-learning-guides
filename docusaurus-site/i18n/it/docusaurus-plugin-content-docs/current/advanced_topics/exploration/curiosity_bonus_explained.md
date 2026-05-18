# Bonus di Curiosità (Motivazione Intrinseca) 🧭

## Cos'è?

Immagina un bambino piccolo lasciato in una stanza nuova. Nessuno lo paga, nessuno
applaude — eppure si dirige subito verso l'armadietto che non ha ancora aperto, il pulsante
che non ha ancora premuto, il giocattolo rumoroso nell'angolo. Sta agendo in base a una
**ricompensa interna**: *"Sembra nuovo. Vai a vedere di cosa si tratta."*

Un **bonus di curiosità** fornisce a un agente di apprendimento per rinforzo la stessa
spinta interna. La vera ricompensa dell'ambiente (la ricompensa "estrinseca" —
punti, denaro, vittoria nel gioco) rimane esattamente com'è. Aggiungiamo solo una
seconda ricompensa, auto-generata, per la visita di elementi che l'agente trova *nuovi*
o *sorprendenti*, e addestriamo sulla somma:

```
ricompensa su cui l'agente impara  =  ricompensa reale  +  beta * bonus di curiosità
```

`beta` è una manopola che inizia con un valore alto (sii curioso!) e si riduce nel tempo
(smetti di gironzolare, vai a mettere a frutto ciò che hai imparato).

## Perché preoccuparsene? Il problema della ricompensa sparsa

I normali agenti RL imparano dalle ricompense che effettivamente ricevono. Questo funziona
benissimo quando le ricompense sono ovunque ("+1 per ogni passo in cui rimani in piedi" in
CartPole). Il sistema fallisce quando la ricompensa è **sparsa** — zero, zero,
zero, ... , zero, e poi finalmente un +1 dopo una lunga e specifica
sequenza di azioni corrette.

Esempi reali di ricompensa sparsa:

- **Montezuma's Revenge** (il gioco Atari): il tuo primo punto arriva solo
  dopo circa 100 mosse precise — scendi da una scala, schiva un teschio, risali,
  afferra una chiave. Fino ad allora il punteggio è un piatto zero.
- **Una serratura a combinazione.** 9.999 codici errati non ti danno nulla; uno solo ti dà
  il premio.
- **Scoperta di farmaci / esperimenti scientifici.** Migliaia di prove fallite,
  poi una che funziona.
- **Scrivere una lunga dimostrazione o un programma.** Nessun credito parziale finché
  l'intera opera non è corretta.

Un agente basato solo sulla ricompensa in questi contesti è come uno studente che si rifiuta di
studiare a meno che non venga pagato per ogni risposta corretta all'esame finale — non
inizierà mai. La curiosità è il bonus che dice *"esplorare è la sua stessa ricompensa,"*
quindi l'agente continua a curiosare finché non inciampa nel vero premio.

## Due varianti della curiosità (entrambe implementate in `curiosity_bonus.py`)
### 1. Novità basata sul conteggio: "Non sono quasi mai stato qui"

Il segnale di novità più semplice possibile. Tieni un conteggio `N(s, a)` di quante
volte hai compiuto l'azione `a` nello stato `s`, e concediti un bonus che
si riduce all'aumentare del conteggio:

```
bonus di curiosità  =  1 / sqrt( N(s, a) + 1 )
```

La prima volta che provi qualcosa: bonus = 1.0. Dopo 100 tentativi: bonus = 0.1.
Dopo 10.000 tentativi: 0.01. L'agente viene premiato per andare dove non è ancora
stato, e l'attrattiva svanisce naturalmente dai percorsi già battuti.

**Analogia con la vita reale:** un turista con una lista di "posti che non ho visitato".
Un quartiere nuovo di zecca? Massima priorità. Il bar dove sei stato cinquanta
volte? Non è più eccitante.

Questa è l'idea più antica del settore (MBIE-EB, UCB). Il suo punto debole: in un
mondo vasto o continuo non visiterai mai *esattamente* lo stesso stato due volte, quindi
il conteggio grezzo sarà sempre 1 — ecco perché esiste la variante successiva.

### 2. Novità basata sull'errore di previsione: "Questo non me lo aspettavo"

Questa è l'idea alla base del famoso **ICM** (Intrinsic Curiosity Module,
Pathak et al. 2017) e del suo "cugino" **RND** (Random Network Distillation,
Burda et al. 2018). Invece di contare, l'agente mantiene un piccolo
**modello che cerca di prevedere cosa succederà dopo** — "se sono qui e faccio
questo, dove andrò a finire?" — e si premia in base a **quanto il modello
si è sbagliato**:

```
bonus di curiosità  =  sorpresa  =  -log P( lo stato che ho effettivamente raggiunto | dove ero, cosa ho fatto )
```

- Una situazione che il modello non ha mai visto → preannuncia male → grande sorpresa
  → grande bonus → "vai a esplorare lì!"
- Una situazione che il modello ha visto cento volte → preannuncia perfettamente →
  zero sorpresa → zero bonus → "già visto, capito, andiamo avanti."

**Analogia con la vita reale:** un bambino che impara come funziona il mondo giocando.
Spingere un bicchiere giù dal tavolo per la *prima* volta è affascinante (si è
frantumato!). La centesima volta, sapevi già che si sarebbe frantumato — non è
interessante. Curiosità = il divario tra ciò che ti aspettavi e ciò che è
accaduto.

Nel nostro codice tabellare il "modello" è solo una tabella di conteggi delle transizioni, e
"quanto si è sbagliato" è la sorpresa `-log P`. I veri ICM/RND usano reti
neurali in modo che la stessa idea funzioni sui pixel grezzi — ma il principio è
identico.

> **Perché due versioni?** Quella basata sul conteggio è semplicissima e un ottimo baseline.
> Quella basata sull'errore di previsione scala verso mondi vasti che non si ripetono mai e fornisce un
> segnale più *nitido*: in un ambiente deterministico, una volta vista una
> transizione, la sorpresa scende istantaneamente a ~0, mentre un bonus di
> conteggio svanisce solo lentamente come `1/sqrt(N)`. Nei nostri esperimenti l'agente
> con errore di previsione risolve MiniMontezuma in un paio di dozzine di episodi;
> l'agente basato sul conteggio ci arriva anch'esso, solo più lentamente e con meno affidabilità.

## Cosa fa il nostro codice

`curiosity_bonus.py` addestra un semplice **Q-learner tabellare** su
`MiniMontezumaEnv` — un minuscolo mondo a griglia di due stanze dove devi camminare fino a una
**chiave**, raccoglierla (ora la **porta** si apre), attraversarla e raggiungere il
**tesoro**. La ricompensa (+1) appare *solo* al tesoro, dopo circa 15
mosse perfette. Esegue tre agenti e ne traccia i grafici:

| Agente | Cosa fa su MiniMontezuma |
|-------|-------------------------------|
| **epsilon-greedy (nessuna curiosità)** | Vaga vicino all'inizio, *mai* raggiunge la chiave, il punteggio rimane 0 per sempre. |
| **bonus basato sul conteggio** | Trova la chiave in modo affidabile; percorre l'intera catena fino al tesoro forse nel ~40% degli episodi. Funziona — solo un po' rumoroso. |
| **bonus basato sull'errore di previsione** | Raggiunge per la prima volta la chiave *e* il tesoro entro ~20–25 episodi; man mano che `beta` decade, converge a risolvere il gioco in ogni episodio. |

La figura mostra:
- una curva di apprendimento: *P(raggiungere il tesoro)* durante l'addestramento,
- una seconda curva per la pietra miliare intermedia *P(raccogliere la chiave)*,
- e **mappe di calore (heat-maps) delle visite degli stati** della griglia — l'agente senza curiosità
  è una macchia densa vicino all'inizio; gli agenti curiosi inondano *entrambe* le stanze.

## Il meccanismo in un'immagine

```
            senza curiosità                       con bonus di curiosità
   premio:  0 0 0 0 0 0 0 0 ... 0  (+1?)        0 0 0 0 0 0 0 0 ... 0  (+1!)
            └──── nulla da cui imparare ──┘     └ + 0.4 0.3 0.9 0.2 ... ┘  (auto-generata,
                                                  densa, punta "verso le novità")
   risultato: cammino casuale, mai trovato +1   esplora il mondo sistematicamente,
                                                 inciampa nel +1, poi il bonus svanisce
```

Il bonus di curiosità trasforma *"non l'ho mai visto"* in una ricompensa, così l'agente
**si spinge deliberatamente in territori inesplorati** invece di
agitarsi casualmente. E poiché il bonus si riduce man mano che le cose diventano
familiari (e `beta` decade), una volta trovata la vera ricompensa l'agente
smette naturalmente di gironzolare e inizia a sfruttare la conoscenza acquisita.

## Alcuni onesti avvertimenti

- **Il problema della "TV rumorosa" (noisy-TV problem).** Un agente basato sull'errore di previsione può rimanere ipnotizzato
  da una fonte di pura casualità (una TV che mostra statico, il lancio di dadi) — non
  potrà *mai* prevederla, quindi la sorpresa non svanirà mai. Il vero trucco di ICM è
  prevedere in uno *spazio di caratteristiche appreso* che ignora le cose che l'agente
  non può controllare; RND lo evita in modo diverso. Il nostro mondo a griglia
  deterministico non ha TV rumorose, quindi non riscontriamo questo problema.
- **La curiosità è un mezzo, non un fine.** Ecco perché `beta` decade. Un agente
  che rimane massimamente curioso per sempre non si stabilizzerà mai per arrivare a
  *vincere* davvero.
- **Scalare l'esplorazione profonda è ancora difficile.** Un bonus nella ricompensa aiuta,
  ma il semplice Q-learning tabellare è lento a propagare l'ottimismo risultante
  lungo una catena complessa (vedi `compare_exploration.py`). Risolvere il pixel-Montezuma
  ha richiesto una potenza di fuoco extra — RND con rete neurale, DQN bootstrap, Go-Explore.

## Parole chiave da ricordare

| Parola | Significato |
|------|---------|
| **Ricompensa intrinseca** | Una ricompensa che l'agente genera per se stesso, distinta dalla ricompensa dell'ambiente |
| **Ricompensa estrinseca** | La vera ricompensa dell'ambiente (punti, vittoria/sconfitta) |
| **Ricompensa sparsa** | La ricompensa è zero quasi ovunque; si ottiene solo dopo una lunga sequenza corretta |
| **Novità / sorpresa** | Quanto uno stato (o una transizione) è nuovo o inaspettato — ciò che la curiosità premia |
| **Bonus basato sul conteggio** | Novità ≈ `1/sqrt(conteggio visite)` — il classico bonus di esplorazione |
| **ICM** | Intrinsic Curiosity Module: novità = errore di previsione di un modello predittivo (in uno spazio di caratteristiche appreso) |
| **`beta`** | Il peso del bonus di curiosità; solitamente ridotto verso 0 così l'agente alla fine sfrutta ciò che ha appreso |

## Riassunto in una frase

> **Un bonus di curiosità è una ricompensa auto-attribuita per la novità — genera un
> segnale denso del tipo "vai a esplorare laggiù" che trascina l'agente attraverso
> mondi con ricompensa sparsa che altrimenti non risolverebbe mai, per poi svanire
> gentilmente una volta che tutto è diventato familiare.**
