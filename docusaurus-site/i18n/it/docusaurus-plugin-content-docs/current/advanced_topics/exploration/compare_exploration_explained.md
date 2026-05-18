# Confronto tra Strategie di Esplorazione 🔦

## Il problema in una frase {#the-one-sentence-problem}

Un agente RL deve fare due cose che spingono in direzioni opposte:

- **Sfruttamento (Exploit)**: fare la cosa che ha funzionato meglio finora.
- **Esplorazione (Explore)**: provare qualcosa di nuovo, nel caso sia ancora meglio.

Se ti inclini troppo verso lo sfruttamento, ti accontenterai felicemente di una routine
mediocre per sempre. Se ti inclini troppo verso l'esplorazione, non raccoglierai mai i frutti. *Come*
esplori — non solo *se* lo fai — è ciò che separa un agente che risolve
Montezuma's Revenge da uno che ottiene un punteggio pari a zero.

Questo script mette **cinque** strategie di esplorazione a confronto sugli stessi
due compiti difficili, così puoi vederne le diverse personalità.

## Analogia con la vita reale: scegliere dove pranzare {#real-life-analogy-picking-a-lunch-spot}

Ti sei appena trasferito in una nuova città con 200 ristoranti.

- **ε-greedy** = "Vado nel mio preferito attuale, ma una volta ogni dieci giorni lancio
  un dado e scelgo un ristorante *totalmente a caso*." Campionerai ampiamente ma
  *senza meta* — e continuerai a tornare in posti che già odiavi.
- **Inizializzazione ottimistica** = "Assumo che *ogni* ristorante che non ho
  provato sia il migliore in città finché non viene dimostrato il contrario." Proverai
  metodicamente tutti i 200, eliminando ognuno man mano che la realtà ti delude —
  e troverai i posti davvero fantastici velocemente.
- **UCB (Upper Confidence Bound)** = "Preferisco il mio preferito, ma do un *bonus* ai posti che ho provato
  appena — meno ne so, più grande è il bonus." Questa strategia è intelligente su *quale*
  posto sconosciuto provare oggi, ma ogni decisione è locale:
  sceglie l'opzione che appare migliore *in questo momento* senza pianificare un percorso
  attraverso interi quartieri inesplorati. Non penserà "dovrei attraversare la città verso il lato est, perché ci sono venti posti non provati concentrati lì" — ogni ristorante viene valutato isolatamente, passo dopo passo.
- **Bonus di ricompensa basato sul conteggio** = come l'UCB, ma *ti godi la novità stessa* — un pasto in un posto nuovo di zecca è intrinsecamente soddisfacente, e
  quella soddisfazione modella il tuo piano a lungo termine su quali quartieri
  visitare.
- **Bonus di ricompensa basato sull'errore di previsione** = "Provo un brivido per un pasto che
  mi ha *sorpreso* — qualcosa che non avrei potuto prevedere." Un posto nuovo che
  si rivela esattamente come previsto? Mah. Uno che è selvaggiamente diverso dal
  tuo modello mentale? Affascinante, e aggiorni il tuo piano per cercarne altri
  simili.

## Le cinque strategie (tutte in `compare_exploration.py`) {#the-five-strategies-all-in-compare_explorationpy}

### 1. ε-greedy — il default, ed è *esitazione (dithering)*, non esplorazione {#1-ε-greedy--the-default-and-its-dithering-not-exploring}

Agisci in modo greedy, ma con probabilità ε compi un'azione uniformemente casuale. È
il baseline standard in DQN e simili. Il suo difetto fatale nei compiti difficili:
**ogni passo è un lancio di moneta indipendente.** Per percorrere una catena di `N`
mosse corrette hai bisogno che la moneta cada dal lato giusto `N` volte di fila — il che è
esponenzialmente improbabile. ε-greedy è un *agitarsi*, non un *esplorare*.

### 2. Inizializzazione ottimistica — "innocente fino a prova contraria" {#2-optimistic-initialisation--innocent-until-proven-boring}

Fai iniziare *ogni* valore Q al più grande ritorno possibile,
`R_max / (1 − γ)`. Ora un'azione che l'agente non ha mai provato sembra la
cosa migliore del mondo, quindi la politica **greedy** è costretta ad andare a provarla;
solo dopo averla visitata il valore scende verso la verità. L'ottimismo
riguardo alle regioni *non* provate si **propaga automaticamente attraverso la funzione
valore** (tramite il bootstrap del Q-learning), così l'agente è trascinato, passo
dopo passo, verso le parti del mondo che non ha visto. Quasi gratuito, nessun
costo extra — e, come vedrai, il più forte esploratore *profondo* in un piccolo
mondo tabellare.

### 3. Selezione dell'azione in stile UCB — bonus nella *scelta*, non nella *ricompensa* {#3-ucb-style-action-selection--bonus-in-the-choice-not-the-reward}

Scegli `argmax_a [ Q(s,a) + c·√(ln t / N(s,a)) ]`: preferisci le azioni di
alto valore, ma gonfia quelle che hai provato raramente. Famoso dai
bandit multi-braccio (multi-armed bandits). L'inghippo: il bonus vive solo nella **regola di selezione dell'azione**,
mai nella ricompensa — quindi *non* fluisce attraverso la funzione valore.
L'UCB è ottimo per "assicurarsi di aver provato ogni azione in *questo* stato" ma
debole per "pianificare un percorso verso una regione inesplorata lontana".

### 4. Bonus di **ricompensa** basato sul conteggio — curiosità, la versione classica {#4-count-based-reward-bonus--curiosity-the-classic-version}

Aggiungi `1/√(N(s,a))` alla **ricompensa** (con un peso `beta` che decade).
Poiché è nella ricompensa, il Q-learning *propaga* questo segnale: gli stati che
portano verso regioni nuove diventano preziosi. Questa è l'idea MBIE-EB / il classico
"bonus di esplorazione" — e la prima parte del punto 1 del lavoro.

### 5. Bonus di **ricompensa** basato sull'errore di previsione — curiosità, la versione ICM/RND {#5-prediction-error-reward-bonus--curiosity-the-icmrnd-version}

Aggiungi `−log P(s'|s,a)` da un piccolo modello predittivo appreso alla ricompensa
(sempre con `beta` decrescente). Il segnale di novità più nitido dei cinque: in
un mondo deterministico, la sorpresa di una transizione scende a ~0 nel momento
in cui l'hai vista una volta, invece di svanire lentamente come `1/√N`. Il
cugino tabellare di ICM / RND — la seconda parte del punto 1 del lavoro.

## I due compiti di test {#the-two-test-tasks}

- **Compito A — MiniMontezuma**: il mondo a griglia chiave→porta→tesoro, ricompensa solo
  al tesoro (a circa 15 mosse perfette di distanza). Verifica se "riesci a sopravvivere a una lunga catena a ricompensa sparsa".
- **Compito B — DeepSea(N)**: la classica catena di esplorazione profonda, eseguita a
  lunghezze `N = 5, 8, 11, 14`. La ricompensa si nasconde dietro `N` mosse corrette,
  ognuna con un piccolo costo immediato — così un agente miope impara a evitare il
  costo e non trova mai il premio. Verifica se "la tua strategia funziona ancora man mano che la catena si allunga".

## Cosa succede realmente (eseguilo e vedrai) {#what-actually-happens-run-it-and-see}

**Compito A — MiniMontezuma:**

| Strategia | Primo tesoro | Tasso di soluzione finale |
|----------|---------------:|-----------------:|
| ε-greedy | mai | 0.00 |
| inizializzazione ottimist. | ~episodio 1 | 1.00 |
| selezione azione UCB | ~episodio 3 | ~0.95 |
| bonus ricompensa conteggio | ~episodio 82 | ~0.41 |
| bonus ricompensa previs. | ~episodio 23 | 1.00 |

**Compito B — DeepSea, frazione di seed che hanno trovato la ricompensa:**

| Strategia | N=5 | N=8 | N=11 | N=14 |
|----------|----:|----:|-----:|-----:|
| ε-greedy | 0 | 0 | 0 | 0 |
| inizializzazione ottimist. | 1.0 | 1.0 | 1.0 | 1.0 |
| selezione azione UCB | 1.0 | 1.0 | 0.0 | 0.0 |
| bonus ricompensa conteggio | 1.0 | 1.0 | ~0.1 | 0.0 |
| bonus ricompensa previs. | ~0.9 | ~0.8 | ~0.9 | ~0.2 |

*(I numeri variano leggermente con i seed casuali, ma la tendenza è solida.)*

## Le lezioni {#the-lessons}

1. **ε-greedy non è esplorazione.** Non risolve mai *nessuno* dei due compiti difficili.
   L'esitazione casuale (dithering) semplicemente non riesce a percorrere lunghe sequenze corrette. (Eppure è ancora il default in molto codice — perché sui compiti *facili* è sufficiente e semplicissimo.)

2. **La vera esplorazione significa essere ottimisti riguardo all'ignoto — in un modo
   o nell'altro.** Sia che tu inserisca l'ottimismo nei *valori iniziali*
   (strategia 2), nella *scelta dell'azione* (strategia 3), o in una
   *ricompensa auto-generata* (strategie 4–5), il filo conduttore è: *fai apparire attraente ciò che non è esplorato*, quindi lascia che l'apprendimento ti porti lì.

3. **Su una griglia a ricompensa sparsa, tutte e quattro le strategie "reali" funzionano — e il
   bonus basato sull'errore di previsione ci arriva più velocemente**, perché produce il
   segnale "questo è nuovo" più nitido.

4. **Su una catena *profonda*, dove l'ottimismo deve viaggiare per un lungo percorso, il
   campione indiscusso è l'inizializzazione ottimistica.** Propaga l'ottimismo
   attraverso la funzione valore gratuitamente. L'UCB fallisce per primo (il suo bonus
   non entra mai nella funzione valore, quindi non può *pianificare* in profondità). I
   bonus di ricompensa fanno di meglio — loro *si* propagano — ma il semplice Q-learning tabellare
   è lento a spingere quell'ottimismo fino in fondo a una lunga catena prima che il
   bonus decada.

5. **Quest'ultimo punto è esattamente il motivo per cui scalare l'esplorazione profonda ai pixel
   ha richiesto potenza di fuoco extra** — DQN bootstrap, RND con una vera rete neurale
   (così l'ottimismo *generalizza* tra stati simili invece di
   propagarsi una cella alla volta), Go-Explore (ricorda letteralmente e
   torna agli stati promettenti). Gli esempi tabellari qui mostrano i
   *principi*; i sistemi reali sono questi stessi principi uniti a una rete
   che generalizza.

## Parole chiave da ricordare {#key-words-to-remember}

| Parola | Significato |
|------|---------|
| **Compromesso esplorazione–sfruttamento (Exploration–exploitation trade-off)** | Provare cose nuove vs. sfruttare ciò che si sa — la tensione centrale nell'RL |
| **Esitazione (Dithering)** | "Esplorazione" aggiungendo rumore casuale alle azioni (ε-greedy, rumore gaussiano) — debole sui compiti difficili |
| **Ottimismo di fronte all'incertezza** | Il principio generale: trattare l'ignoto come se fosse fantastico finché non si è verificato |
| **Inizializzazione ottimistica** | Implementazione di quel principio impostando tutti i valori iniziali al massimo ritorno possibile |
| **UCB** | Upper Confidence Bound: scegliere `argmax (valore + bonus che si riduce con il conteggio delle visite)` |
| **Esplorazione profonda (Deep exploration)** | Esplorazione che richiede una lunga sequenza *coerente* di azioni "insolite", non solo una |
| **Annealing di `beta`** | Riduzione del peso della curiosità nel tempo in modo che l'agente alla fine smetta di esplorare e sfrutti ciò che ha appreso |

## Riassunto in una frase {#one-sentence-summary}

> **ε-greedy è solo rumore; ogni vera strategia di esplorazione funziona rendendo
> attraente l'inesplorato — tramite valori ottimistici, un bonus nella scelta dell'azione,
> o una ricompensa di novità auto-generata — e la scelta giusta dipende
> dal fatto che la ricompensa sia solo *sparsa* (come trovare un singolo premio nascosto
> in un campo piatto) o genuinamente *profonda* (come una serratura a combinazione che richiede una
> sequenza lunga e precisa di scelte specifiche per essere aperta).**
