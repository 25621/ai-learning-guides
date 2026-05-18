# Dataset di Benchmark D4RL 📦

## Cos'è?

Immagina di voler insegnare a un robot a girare i pancake. Lasciarlo esercitare su un
vero fornello per un mese sarebbe lento, pericoloso e costoso. Ma hai
dieci anni di video registrati di chef che girano pancake (alcuni bravi,
alcuni meno, alcuni casuali). Puoi insegnare al robot *solo da quei dati*, senza mai
lasciargli toccare una vera padella?

Questo è l'**apprendimento per rinforzo offline (offline RL)**. L'agente impara da un
dataset fisso di esperienze passate — nessun ambiente dal vivo. La parte più difficile è che
l'agente non può mai *provare* ciò che ha imparato fino alla fine.

Per rendere equo lo studio di questo approccio, la comunità di ricerca aveva bisogno di un *dataset
standard*. Quel dataset è **D4RL** (**D**atasets for **D**eep **D**ata-**D**riven
**R**einforcement **L**earning): una raccolta di transizioni pre-registrate per
compiti di controllo classici, rilasciata dalla UC Berkeley nel 2020. Ogni articolo scientifico si addestra
sugli stessi byte, rendendo i risultati comparabili.

---

## Cosa c'è in un dataset D4RL?

Per ogni compito, D4RL fornisce **quattro livelli di qualità**:

| Livello | Origine dei dati | Perché è importante |
|-------|---------------------------|----------------|
| **random**        | Una politica che sceglie le azioni in modo uniformemente casuale | Caso peggiore: si può comunque imparare qualcosa di utile? |
| **medium**        | Una politica parzialmente addestrata (circa metà del punteggio di un esperto) | Realistico: la maggior parte dei dati registrati è mediocre |
| **expert**        | Una politica quasi convergente | Caso migliore: si riesce a eguagliare la politica sorgente? |
| **medium-replay** | L'*intero buffer di replay* usato per addestrare la politica "medium" | Misto: contiene fallimenti iniziali E successi successivi |

La differenza tra `medium` e `medium-replay` è fondamentale:
- **`medium`** è generato prendendo una singola politica "media" fissa e lasciandole giocare molte partite. Tutti i dati riflettono questo livello di abilità costante e medio.
- **`medium-replay`** è un registro storico. Contiene tutte le esperienze raccolte *durante l'apprendimento* da zero fino al livello "medium". Mescola transizioni **pessime e ottime**
— esattamente come appare un log del mondo reale (i primi goffi tentativi di un robot
*e* il suo successivo comportamento raffinato, tutto in un unico contenitore).

---

## Esempi di vita reale di dataset offline

- **Cartelle cliniche.** Anni di tuple (stato_paziente, trattamento, esito).
  Non puoi randomizzare i trattamenti su persone viventi, ma puoi imparare una
  politica migliore dai log.
- **Log delle chat del servizio clienti.** Milioni di record (messaggio_utente, risposta_agente,
  soddisfazione). Addestra un assistente migliore senza disturbare altri
  utenti.
- **Dati di flotte a guida autonoma.** Ogni auto Tesla / Waymo carica i propri
  viaggi. La flotta è un gigantesco dataset "medium-replay".
- **Sistemi di raccomandazione.** I log dei clic dell'anno scorso sono un dataset congelato:
  non puoi mostrare di nuovo gli stessi annunci agli stessi utenti.

In tutti e quattro i casi, **non puoi chiedere all'ambiente un nuovo campione.** Il
dataset è tutto ciò che hai. Per sempre.

---

## Cosa fa il nostro codice

I veri dataset D4RL sono registrati su compiti di locomozione MuJoCo (Multi-Joint dynamics with
Contact)
(come HalfCheetah, Hopper, Walker2d, Ant — queste sono simulazioni fisiche 3D avanzate in cui robot virtuali imparano a camminare e correre). MuJoCo è pesante da installare, quindi ricreiamo la
**stessa struttura a quattro livelli su CartPole-v1** — l'ambiente standard
per principianti delle fasi precedenti. Le lezioni apprese si trasferiscono direttamente.

Lo script `d4rl_dataset.py`:

1. **Addestra un DQN** (Deep Q-Network, un algoritmo RL standard) su CartPole finché non risolve il compito (ritorno ≥ 475).
2. **Salva due istantanee (checkpoint)** lungo il percorso:
   - "medium" — il momento in cui il ritorno recente ha superato 150
   - "expert" — il momento in cui il ritorno recente ha superato 475
3. **Salva l'intero buffer di replay della politica "medium"** — ogni transizione
   che abbia mai visto. Questo è il nostro dataset "medium-replay".
4. **Esegue tre nuove politiche** per 10.000 transizioni ciascuna:
   - `random`   — casuale uniforme
   - `medium`   — il checkpoint medium + rumore ε=0.10
   - `expert`   — il checkpoint expert + rumore ε=0.02
5. **Salva quattro file `.npz`** (formato array compresso di NumPy) in
   `outputs/`, ciascuno con gli array `obs / action / reward / next_obs / terminal`.

Questi quattro file sono gli input per `cql.py` e `behavioral_cloning.py`.

---

## Cosa dovresti vedere quando lo esegui

Un riassunto testuale stampato sulla console e salvato in
`outputs/d4rl_summary.txt`:

```
dataset         |   N    |  ritorno medio |  min  |  max
------------------------------------------------------------
random          | 10000  |          ~22  |    ~9 |   ~80
medium          | 10000  |         ~180  |   ~50 |  ~500
expert          | 10000  |         ~490  |  ~400 |   500
medium-replay   | 10000  |          ~60  |    ~9 |  ~200
```

Genera anche un istogramma (`outputs/d4rl_returns.png`) che mostra come i
quattro dataset si sovrappongono. Le caratteristiche chiave da notare:

- **Random** si raggruppa intorno a 20 (la lunghezza media di un episodio casuale di CartPole).
- **Expert** si raggruppa al soffitto di 500.
- **Medium** si trova nel mezzo, con un'alta varianza.
- **Medium-replay** ha una lunga coda a destra — consiste principalmente di prime esecuzioni fallite (bassi ritorni) ma ha una coda che si estende verso ritorni più alti man mano che l'agente imparava.

---

## Perché il dataset è importante

Qualunque sia il dataset su cui addestri il tuo algoritmo offline, stai ponendo un
*soffitto* a ciò che è possibile ottenere:

- **Da `expert`** — anche un algoritmo semplice come BC (Behavioral Cloning, ovvero clonazione comportamentale, che copia esattamente i dati) può dare buoni risultati,
  perché tutti i dati sono ottimi.
- **Da `random`** — serve un algoritmo intelligente in grado di *ricucire insieme (stitch together)*
  rare transizioni positive (trovando un percorso verso il successo combinando brevi sequenze di azioni corrette da diversi tentativi). Il BC fallirà completamente.
- **Da `medium-replay`** — il caso più realistico e interessante.
  Gli algoritmi validi (come **CQL** — Conservative Q-Learning, che evita di
  essere troppo fiducioso su azioni che non ha mai visto) possono a volte **superare
  la qualità media dei dati** perché estraggono una struttura da segnali misti. Gli algoritmi semplici (BC) regrediscono verso la media.

Vedremo esattamente questa storia nei prossimi due script.

---

## Parole chiave da ricordare

| Parola | Significato |
|------|---------|
| **RL Offline**         | Addestramento da un dataset fisso; nessuna interazione con l'ambiente consentita |
| **Politica di comportamento (Behaviour policy)** | La politica che ha *prodotto* il dataset |
| **Qualità del dataset**    | Quanto era buona la politica di comportamento (random / medium / expert) |
| **Buffer di replay (Replay buffer)**      | La cronologia completa delle transizioni viste durante un'esecuzione di addestramento |
| **Spostamento della distribuzione (Distribution shift)** | Il divario tra le azioni nel dataset e le azioni che la tua politica addestrata vuole intraprendere. Poiché il dataset non mostra mai cosa succede quando la nuova politica prova qualcosa che non è stato registrato, le stime del valore dell'algoritmo per quelle nuove azioni possono essere pericolosamente errate. |

---

## Riassunto in una frase

> **D4RL trasforma l'RL in un benchmark in stile apprendimento supervisionato: gli stessi byte
> per tutti, nessun trucco nell'ambiente, vince l'algoritmo migliore.**
