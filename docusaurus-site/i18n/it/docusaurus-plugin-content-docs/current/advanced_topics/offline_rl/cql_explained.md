# Conservative Q-Learning (CQL) 🛡️

## Cos'è?

Immagina di imparare a investire denaro leggendo un enorme registro di passate
operazioni di borsa effettuate da altre persone. Il registro riporta acquisti, vendite e mantenimenti —
ma **nessuna traccia di operazioni che nessuno ha effettivamente compiuto**.

Ora immagina che uno studente troppo sicuro di sé guardi il registro e dica:
*"E se qualcuno avesse comprato biglietti della lotteria ogni lunedì? Sarebbe stata
un'operazione incredibile!"*

Il problema: **il registro non ha dati sull'acquisto di biglietti della lotteria il lunedì**, quindi lo
studente sta solo allucinando. Eppure, quell'operazione allucinata sembra ottima sulla
carta, quindi la "politica" dello studente continua a volerla compiere.

Questo problema di allucinazione è lo **spostamento della distribuzione (distribution shift)**: un apprendista offline
ama le azioni che il dataset non ha mai testato, perché non ci sono dati per
contraddire l'ottimismo. CQL è la cura.

---

## Perché il Q-Learning fallisce in modalità Offline

L'obiettivo (target) del normale Q-learning è:

```
target(s, a) = r + γ · max_{a'} Q(s', a')
```

Quel `max_{a'}` è il pericolo. Quando il dataset non ha mai registrato l'azione `a'`
nello stato `s'`, la rete si limita a *indovinare* un valore Q — e le reti neurali
tendono a **sovrastimare** Q per input mai visti. Il target eredita la
sovrastima, la rete impara a prevedere quel numero più grande e, al
passo successivo, effettuiamo un'**estrapolazione** (proiettando ancora più in là di quanto i
dati supportino) ancora più alta. La politica insegue un fantasma.

Se potessi continuare a raccogliere più dati, questo si correggerebbe da solo (l'azione
fantasma si rivelerebbe pessima nella realtà). Ma **nell'RL offline non
puoi raccogliere altri dati.** Il fantasma è per sempre.

---

## Il trucco di CQL

CQL (Kumar et al., 2020) aggiunge un termine di penalità alla perdita (loss):

```
cql_loss(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Due parti:

1. **`log Σ_a exp Q(s, a)`** (leggi: *"log-sum-exp su tutte le azioni"*) è un
   **massimo morbido (soft maximum)** su tutte le azioni — un'approssimazione fluida e differenziabile
   di `max` che considera ogni azione contemporaneamente invece di
   selezionare rigidamente un vincitore. Penalizzarlo riduce i valori Q
   **su tutta la linea** (spingendo verso il basso tutte le previsioni
   uniformemente) — specialmente per le azioni con il valore Q *più alto*, che è esattamente
   dove vivono le allucinazioni.
2. **`- Q(s, a_dataset)`** premia un Q alto sull'azione che il dataset
   ha effettivamente registrato — proteggendo i valori Q "in-distribution" dalla
   riduzione sopra citata.

Effetto netto: **Q viene spinto verso il basso sulle azioni non viste, e verso l'alto sulle azioni
viste.** Il valore Q appreso diventa un *limite inferiore (lower bound)* sul valore Q reale. La
politica **`argmax`** (la regola che sceglie semplicemente l'azione con il Q più alto)
smette di inseguire fantasmi.

Perdita completa:

```
L  =  Bellman_MSE   +   α · cql_loss
```

(Dove **`Bellman_MSE`** è l'errore standard del normale Q-learning,
che misura quanto l'ipotesi attuale della rete dissenta dalla sua stessa
ipotesi futura).

`α` è la manopola del conservatorismo. Troppo piccolo → lo spostamento della distribuzione ritorna.
Troppo grande → l'agente è così conservatore che non migliora mai oltre i
dati.

---

## Esempi di Vita Reale

- **Allenatore di scacchi conservatore.** Puoi imparare solo da partite già
  giocate. Un allenatore spericolato direbbe "questa mossa ipotetica senza
  precedenti potrebbe essere geniale!". CQL è l'allenatore che dice "non abbiamo
  dati su questo — atteniamoci alle mosse che i veri giocatori hanno provato".
- **Scelte del menu del ristorante.** Le recensioni su Yelp non coprono mai i
  piatti fuori menu. Una politica ingenua raccomanderebbe i piatti fuori menu
  basandosi su valutazioni a cinque stelle allucinate. CQL raccomanda solo ciò che è
  stato ordinato abbastanza volte da essere affidabile.
- **Presa robotica dai log.** Il robot ha video di prese di tazze,
  bottiglie e libri — ma mai di un coltello. CQL si rifiuta di raccomandare con sicurezza
  di "afferrare il coltello dalla lama".

---

## Cosa fa il nostro codice

Lo script `cql.py`:

1. **Carica i quattro dataset** costruiti da `d4rl_dataset.py`.
2. **Sceglie `medium-replay`** come set di addestramento — è il più realistico
   (qualità mista) e il più dannoso per i metodi ingenui.
3. **Addestra tre agenti puramente offline**, in condizioni identiche eccetto
   per `α`:
   - `α = 0`   →  DQN offline ingenuo (nessuna penalità — il baseline difettoso)
   - `α = 1.0` →  CQL lieve
   - `α = 5.0` →  CQL forte
4. **Valuta ciascuno ogni 2.500 passi di gradiente** eseguendo rollout in modo "greedy"
   nell'ambiente reale (10 episodi). Questo è l' *unico* contatto con l'ambiente;
   l'addestramento stesso non vede mai l'ambiente.
5. **Genera i grafici delle curve di apprendimento** in `outputs/cql.png`.

---

## Cosa dovresti vedere

Un'esecuzione tipica stampa qualcosa di simile:

```
Final evaluation returns (avg over 10 episodes, greedy):
  naive offline DQN (alpha=0)         ->  ~30-150  (instabile; spesso crasha)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

Nel grafico delle curve di apprendimento:

- La **curva rossa** (`α = 0`) sale presto e poi spesso **precipita**
  una volta che le allucinazioni da spostamento della distribuzione infettano l'**obiettivo di Bellman (Bellman target)**
  (il numero che usiamo come "risposta corretta" quando addestriamo la rete Q:
  `r + γ · max Q(s', ·)`). Quando i valori Q fantasma inquinano quell'obiettivo,
  ogni passo di gradiente peggiora le cose. La **perdita di Bellman (Bellman loss)** (l'MSE
  tra la previsione della rete Q e il target di Bellman) sembra a posto —
  questa è la **perfidia** del problema: la rete è perfettamente
  coerente con le proprie convinzioni errate, quindi la perdita non dà alcun avvertimento.
- La **curva arancione** (`α = 1.0`) sale più lentamente ma **rimane alta**.
- La **curva verde** (`α = 5.0`) è la più stabile e solitamente la migliore.

Il pannello della perdita di Bellman mostra un altro indizio: la perdita del DQN ingenuo può rimanere
piccola mentre la sua politica è terribile, perché la rete è internamente
coerente con le proprie allucinazioni.

---

## La posizione di CQL nel campo

CQL è stato un passo *molto importante* perché ha fornito una soluzione semplice e basata su principi allo
spostamento della distribuzione. La stirpe:

```
DQN (online)
   │
   ▼
DQN offline ingenuo  ── fallisce a causa dello spostamento della distribuzione
   │
   ▼
CQL (Kumar 2020)    ── aggiunge una penalità conservativa: Q è un limite inferiore
   │
   ▼
IQL (Kostrikov 2021)  ── evita del tutto di interrogare Q su azioni non viste
   │
   ▼
Decision Transformer (Chen 2021)  ── salta completamente Q, tratta l'RL come modellazione di sequenze
                                      (prevede l'*azione successiva* dati gli stati passati e
                                       un ritorno totale desiderato, esattamente come un LLM
                                       prevede la parola successiva)
```

Ogni passo in questa stirpe è una risposta diversa alla stessa domanda:
**come faccio a evitare di chiedere alla mia rete Q informazioni su cose che non ha mai visto?**

---

## Parole chiave da ricordare

| Parola | Significato |
|------|---------|
| **Spostamento della distribuzione (Distribution shift)** | La politica addestrata desidera azioni al di fuori dei dati |
| **Fuori distribuzione (OOD)** | Una coppia (s, a) che il dataset non ha mai registrato |
| **Q reale (True Q)** | Il vero ritorno futuro atteso per l'azione `a` nello stato `s`, se potessimo misurarlo perfettamente |
| **Q conservativo (Conservative Q)** | Una funzione Q appresa che cerca di rimanere al di sotto del Q reale invece di promettere troppo |
| **Logsumexp** | Un'approssimazione fluida e differenziabile di `max` |
| **Alpha (α)** | La manopola del conservatorismo di CQL — quanto forte spingere Q verso il basso sulle azioni OOD |

---

## Riassunto in una frase

> **CQL aggiunge una "penalità di pessimismo" che punisce i valori Q elevati sulle azioni
> che il dataset non ha mai provato — così la politica non può innamorarsi delle
> allucinazioni.**
