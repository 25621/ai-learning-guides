# Dyna-Q: Imparare Più Velocemente Immaginando 🧠

## Cos'è?

Immagina una bambina di nome Mia che impara a orientarsi nella sua nuova scuola. Ogni giorno cammina per i corridoi e scopre cose nuove: "La biblioteca è dopo la mensa", "L'aula del signor Smith è al piano di sopra vicino alle scale".

Uno studente che usa il **Q-learning semplice** impara solo da ciò che fa *oggi*. Se oggi è andata solo dalla classe alla mensa, aggiorna la sua memoria solo su quell'unico percorso.

Uno studente **Dyna-Q** è diverso. Dopo ogni camminata reale, si siede per un minuto e **riproduce nella sua testa** diverse camminate passate che ricorda. Ogni riproduzione rafforza la sua mappa mentale. Dopo poche settimane conosce la scuola a menadito — non perché ha camminato di più, ma perché ha **pensato di più a ciò che aveva già visto**.

Questo è esattamente ciò che Dyna-Q fa per un agente di RL: impara dall'esperienza reale **e** dall'esperienza immaginata tratta da un modello che costruisce lungo il percorso.

---

## I Tre Ingredienti

Dyna-Q è "Q-learning + modello + pianificazione". Un singolo passo reale svolge **tre** compiti:

1. **RL Diretto** — il solito aggiornamento Q-learning da `(s, a, r, s')`.
2. **Apprendimento del modello** — annota: "Quando ho fatto *a* in *s*, ho ottenuto *r* e sono finito in *s'*."
3. **Pianificazione** — sceglie *n* coppie casuali `(s, a)` dalla memoria del modello ed esegue altri *n* aggiornamenti Q-learning, **fingendo** che quei passi siano appena avvenuti.

Quel terzo passo è la magia. Con `n = 50`, ogni passo reale nel mondo causa **51 aggiornamenti** alla tabella Q. L'agente impara circa 50 volte più velocemente — in termini di passi reali — rispetto a un puro Q-learner.

---

## Un'Immagine del Ciclo

```
                   ┌────────────────────────────────────┐
                   │                                    │
   mondo reale ──► compi azione a ──► osserva (r, s')    │
                            │                           │
              ┌─────────────┼──────────────┐            │
              ▼             ▼              ▼            │
        aggiornamento   Modello[s,a] ← (r,s') Pianificazione ─┘
         Q-learning                       (n aggiornamenti immaginati)
```

Il modello è solo una tabella di consultazione:
`(stato, azione) → (ricompensa, stato_successivo)`. Economico da costruire, economico da interrogare.

---

## Esempi di Vita Reale

- **Studio degli scacchi.** I grandi maestri passano ore a riprodurre le proprie partite e le partite dei maestri nelle loro teste. Ogni riproduzione è "pianificazione" — apprendimento extra da esperienze che sono già avvenute.
- **Un musicista che prova le scale.** Dopo aver suonato una battuta difficile una volta, la prova mentalmente altre dieci volte prima di andare avanti. Le dita non si muovono, ma il cervello si sta aggiornando.
- **Un'auto a guida autonoma.** Mentre è ferma al semaforo rosso, riproduce gli ultimi cento cambi di corsia in simulazione per affinare la sua politica senza consumare gli pneumatici.

---

## Cosa Fa il Nostro Codice

Usiamo il classico **Dyna Maze** ([Sutton & Barto, Figura 8.2](http://incompleteideas.net/book/the-book.html)): una griglia 6×9 con alcuni muri, un punto di inizio `S` nella parte centrale sinistra e un obiettivo `G` in alto a destra.

Eseguiamo tre varianti, ciascuna mediata su 30 seed casuali:

| Impostazione | Passi di pianificazione per passo reale | Significato |
|--------------|-----------------------------------------|-------------|
| `n = 0`      | 0                                       | Q-learning semplice |
| `n = 5`      | 5                                       | un po' di pratica immaginata |
| `n = 50`     | 50                                      | molta pratica immaginata |

Lo script riporta il **numero medio di passi reali per episodio** man mano che l'addestramento procede. Meno passi significano che l'agente ha imparato un percorso più diretto verso l'obiettivo.

### Cosa dovresti vedere quando lo esegui

Il percorso più breve su questo labirinto è di circa 9 passi; con l'esplorazione ε-greedy, un agente ben addestrato ha una media di circa 10 passi per episodio. Esegui per 50 episodi e tutte e tre le impostazioni convergeranno lì — la differenza è *quanto velocemente*:

| Impostazione | Passi per episodio (ultimi 10 ep) | Cosa significa |
|--------------|----------------------------------:|---------------|
| `n = 0`      | ~10                               | Convergenza raggiunta — ma ci sono voluti ~30–50 episodi di esplorazione |
| `n = 5`      | ~10                               | Convergenza entro ~10 episodi |
| `n = 50`     | ~10                               | Convergenza entro ~3–5 episodi |

Il segnale interessante è la *curva di apprendimento*, non il numero finale. Il grafico salvato in `outputs/dyna_q.png` mostra tre curve che scendono verso il basso a velocità molto diverse: `n = 50` lo raggiunge in una manciata di episodi, mentre `n = 0` sta ancora scendendo molto più avanti nell'esecuzione. (In un labirinto deterministico così piccolo, il Q-learning semplice alla fine ci arriva — Dyna-Q ha solo bisogno di molti meno episodi reali, che è l'intero scopo in ambienti dove i passi reali sono costosi).

---

## Perché Funziona Così Bene su Questo Labirinto

Due ragioni:

1. **L'ambiente è deterministico.** Ogni `(s, a)` dà sempre lo stesso `(r, s')`, quindi il modello è esatto dopo una singola visita. L'esperienza immaginata è buona quanto l'esperienza reale.
2. **I passi reali sono costosi, quelli immaginati sono gratuiti.** Ogni aggiornamento immaginato è solo una rapida consultazione di tabelle, mentre un passo reale richiede che l'agente si muova. Quando le interazioni reali sono costose (pensa a un vero robot o a un gioco vero), Dyna-Q è enormemente efficiente in termini di campioni.

---

## Dove Dyna-Q Fatica

- **Ambienti stocastici.** Se `(s, a)` può portare a molti valori diversi di `s'`, un modello che "ricorda l'ultimo risultato" ti darà informazioni errate. Soluzione: memorizzare i conteggi delle visite o addestrare un modello probabilistico.
- **Ambienti non stazionari.** Se il mondo cambia — ad esempio, una porta che era aperta improvvisamente si chiude, o appare una scorciatoia — il modello diventa obsoleto e fornisce previsioni errate. **Dyna-Q+** risolve questo problema aggiungendo un *bonus di esplorazione*: gli stati che non sono stati rivisitati per molto tempo ricevono una piccola ricompensa extra, spingendo l'agente a tornare e controllare se il mondo è cambiato.
- **Grandi spazi di stato.** Un dizionario con chiave `(s, a)` non è scalabile a milioni di stati o a stati continui. Questa è esattamente la lacuna che colmano i **modelli del mondo appresi (reti neurali)** — vedi il file `world_model.py` per i dettagli.

---

## Parole Chiave da Ricordare

| Parola | Significato |
|--------|-------------|
| **Modello** | Memoria di `(stato, azione) → (ricompensa, stato_successivo)` |
| **Passo di pianificazione** | Eseguire un aggiornamento Q utilizzando dati immaginati dal modello |
| **RL Diretto** | Un aggiornamento Q utilizzando dati reali |
| **Efficienza dei campioni** | Misura l'efficacia con cui un modello o un algoritmo di IA utilizza i dati di addestramento per raggiungere un livello specifico di prestazioni |
| **Dyna** | L'architettura di Sutton che alterna apprendimento e pianificazione |

---

## Riassunto in Una Sola Frase

> **Dyna-Q impara dal fare E dall'immaginare — e immaginare è gratuito.**

Questa idea, nella sua moderna forma neurale, alimenta alcuni dei più forti agenti di RL mai costruiti (MuZero, Dreamer, World Models).
