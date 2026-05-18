# Q-Learning Lineare per CartPole 🎪

## Cos'è CartPole?

Immagina un manico di scopa in equilibrio verticale sul tuo dito. Se sposti il dito a sinistra o a destra anche solo di poco, puoi evitare che il manico cada. Questo è **CartPole**!

Un piccolo robot è seduto su un carrello (una scatola su ruote) e ha un'asta fissata sopra. Il robot può solo spingere il carrello a **sinistra** o a **destra**. Deve imparare a mantenere quell'asta in equilibrio il più a lungo possibile — proprio come faresti tu con un manico di scopa!

Il robot può vedere 4 cose del mondo:
1. Dove si trova il carrello
2. Quanto velocemente si muove il carrello
3. Quanto è inclinata l'asta
4. Quanto velocemente si sta inclinando l'asta

---

## Il Grande Problema: Troppi Stati!

Ricordi il Q-learning della Fase 2? Usava una grande tabella per ricordare quanto fosse buona ogni azione in ogni situazione (stato). Funzionava benissimo per Frozen Lake — c'erano solo 16 quadrati sul ghiaccio.

But CartPole è diverso! Il carrello può trovarsi in **qualsiasi posizione**, muovendosi a **qualsiasi velocità**, con l'asta a **qualsiasi inclinazione**. Ci sono fondamentalmente **infiniti stati possibili**! Non possiamo creare una tabella con infinite righe. Avremmo bisogno di un quaderno grande quanto l'universo!

**Esempio di vita reale:** Immagina di imparare ad andare in bicicletta. Non puoi memorizzare ogni possibile oscillazione — ce ne sono troppe! Invece, impari una **regola**: "quando pendo a sinistra, spingo a destra; quando pendo a destra, spingo a sinistra". Una semplice regola funziona per TUTTE le oscillazioni.

---

## La Soluzione: Una Formula Magica

L'**approssimazione di funzione lineare** sostituisce la tabella gigante con una **piccola formula**:

> **Punteggio(situazione, azione) = w₁ × posizione_carrello + w₂ × velocità_carrello + w₃ × angolo_asta + w₄ × velocità_asta**

- I numeri `w` sono chiamati **pesi** (weights) — sono come manopole che puoi ruotare
- Impariamo **pesi diversi per ogni azione** (spingi a sinistra e spingi a destra)
- La formula fornisce un punteggio per quanto sia buona ogni azione in questo momento

**Esempio di vita reale:** Pensa a una semplice ricetta: "1 tazza di farina + 2 uova + ½ tazza di burro". I pesi (1, 2, ½) ti dicono quanto conta ogni ingrediente. Stiamo imparando la ricetta per prendere buone decisioni!

---

## Come impara?

Il robot prova le cose, riceve feedback e ritocca i pesi:

1. **Il robot spinge il carrello** (sceglie l'azione con il punteggio più alto)
2. **Interviene la fisica** (l'asta si inclina un po', il carrello si muove)
3. **Il robot ottiene una ricompensa** (+1 per ogni passo in cui l'asta rimane su, 0 se cade)
4. **Il robot chiede:** "Il risultato effettivo è stato migliore o peggiore di quanto previsto?"
5. **Il robot ritocca i pesi** per essere più vicino alla realtà la prossima volta

Questo è l'**Aggiornamento TD Semi-Gradiente** — un nome complicato per dire "dai una piccola spinta alla ricetta in base alla sorpresa".

> **Nuovo peso = Vecchio peso + Tasso di apprendimento × (Cosa è successo davvero − Cosa ho previsto) × Caratteristica**

---

## Cosa ha scoperto il nostro codice

Quando esegui `linear_q_cartpole.py`, il robot:

- Inizia malissimo (l'asta cade in 10–30 passi)
- Impara gradualmente i pesi corretti in 3.000 tentativi
- Alla fine mantiene l'asta in equilibrio per 100–400+ passi!

Il grafico mostra la **curva di apprendimento** — come il punteggio migliora nel tempo. Sarà irregolare (l'apprendimento non è mai lineare!), ma la tendenza dovrebbe andare verso l'alto.

---

## Perché è fantastico (e limitato!)

**Fantastico:** Una piccola formula con soli 8 numeri (4 pesi × 2 azioni) può bilanciare un'asta! Non è necessaria alcuna tabella gigante.

**Limitato:** La formula è troppo semplice per compiti complessi. Presuppone che numeri più grandi significhino sempre effetti più grandi (il che non è sempre vero). Per giochi più difficili come Atari, abbiamo bisogno di **reti neurali** — che è ciò che fa DQN!

---

## Vocabolario Chiave

| Parola | Significato |
|--------|---------|
| **Caratteristica (Feature)** | Una cosa misurabile del mondo (es. angolo dell'asta) |
| **Peso (Weight)** | Quanto una caratteristica influenza la decisione |
| **Lineare** | La formula è solo moltiplicazione e addizione (niente curve complicate) |
| **Semi-gradiente** | Aggiornare i pesi seguendo la direzione del minor errore |
| **Approssimazione di funzione** | Usare una formula invece di una tabella |

---

## Cosa viene dopo?

L'approssimazione lineare è come usare un righello dritto per disegnare una curva — funziona bene per forme semplici ma non per quelle conplesse. Per i giochi Atari con milioni di possibili situazioni, abbiamo bisogno di **Deep Q-Networks (DQN)** — reti neurali che possono apprendere modelli molto più complessi. Questo è nel prossimo file!
