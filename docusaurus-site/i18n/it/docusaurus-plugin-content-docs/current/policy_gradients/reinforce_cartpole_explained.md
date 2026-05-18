# REINFORCE: Insegnare a un Robot a Fare Scelte Migliori

## Cosa Stiamo Cercando di Fare?

Immagina di avere un robot che gioca a un videogioco. Ogni secondo, il robot deve scegliere:
**"Dovrei premere il pulsante o no?"**

Invece di memorizzare ogni situazione in una tabella (come nel Q-learning), vogliamo che il robot impari una **ricetta** — un insieme di regole che dice direttamente: "In questa situazione, compi questa azione."

Questa ricetta è chiamata **politica** (π, pi). Nell'apprendimento per rinforzo, π significa "la regola per scegliere le azioni."

---

## Il Vecchio Modo vs. Il Nuovo Modo {#the-old-way-vs-the-new-way}

**Vecchio modo (Q-learning / DQN):** Impara quanto è BUONA ogni azione (valori Q), quindi scegli la migliore.
> "Spingere a SINISTRA ha un punteggio di 7, spingere a DESTRA ha un punteggio di 5 → spingi a SINISTRA!"

**Nuovo modo (Gradiente della Politica):** Impara direttamente quale azione SCEGLIERE.
> "Quando l'asta si inclina a destra, spingi a DESTRA con l'80% di probabilità, spingi a SINISTRA con il 20% di probabilità."
*(La parola **Gradiente** si riferisce al "passo" matematico che facciamo per regolare lentamente queste probabilità nella giusta direzione).*

**Esempio di vita reale:** Imparare ad andare in bicicletta.
- Il vecchio modo: calcolare il *punteggio esatto* per "inclinati a sinistra di 5 gradi" rispetto a "inclinati a sinistra di 7 gradi."
- Il nuovo modo: fare pratica finché il tuo **corpo** non impara — spingi il piede che ti sembra giusto!

---

## Come Funziona REINFORCE?

REINFORCE guarda il robot giocare una partita completa dall'inizio alla fine (un **episodio**), quindi chiede: "Quali azioni hanno portato a un buon punteggio? Facciamone di più!"

### Passo dopo Passo

**1. Gioca un episodio**

Il robot compie delle scelte e raccoglie esperienza:
```
Passo 1: Stato = [asta inclinata a destra] → Azione = spingi a DESTRA → Ricompensa = +1
Passo 2: Stato = [asta quasi in equilibrio] → Azione = spingi a DESTRA → Ricompensa = +1
Passo 3: Stato = [asta inclinata a sinistra] → Azione = spingi a SINISTRA → Ricompensa = +1
...
Passo 47: Stato = [l'asta è caduta!] → Episodio finito
```

**2. Calcola i ritorni**

Per ogni passo, calcola G_t — il **ritorno totale da quel momento in poi**:
```
G_al_passo_47 = 1
G_al_passo_46 = 1 + 0.99 × 1 = 1.99
G_al_passo_45 = 1 + 0.99 × 1.99 = 2.97
...
G_al_passo_1  = 47 (all'incirca — ritorno più alto perché calcolato dall'inizio)
```

Il **fattore di sconto** γ = 0.99 significa che le ricompense lontane nel futuro contano un po' meno.

**Esempio di vita reale:** Ricevere una stella d'oro il primo giorno di scuola è più eccitante che sapere che *potresti* riceverne una al centesimo giorno. Le ricompense future sono "scontate" leggermente.

**3. Aggiorna la politica**

Per ogni azione compiuta:
> Se G_t era ALTO (quell'azione ha portato a un ottimo risultato): **falla più spesso!**
> Se G_t era BASSO (quell'azione ha portato a un cattivo risultato): **falla meno spesso!**

La matematica: `perdita = -log_prob(azione) × G_t`

Calcolare il gradiente e aggiornare la politica è come dire al robot:
*"Quell'azione che hai fatto al passo 20? Dovresti farla il 5% più spesso la prossima volta!"*

---

## Cos'è una Rete della Politica?

Invece di una tabella, usiamo una **rete neurale** per rappresentare la politica.

```
Osservazione         Rete della Politica      Probabilità dell'Azione
[pos carrello]  →   [128 neuroni]  →  →  [spingi a SINISTRA: 30%]
[vel carrello]  →   [128 neuroni]        [spingi a DESTRA: 70%]
[angolo asta]   →
[vel asta]      →
```

La rete emette **probabilità** per ogni azione. Quindi campioniamo:
> Lancia un dado → 1-30: spingi a SINISTRA, 31-100: spingi a DESTRA

**Esempio di vita reale:** Un'app meteo dice "70% di probabilità di pioggia." Non SAI che pioverà — decidi in base alla probabilità. Il robot fa la stessa cosa!

---

## Normalizzazione: Perché Sottraiamo la Media

Prima di usare G_t per l'aggiornamento, normalizziamo:
```
G_normalizzato = (G - media(G)) / dev_std(G)
```

**Perché?** Immagina che tutte le ricompense siano positive (come in CartPole — sempre +1 per passo). Senza normalizzazione, OGNI azione sembrerebbe "buona" e il segnale di aggiornamento sarebbe confuso.

Dopo la normalizzazione, alcuni ritorni sono positivi (sopra la media → spingi di più) e altri sono negativi (sotto la media → spingi di meno). Il segnale diventa molto più pulito!

**Esempio di vita reale:** Il tuo insegnante valuta in base a una curva. Se il punteggio medio è 70 e tu hai preso 85, è ottimo! Ma se la media è 90 e tu hai preso 85, sei sotto la media. Il punteggio grezzo da solo non racconta tutta la storia.

---

## Il Problema: Alta Varianza

REINFORCE ha una grande debolezza: la **varianza**. I ritorni G_t sono molto rumorosi.

**Esempio di vita reale:** Immagina di giudicare uno chef assaggiando solo UN pasto da ogni ristorante. A volte lo chef ha avuto una brutta giornata, a volte gli ingredienti non erano buoni. Un solo pasto non è sufficiente per sapere con certezza se il ristorante è buono!

REINFORCE aspetta un episodio COMPLETO prima di aggiornare. Un episodio potrebbe essere molto fortunato, un altro molto sfortunato. I gradienti saltano ovunque.

Questo è il motivo per cui la curva di apprendimento (nel grafico) appare frastagliata:
- Alcune esecuzioni arrivano a 500 (fantastico!)
- Alcune esecuzioni scendono a 50 (terribile!)

Nonostante il rumore, REINFORCE alla fine impara — ma ci vuole pazienza.

---

## I Risultati

```
Episodio  100 | Ricompensa media (ultimi 100):  43.1
Episodio  200 | Ricompensa media (ultimi 100): 193.9
Episodio  500 | Ricompensa media (ultimi 100): 408.4
Episodio 1000 | Ricompensa media (ultimi 100): 500.0  ← Risolto!
```

Il robot impara a bilanciare l'asta per il massimo di 500 passi — RISOLTO!

Nonostante i suoi problemi di varianza, REINFORCE su CartPole è efficace perché:
1. Gli episodi sono brevi (quindi ne otteniamo molti per ogni ciclo di addestramento).
2. La politica ottimale è semplice (principalmente spingere nella direzione in cui l'asta si inclina).

---

## Concetti Chiave

| Concetto | Linguaggio Semplice |
|---------|---------------|
| **Politica** | La ricetta del robot per scegliere le azioni |
| **Episodio** | Una partita completa dall'inizio alla fine |
| **Ritorno G_t** | Ricompensa futura totale da questo momento |
| **Sconto γ** | Le ricompense future contano un po' meno di quelle immediate |
| **Normalizzare** | Sottrai la media in modo che il segnale sia più pulito |
| **Varianza** | Quanto le stime del gradiente saltano in giro |

---

## Prossimi Passi

La grande debolezza di REINFORCE è la **varianza**. Nel prossimo script (`reinforce_baseline.py`), aggiungiamo una **baseline** che riduce drasticamente questo rumore — senza cambiare ciò che l'algoritmo impara in media.

L'idea chiave: invece di chiedere "questa azione è stata buona?", chiediamo "questa azione è stata **migliore del previsto**?". Questo piccolo cambiamento rende l'apprendimento molto più stabile.
