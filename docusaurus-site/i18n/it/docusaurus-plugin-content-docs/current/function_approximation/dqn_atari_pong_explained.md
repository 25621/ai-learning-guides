# DQN su Atari Pong 🏓

## Cos'è Atari Pong?

Pong è uno dei videogiochi più vecchi mai realizzati — è come il tennistavolo digitale! Due
racchette fanno rimbalzare una pallina avanti e indietro. Vinci un punto se l'avversario manca la pallina.
Il gioco termina quando qualcuno raggiunge 21 punti.

Nella nostra versione, l'IA controlla una racchetta. L'avversario (computer) controlla l'altra.
Il gioco inizia sempre con un punteggio di −21 (il peggior risultato possibile). Un buon agente raggiunge 0 o +21.

---

## Perché Pong è Difficile per un'IA?

In CartPole, il robot poteva VEDERE direttamente i numeri (angolo dell'asta, velocità del carrello...).
In Pong, tutto ciò che vede sono **pixel grezzi** — migliaia di piccoli punti colorati su uno schermo!

```
Input CartPole: [0.02, −0.14, 0.01, −0.23]   ← 4 numeri, facile!
Input Pong:     [griglia pixel: 210×160×3]    ← 100.800 numeri, MOLTO più difficile!
```

Il robot deve capire dai pixel:
- Dov'è la mia racchetta?
- Dov'è la pallina?
- La pallina si muove a sinistra o a destra?
- Quanto velocemente?

Gli esseri umani lo fanno automaticamente (abbiamo una vista incredibile!). Per un'IA, questa è una sfida enorme.

---

## Vedere il Movimento: Frame Stacking 🎬

Un singolo frame (fermo immagine) non ti dice se la pallina si sta muovendo a sinistra o a destra. Hai bisogno
di vedere PIÙ frame per capire il movimento — proprio come un flip book (foglietto animato) funziona solo quando
sfogli molte pagine.

**Frame Stacking:** Fornisce simultaneamente gli ultimi 4 frame alla rete.

```
Frame 1: pallina in posizione 40
Frame 2: pallina in posizione 43    → Sovrapponi questi 4 frame → La rete vede il MOVIMENTO!
Frame 3: pallina in posizione 46
Frame 4: pallina in posizione 49
```

La rete può ora dedurre: "la pallina si muove verso destra a velocità 3"

**Esempio di vita reale:** Guardare un film rispetto a guardare un singolo fotogramma. Un fotogramma di una corsa
automobilistica è solo un'immagine sfocata. Guardane 4 e potrai dire quale auto è più veloce!

---

## Vedere con una CNN 🔍

Per gli input di pixel, usiamo una rete neurale speciale chiamata **Rete Neurale Convoluzionale
(Convolutional Neural Network - CNN)**. Invece di guardare tutti i pixel contemporaneamente, una CNN usa finestre scorrevoli per rilevare
pattern (modelli) — come occhi che scansionano un'immagine.

```
Pixel grezzi (84×84×4 frame)
       ↓
Strato Conv 1 (filtro 8×8, passo 4) → trova bordi e forme
       ↓
Strato Conv 2 (filtro 4×4, passo 2) → trova oggetti (racchette, pallina)
       ↓
Strato Conv 3 (filtro 3×3, passo 1) → trova relazioni
       ↓
Flatten → 512 neuroni → Valori Q (uno per azione)
```

**Esempio di vita reale:** Quando cerchi un amico in mezzo alla folla, il tuo cervello nota prima
le forme (una persona), poi le caratteristiche (colore dei capelli), infine i dettagli (il viso). Le CNN funzionano allo
stesso modo — dai pattern semplici a quelli complessi!

---

## Preprocessing: Rimpicciolire il Mondo

I frame di Pong sono 210×160 pixel a colori. È troppo grande! Pre-elaboriamo ogni frame:

1. **Scala di grigi (Grayscale)** — il colore non conta per Pong (la pallina è comunque sempre bianca)
2. **Ridimensionamento a 84×84** — più piccolo = addestramento più veloce, ma ancora abbastanza chiaro da vedere
3. **Normalizzazione a [0,1]** — dividiamo i valori dei pixel per 255 in modo che siano numeri piccoli

**Esempio di vita reale:** Como fare una fotocopia al 50%. I dettagli importanti
(pallina, racchette) sono ancora visibili, solo più piccoli. Alla fotocopiatrice non importa nemmeno dei
colori!

---

## Reward Clipping: Trattare Tutti i Giochi Ugualmente ✂️

In Pong, ottieni +1 per un punto segnato, −1 per un punto subito.
In altri giochi Atari, i punteggi possono essere migliaia!

Noi effettuiamo il **clipping delle ricompense** nell'intervallo [−1, +1] in modo che la rete non si curi della scala delle ricompense.
Questo stesso codice può addestrare su QUALSIASI gioco Atari senza dover regolare la scala delle ricompense.

---

## Quanto Tempo Richiede l'Addestramento?

| Durata Addestramento | Cosa Impara l'Agente |
|---|---|
| 100.000 passi | Per lo più casuale, reagisce a malapena |
| 1M passi | Inizia a muoversi verso la pallina a volte |
| 5M passi | Restituisce alcuni colpi |
| 10M passi | Gioco competitivo, potrebbe vincere alcune partite |
| 20M+ passi | Spesso batte l'avversario computer |

La nostra demo esegue **300.000 passi** — sufficienti per vedere che l'architettura di addestramento funziona e
osservare i primi apprendimenti, ma non abbastanza per padroneggiare il gioco.

**Esempio di vita reale:** Imparare a suonare il pianoforte richiede mesi. Una sessione di pratica di 10 minuti
mostra che lo stai facendo bene, ma non aspettarti di esibirti ancora in concerto!

---

## Cosa ha scoperto il nostro codice

Dopo 300.000 passi su Pong:
- L'agente inizia con punteggi intorno a −20 (restituisce a malapena la pallina)
- Alla fine, in genere migliora arrivando intorno a −15 o −10
- La curva di apprendimento mostra un miglioramento graduale rispetto al gioco casuale

Per vedere prestazioni competitive reali su Pong, dovresti eseguire circa 10M+ passi con una GPU.
L'implementazione è completa e corretta — ha solo bisogno di più tempo!

---

## Vocabolario Chiave

| Parola | Significato |
|--------|-------------|
| **CNN** | Rete Neurale Convoluzionale — specializzata per input di immagini |
| **Frame Stacking** | Fornire più frame consecutivi per catturare il movimento |
| **Preprocessing** | Trasformazione dei frame grezzi (scala di grigi, ridimensionamento, normalizzazione) prima di fornirli alla rete |
| **Reward Clipping** | Limitazione delle ricompense a [−1, +1] per funzionare su diversi giochi |
| **ALE** | Arcade Learning Environment — la libreria che esegue i giochi Atari |

---

## Lo Storico Traguardo

Quando DeepMind pubblicò il DQN nel 2015, il mondo rimase sbalordito. Un SINGOLO algoritmo,
con la STESSA architettura, imparò a giocare a 49 diversi giochi Atari — molti a livello
superiore a quello umano — partendo solo dai pixel grezzi e dal punteggio!

Prima del DQN, si pensava che fosse necessario programmare a mano la strategia per ogni gioco.
Il DQN ha mostrato che un sistema di apprendimento generico poteva capire tutto da solo.
È stato un momento storico per l'IA!
