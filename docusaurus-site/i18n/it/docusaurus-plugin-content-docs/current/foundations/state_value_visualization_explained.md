# Funzioni Stato-Valore 🗺️

## Cos'è uno "Stato"?

Pensa a quando giochi a un gioco da tavolo. In ogni momento, ti trovi su *una* casella del tabellone. Quella casella è il tuo **stato** — è il luogo in cui ti trovi in questo preciso istante.

Nel nostro gioco a griglia 4×4, ci sono 16 caselle (stati). Ogni casella è un posto in cui l'agente può trovarsi.

---

## Cos'è un "Valore"?

Ora ecco la domanda magica: **"Se mi trovo su questa casella in questo momento, quanti tesori posso aspettarmi di raccogliere prima che il gioco finisca?"**

La risposta è il **valore** di quello stato!

Una casella con un **valore alto** significa: "Questo è un ottimo posto — probabilmente raccoglierò molti tesori partendo da qui!"

Una casella con un **valore basso** significa: "Ohi ohi — da qui le cose di solito vanno male."

**Esempio di vita reale:** Immagina di giocare a nascondino. Se ti nascondi dietro un grande albero (un ottimo posto), la tua possibilità di vincere è alta — quello è uno stato ad alto valore! Se ti nascondi nel mezzo di una stanza vuota, probabilmente verrai trovato — quello è uno stato a basso valore.

---

## Il Nostro Grid World

Ecco il tabellone di gioco che abbiamo usato:

```
S  .  .  .      S = Start (Inizio)
.  H  .  H      H = Hole (Buco, ricompensa -1, il gioco finisce)
.  .  .  H      G = Goal (Obiettivo, ricompensa +1, il gioco finisce)
H  .  .  G      . = Casella vuota sicura
```

- Se raggiungi **G** (Goal): ottieni **+1 punto** 🎉
- Se calpesti **H** (Hole): ottieni **-1 punto** 😢
- Altri passi: **0 punti**

Abbiamo usato γ (gamma) = 0,99, il che significa che le ricompense future contano quasi quanto le ricompense immediate. (Una caramella domani è quasi altrettanto buona di una caramella oggi!)

---

## Due Piani Diversi (Politiche)

Abbiamo testato due politiche e calcolato il valore di ogni casella per ciascuna:

### Politica 1: Casuale Uniforme (Uniform Random)
Sceglie casualmente su, giù, sinistra o destra con la stessa probabilità.

```
Valori (Politica Casuale Uniforme):
-0.912  -0.932  -0.912  -0.942
-0.929   (H)   -0.898   (H)
-0.901  -0.801  -0.696   (H)
 (H)   -0.630  -0.104   (G)
```

Quasi ovunque è **negativo** — la politica casuale cade nei buchi così spesso che trovarsi in qualsiasi punto è piuttosto brutto!

---

### Politica 2: Orientata verso l'Obiettivo (Biased Toward Goal)
Preferisce muoversi verso destra e verso il basso (verso l'obiettivo), ma a volte va ancora in altre direzioni.

```
Valori (Politica Orientata verso l'Obiettivo):
-0.838  -0.895  -0.814  -0.961
-0.798   (H)   -0.665   (H)
-0.595  -0.143  -0.213   (H)
 (H)    0.254   0.673   (G)
```

Ora le caselle vicino all'**Obiettivo** hanno **valori positivi** (0,254 e 0,673)! La politica intelligente rende quelle caselle dei buoni posti in cui trovarsi.

---

## Cosa significano i Colori nella nostra Immagine

Nella nostra visualizzazione:
- **Caselle verdi** = valore alto (ottimi posti in cui trovarsi)
- **Caselle rosse** = valore basso (evita questi!)
- **Caselle gialle** = una via di mezzo

Puoi vedere il **gradiente** — i valori diventano più verdi man mano che ti sposti verso l'Obiettivo e più rossi vicino ai Buchi.

---

## Perché ci interessano i Valori?

I valori sono le *fondamenta* dell'apprendimento per rinforzo! Una volta che conosci il valore di ogni stato, puoi prendere decisioni intelligenti:

> "Sono alla casella A. Posso andare alla casella B (valore = 0,5) o alla casella C (valore = -0,3). Andrò in B — ha un valore più alto!"

Questo è esattamente il modo in cui molti algoritmi di RL (come il Q-learning) imparano a prendere buone decisioni senza che vengano spiegate loro le regole.

**Esempio di vita reale:** Immagina di scegliere in quale fila metterti al supermercato. Ogni fila è uno "stato". Il valore di quello stato è quanto velocemente supererai la cassa. Guardi le file (osservi gli stati) e scegli quella con il valore più alto (attesa più breve + meno articoli).

---

## Come abbiamo calcolato i Valori

Abbiamo usato la **Valutazione Iterativa della Politica** (Iterative Policy Evaluation), che funziona così:

1. Inizio: ipotizza che tutti i valori siano 0.
2. Aggiornamento: per ogni casella, calcola quale *dovrebbe* essere il valore in base a dove ti porterà la politica successivamente.
3. Ripeti finché i valori smettono di cambiare (convergenza).

Matematicamente: **V(s) = Σ_a π(a|s) × [R(s,a) + γ × V(next_state)]**

In parole semplici: "Il valore di questa casella = la ricompensa media che otterrò subito + una piccola parte del valore di ovunque andrò a finire."

---

## Parole Chiave da Ricordare

- **Stato (State)**: Dove ti trovi in questo momento (una casella sul tabellone).
- **Valore V(s)**: Ricompensa totale attesa partendo dallo stato s.
- **Politica (Policy)**: Il tuo piano su cosa fare in ogni stato.
- **Fattore di sconto γ**: Quanto ti interessano le ricompense future (0,99 = molto!).
- **Valutazione della Politica (Policy Evaluation)**: Calcolare i valori per ogni stato sotto una determinata politica.

L'idea principale: **Alcuni posti sono migliori di altri — e la funzione valore ti dice esattamente quanto è buono ogni posto!**
