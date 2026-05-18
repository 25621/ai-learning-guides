# REINFORCE con Baseline: Ridurre il Rumore

## Il Problema con il REINFORCE Semplice

Immagina di essere uno studente che cerca di capire se la tua risposta a un test è stata buona.

**Feedback cattivo:** "Hai preso 7 punti!"

7 è buono? Se il massimo è 10, sì! Se tutti gli altri hanno preso 9, no! Senza contesto, non puoi sapere se dovresti cambiare il tuo stile di risposta.

Questo è esattamente il problema di REINFORCE: usa i **ritorni grezzi** (G_t) per valutare le azioni. Un punteggio di ritorno totale di 200 punti potrebbe essere fantastico o terribile a seconda della situazione.

---

## L'Ingresso della Baseline

Una **baseline** b(s) è un punto di riferimento: "Quale ricompensa mi **aspetto** in questa situazione?"

Invece di chiedere "Questa azione è stata buona?", chiediamo:

> **"Questa azione è stata migliore di quello che mi aspetterei normalmente?"**

```
Vecchio segnale: aggiornamento ∝ G_t
Nuovo segnale: aggiornamento ∝ (G_t - b(s_t))
```

**Esempio di vita reale:** Hai preso 85 a un test di matematica.
- Se la media della classe è 60 → la tua risposta è stata **25 punti sopra la media** → ottimo!
- Se la media della classe è 90 → la tua risposta è stata **5 punti sotto la media** → c'è da lavorare!

Il **vantaggio** (G_t - b(s)) è positivo quando hai fatto meglio del previsto e negativo quando hai fatto peggio. Questo è un segnale di apprendimento molto più pulito!

---

## Cos'è la Baseline?

La baseline naturale è la **funzione valore V(s)**:

> V(s) = "Ritorno totale atteso se mi trovo nello stato s e seguo la mia politica attuale"

Impariamo questo valore con una **Rete del Valore (Value Network)** separata (chiamata anche rete baseline o critico):

```
Stato  →  [128 neuroni]  →  [128 neuroni]  →  V(s)   (singolo numero)
```

Per ogni stato visitato dall'agente, V(s) predice il ritorno atteso. Se il ritorno effettivo G_t è superiore a V(s), l'azione è stata migliore del previsto!

---

## Due Reti che Imparano Insieme

```
L'episodio avviene
     ↓
Calcola i ritorni effettivi G_t
     ↓
         ┌─────────────────────────────┐
         │ Vantaggio = G_t - V(s_t)    │
         │  +: l'azione è stata migliore│
         │  -: l'azione è stata peggiore│
         └─────────────────────────────┘
              ↓                  ↓
 Aggiorna la Rete della Politica  Aggiorna la Rete del Valore
    (rendi le azioni buone        (rendi le previsioni più
    più/meno probabili)            accurate la prossima volta)
```

**Esempio di vita reale:** Due amici vanno al ristorante insieme.

- Amico 1 (Rete del Valore): "Prevedo che questo piatto sarà da 7/10"
- Amico 2 (Rete della Politica): Provi il piatto e gli dai 9/10
- Vantaggio = 9 - 7 = +2 → "È stato meglio del previsto! Ordinalo di nuovo!"

Alla visita successiva, l'Amico 1 aggiorna la sua previsione avvicinandola a 9/10. L'Amico 2 avrà più probabilità di ordinare quel piatto la prossima volta.

---

## Perché Questo Riduce la Varianza?

**Intuizione matematica:**

Senza baseline: `gradiente ∝ ∇log π(a|s) × G_t`

I valori di G_t variano molto da episodio a episodio:
```
Episodio 1: G = [45, 44, 43, ..., 1]   (partita media)
Episodio 2: G = [500, 499, ..., 1]      (partita fantastica!)
Episodio 3: G = [12, 11, ..., 1]        (partita terribile)
```

Le stime del gradiente saltano selvaggiamente perché G_t è grande e rumoroso.

Con baseline: `gradiente ∝ ∇log π(a|s) × (G_t - V(s_t))`

Il vantaggio (G_t - V(s_t)) è molto più piccolo e centrato vicino allo zero:
```
Episodio 1: vantaggio ≈ [-2, +1, -3, ..., 0]   (piccolo, centrato)
Episodio 2: vantaggio ≈ [+10, +8, ..., +3]      (questa partita È STATA fantastica)
Episodio 3: vantaggio ≈ [-5, -6, ..., -2]       (questa partita È STATA brutta)
```

**Esempio di vita reale:** Misurare la tua velocità di corsa.
- Senza baseline: "Ho corso a 8 km/h" (senza senso senza contesto)
- Con baseline: "Ho corso 2 km/h PIÙ VELOCE della mia media" (chiaramente buono!)

Il vantaggio è sempre un confronto — è naturalmente più piccolo e più stabile.

---

## Fondamentale: Nessun Bias!

La baseline non cambia COSA l'algoritmo impara — cambia solo quanto VELOCEMENTE e in modo STABILE lo impara.

**Perché?** Perché il vantaggio atteso è sempre 0 in termini di valore atteso:

> E[G_t - V(s_t)] = E[G_t] - V(s_t) = V(s_t) - V(s_t) = 0

Qualsiasi b(s) che non dipenda dall'azione funziona come una baseline valida!

**Esempio di vita reale:** Valutare in base a una curva di distribuzione non cambia chi ha ottenuto i risultati migliori — rende solo i punteggi più facili da interpretare. La classifica rimane la stessa; cambia solo la scala.

---

## I Risultati

```
Senza baseline — Media finale 100 ep: 500.0, var gradiente: 599.3
Con baseline   — Media finale 100 ep: 491.4, var gradiente: 578.8
```

Entrambi i metodi raggiungono prestazioni quasi perfette su CartPole, ma nota:
1. La **varianza del gradiente** è misurabile (il grafico a destra mostra la varianza durante l'addestramento).
2. Con la baseline, l'agente raggiunge prestazioni elevate in modo **più affidabile** — meno crolli verso ricompense basse durante l'addestramento.

La riduzione della varianza è più drammatica in ambienti più difficili (LunarLander, MuJoCo).

---

## Equazioni Chiave

```
Valore baseline:     V(s) ← V(s) + α(G_t - V(s))   [minimizza MSE]
Gradiente politica:  θ ← θ + α ∇log π(a_t|s_t) · (G_t - V(s_t))
Vantaggio:           A_t = G_t - V(s_t)
```

---

## Concetti Chiave

| Concetto | Linguaggio Semplice |
|---------|---------------|
| **Baseline b(s)** | Ricompensa attesa nello stato s — il nostro punto di riferimento |
| **Vantaggio A_t** | "Questa azione è stata migliore del previsto?" |
| **Rete del Valore** | Una rete neurale che impara a predire i ritorni attesi |
| **Riduzione della varianza** | Meno rumore nelle stime del gradiente → apprendimento più stabile |
| **Non polarizzato (Unbiased)** | La baseline non cambia la politica target in media; rende solo il segnale di apprendimento meno rumoroso e più stabile |

---

## Prossimi Passi

La baseline è in realtà l'inizio di qualcosa di molto più potente: i metodi **Actor-Critic**.

Invece di calcolare V(s) solo alla fine di un episodio, l'Actor-Critic aggiorna V(s) ad ogni singolo passo usando l'apprendimento per **Differenza Temporale (Temporal Difference)**. Questo rende gli aggiornamenti molto più veloci e permette all'agente di imparare da episodi incompleti!

Vedi `a2c_lunarlander.py` per l'implementazione completa di Actor-Critic.
