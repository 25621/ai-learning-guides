# Frozen Lake con una Politica Casuale 🧊

## Cos'è Frozen Lake?

Immagina di giocare su un **laghetto ghiacciato** con i tuoi amici.

Il ghiaccio è per lo più sicuro, ma alcuni punti presentano dei **buchi** (hole) — se calpesti un buco, ci cadi dentro e il gioco finisce! Ad un'estremità del laghetto c'è un **regalo** 🎁. Il tuo compito è scivolare dalla **partenza** fino al **regalo** senza cadere dentro i buchi.

Ecco come appare il lago ghiacciato (4 quadrati × 4 quadrati):

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **S** = Start (dove inizi)
- **F** = Frozen ice (ghiaccio ghiacciato, sicuro!)
- **H** = Hole (buco, se ci cadi il gioco finisce 😨)
- **G** = Goal (obiettivo, il regalo! 🎁)

---

## La parte difficile: il ghiaccio scivoloso!

Su un vero laghetto ghiacciato, quando provi a camminare verso *destra*, a volte il ghiaccio ti fa scivolare in *alto* o in *basso*! È questo che rende il gioco difficile.

Anche se *vuoi* andare a destra, il gioco potrebbe farti scivolare altrove. Questo fenomeno è chiamato **stocasticità** — una parola complicata per dire che "le cose non vanno sempre come previsto".

---

## Cos'è una Politica Casuale?

Una **politica** (policy) è semplicemente un piano: "In questa situazione, compirò QUESTA azione."

Una **politica casuale** significa: "Non ho alcun piano! Sceglierò una direzione a caso ogni volta — su, giù, sinistra o destra — come far girare una trottola!"

È come un bambino che cammina sul ghiaccio senza avere idea di dove sia il regalo.

---

## Cosa ha scoperto il nostro codice

Abbiamo testato la politica casuale per **1.000 partite**:

| Risultato | Valore |
|--------|-------|
| **Volte in cui è stato raggiunto il regalo** | 11 su 1.000 (1,1%) |
| **Passi medi per partita** | 7,5 passi |
| **Partita più veloce** | 2 passi |
| **Partita più lunga** | 33 passi |

La maggior parte delle volte, il camminatore casuale è caduto rapidamente in un buco. Solo 1 partita su 100 è finita con il ritrovamento del regalo!

---

## Perché è utile?

Anche se la politica casuale è pessima, ci fornisce una **baseline** — un punto di partenza per i confronti.

Quando in seguito costruiremo una politica *intelligente* (usando Q-learning o altri algoritmi), potremo dire: "Il nostro agente intelligente ha successo nel 75% dei casi — molto meglio dell'1% del camminatore casuale!"

**Esempio di vita reale:** Immagina di cercare la tua aula in una nuova scuola girando casualmente a sinistra o a destra in ogni corridoio. Potresti arrivarci alla fine, ma ci vorrebbe molto tempo! Una politica intelligente è come avere una mappa.

---

## Cosa mostra la Mappa di Calore

Nella nostra immagine, la **mappa di calore** (heatmap) mostra quali quadrati il camminatore casuale ha visitato più spesso:

- Il quadrato di **Partenza** (Start) è visitato molto spesso (ogni partita inizia lì).
- I quadrati vicino ai **buchi** (holes) sono visitati meno (il camminatore spesso cade dentro prima di raggiungerli).
- L'**Obiettivo** (Goal) è visitato molto raramente perché il camminatore casuale quasi mai riesce ad arrivarci.

Questo ci dice qualcosa di importante: la politica casuale rimane bloccata vicino all'inizio e non esplora mai veramente tutto il lago.

---

## Parole Chiave da Ricordare

- **Politica (Policy)**: Il tuo piano su cosa fare in ogni situazione.
- **Politica Casuale**: Nessun piano — scegli un'azione a caso!
- **Baseline**: Un risultato scarso che usiamo per il confronto (quanto meglio possiamo fare?).
- **Stocastico**: Le cose non vanno sempre come previsto (come il ghiaccio scivoloso!).
- **Tasso di successo (Success rate)**: Quanto spesso abbiamo vinto? (Qui: 1,1% — molto basso!)

L'idea principale: **Una politica casuale è un punto di partenza. Il vero apprendimento significa costruire un piano migliore!**
