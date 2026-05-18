# SARSA per Cliff Walking 🏔️

## Cos'è?

Immagina un **corridoio molto lungo** con un **terribile precipizio** lungo un bordo. Se cadi nel
precipizio, devi tornare all'inizio! Il tuo obiettivo è camminare da un'estremità all'altra
il più velocemente possibile, senza cadere.

**SARSA** è un robot che impara a percorrere questo corridoio facendo pratica. Impara a intraprendere un
*percorso sicuro* che evita il precipizio — anche se è un po' più lungo — perché sa che potrebbe
accidentalmente scivolare vicino al bordo mentre esplora!

---

## La Grande Idea: Imparare da Ciò che Fai Veramente

SARSA sta per: **S**tato (State) → **A**zione (Action) → **R**icompensa (Reward) → **S**tato (State) → **A**zione (Action)

Queste sono le cinque informazioni che SARSA usa per imparare:

1. **S** — Dove mi trovo in questo momento? (stato attuale)
2. **A** — Quale azione ho effettivamente compiuto?
3. **R** — Quale ricompensa ho ottenuto?
4. **S** — Dove sono finito?
5. **A** — Quale azione *compirò effettivamente dopo*?

L'ultima "A" è ciò che rende SARSA speciale! Si aggiorna usando l'azione che *compirà effettivamente
dopo* (anche se si tratta di una mossa esplorativa casuale), non l'azione ideale perfetta.

**Esempio di vita reale:** Pensa a quando impari ad andare in bicicletta. Se sai che a volte ondeggi
casualmente (esplorazione), rimani un po' più lontano dalle auto parcheggiate — perché sai che il tuo
sé "ondeggiante" potrebbe sterzare! SARSA fa questo: impara un percorso sicuro perché tiene conto dei
propri errori casuali.

---

## La Mappa del Cliff Walking

```
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
[S][C][C][C][C][C][C][C][C][C][C][G]
   ← ← ← ← PRECIPIZIO ← ← ← ← ←
```

- **S** = Inizio (Start) (in basso a sinistra)
- **G** = Obiettivo (Goal) (in basso a destra)
- **C** = Precipizio (Cliff) — calpestare qui = -100 di ricompensa, ricomincia!
- Ogni altro passo = -1 di ricompensa

---

## Cosa ha scoperto il nostro codice

Dopo aver addestrato SARSA per 500 episodi:

| Risultato | Valore |
|-----------|--------|
| Ricompensa media ultimi 50 episodi | **-21.6** |
| Ricompensa percorso ottimale (rischioso) | -13 |

La politica appresa da SARSA passa **lungo la parte superiore della griglia** — una deviazione sicura! Costa alcuni
passi extra (-21 invece di -13), ma quasi non cade mai dal precipizio durante l'addestramento.

---

## Esempi di vita reale

- **Infermiera che somministra medicinali**: Adotta il protocollo sicuro collaudato (percorso sicuro) anche se esiste un
  metodo leggermente più veloce, perché piccoli errori (esplorazione) potrebbero essere pericolosi.
- **Piloti di linea**: Seguono liste di controllo rigorose (percorsi sicuri) anche quando le scorciatoie potrebbero sembrare
  più veloci, tenendo conto dell'errore umano.
- **Imparare a cucinare**: Iniziare con ricette ben collaudate (sicure), non con scorciatoie rischiose.

---

## Parole Chiave da Ricordare

- **On-policy**: Impara riguardo alla politica che sta effettivamente usando (inclusi i suoi errori casuali)
- **Aggiornamento SARSA**: Usa l'azione successiva *effettiva*, non quella teoricamente migliore
- **Percorso sicuro**: Un percorso più lungo che evita il pericolo, tenendo conto degli errori di esplorazione
- **Controllo TD (Temporal Difference)**: Aggiornamento dei valori dopo ogni singolo passo (senza aspettare l'intero episodio)

La grande idea: **SARSA è onesto — impara da ciò che fa realmente, non da ciò che vorrebbe
fare. Questo lo rende cauto e sicuro vicino al pericolo!**
