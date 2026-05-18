# Iterazione della Politica per GridWorld 🗺️

## Cos'è?

Immagina di giocare a un gioco da tavolo su una **griglia 4×4** (come una minuscola scacchiera). Inizi in un angolo e devi raggiungere l'altro angolo. Ogni passo costa 1 punto (non vuoi sprecare passi!) e raggiungere l'obiettivo non ti fa guadagnare nulla extra: vuoi solo arrivarci il più velocemente possibile.

L'**Iterazione della Politica** (Policy Iteration) è il modo in cui un computer calcola le **mosse migliori per ogni casella** del tabellone - tutto in una volta!

---

## L'Idea Centrale: Due Passaggi, Continuamente

Pensalo come se stessi pulendo la tua stanza con un aiutante:

1. **Passaggio 1 — Capire quanto è buona ogni casella (Valutazione della Politica)**
   Il tuo aiutante gira per ogni casella e scrive: "Se seguo il piano attuale, quanti passi mi ci vorranno per raggiungere l'uscita da qui?". Lo fa più e più volte finché i numeri smettono di cambiare.

2. **Passaggio 2 — Migliorare il piano (Miglioramento della Politica)**
   Ora guardi ogni casella e chiedi: "C'è una direzione migliore che potrei prendere da qui?". Se sì, aggiorna il piano!

Ripeti i passaggi 1 e 2 finché il piano smette di cambiare: quella è la **politica ottimale**!

**Esempio di vita reale:** Immagina di trovare il percorso più veloce per andare a scuola. Prima indovini un percorso e cronometri il tempo (Passaggio 1). Poi guardi ogni angolo di strada e chiedi "c'è una scorciatoia da qui?" (Passaggio 2). Aggiorni il tuo percorso e ripeti finché non riesci a trovare altre scorciatoie!

---

## Cosa ha Trovato il Nostro Codice

Il nostro GridWorld 4×4 ha due stati terminali (angoli) e l'agente paga -1 per passo.
L'iterazione della politica è confluita in soli **4 round** (139 scansioni di valutazione totali):

```
Valori di Stato V(s):    Politica Ottimale:
 0.0  -1.0  -1.9  -2.7    T   ←   ←   ↓
-1.0  -1.9  -2.7  -1.9    ↑   ↑   ↑   ↓
-1.9  -2.7  -1.9  -1.0    ↑   ↑   ↓   ↓
-2.7  -1.9  -1.0   0.0    ↑   →   →   T
```

**I valori hanno perfettamente senso!** Le caselle accanto a un terminale hanno valore -1 (a un passo di distanza). Le caselle a due passi hanno valore -1.9 (= -1 + 0.9 × -1) e così via.

---

## Esempi di Vita Reale

- **Navigazione GPS**: Calcolare la svolta migliore ad *ogni* incrocio sulla mappa.
- **Controllo dell'ascensore**: In quale piano deve andare l'ascensore quando ha più richieste?
- **Robot di fabbrica**: Pianificare il percorso più efficiente intorno a una griglia di magazzino.

---

## Parole Chiave da Ricordare

- **Politica**: Il piano - quale azione intraprendere in ogni stato
- **Funzione Valore V(s)**: Quanto è bene trovarsi nello stato s (più alto = più vicino all'obiettivo)
- **Valutazione della Politica**: Calcolare quanto è buono il piano attuale
- **Miglioramento della Politica**: Migliorare il piano usando la funzione valore
- **Politica Ottimale**: Il miglior piano possibile - non può essere migliorato ulteriormente

L'idea centrale: **Non è necessario provare ogni piano possibile! Basta continuare a migliorare quello attuale e troverai il piano migliore in pochissimi round.**
