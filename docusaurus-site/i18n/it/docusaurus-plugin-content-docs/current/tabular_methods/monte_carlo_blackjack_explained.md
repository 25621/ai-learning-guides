# Controllo Monte Carlo per il Blackjack 🃏

## Cos'è?

Hai mai giocato a un gioco di carte dove devi decidere: **"Prendo un'altra carta o mi accontento di quello che ho?"**

Il **Blackjack** (chiamato anche "21") è esattamente questo! Vuoi che le tue carte sommino un valore il più vicino possibile a 21, senza superarlo. Se superi 21, "sballi" (bust) e perdi!

Il **controllo Monte Carlo** è il modo in cui un robot impara a giocare a Blackjack — giocando *migliaia di partite complete* e ricordando cosa ha funzionato e cosa no.

---

## L'Idea Principale: Imparare da Storie Complete

La parola "Monte Carlo" deriva dal famoso casinò di Monaco. In matematica, significa: **usare esperimenti casuali per imparare qualcosa**.

Ecco come funziona:

1. **Gioca una partita completa** (un episodio completo) usando qualsiasi strategia tu abbia.
2. **Guarda cosa è successo**: Hai vinto? Perso? Pareggiato?
3. **Lavora a ritroso**: Chiedere carta (hit) con 17 è stata una buona idea? E con 14?
4. **Aggiorna la tua memoria**: Ricorda se ogni decisione ha portato alla vittoria o alla sconfitta.

Fallo per **500.000 partite** e diventerai molto bravo!

**Esempio di vita reale:** Immagina di imparare a cucinare preparando 500.000 pasti. Ogni volta, ricordi esattamente cosa hai fatto — e se il pasto era buono. Dopo abbastanza tentativi, saprai: "Aggiungere troppo sale in questo passaggio ha sempre reso il piatto cattivo." Monte Carlo funziona allo stesso modo!

---

## Differenza Chiave da SARSA e Q-Learning

SARSA e Q-Learning aggiornano la loro conoscenza **dopo ogni singolo passo** (anche a metà episodio). Monte Carlo aspetta finché l'**intero episodio è finito**, poi guarda indietro a tutto ciò che è successo.

| Metodo | Aggiorna quando? | Necessita dell'episodio completo? |
|--------|---------------|------------------------|
| **TD (SARSA, Q-Learning)** | Dopo ogni passo | No |
| **Monte Carlo** | Dopo ogni episodio completo | Sì |

Questo rende Monte Carlo più semplice da capire, ma non può imparare finché ogni episodio non termina.

---

## Lo Stato del Blackjack

Il robot guarda 3 cose ad ogni turno:
1. **Il mio totale delle carte** (da 12 a 21)
2. **Quale carta mostra il mazziere?** (dall'Asso al 10)
3. **Ho un asso utilizzabile?** (Un Asso può valere 1 o 11)

Da queste 3 informazioni, decide: **Chiedere carta (Hit) o Stare (Stick)?**

---

## Cosa ha scoperto il nostro codice

Dopo **500.000 partite** di Blackjack:

| Esito | Percentuale |
|---------|------------|
| **Vittorie** | **43,1%** |
| **Pareggi** | 8,9% |
| **Sconfitte** | 48,0% |

Questo valore è vicino alla "strategia di base" matematicamente ottimale (circa il 42-43% di vittorie)! Il robot ha imparato quando chiedere carta e quando stare — semplicemente giocando e ricordando.

La politica appresa mostra:
- **Chiedere carta** (Hit) quando il tuo totale è basso (è improbabile che tu sballi).
- **Stare** (Stick) quando il tuo totale è alto (potresti sballare se prendi un'altra carta).
- Avere un **asso utilizzabile** (usable Ace) ti permette di essere più aggressivo (può passare da 11 a 1 se necessario).

---

## Esempi di Vita Reale

- **Previsioni meteorologiche**: Le simulazioni Monte Carlo eseguono migliaia di scenari "e se" per prevedere il tempo di domani.
- **Modellazione del mercato azionario**: Gli analisti simulano migliaia di futuri possibili per stimare il rischio.
- **Imparare a giocare a scacchi**: Un giocatore rivede intere partite (non solo singole mosse) per capire quale strategia ha portato alla vittoria.

---

## Parole Chiave da Ricordare

- **Episodio (Episode)**: Una partita completa dall'inizio alla fine.
- **Ritorno (G)**: Ricompensa totale raccolta da un punto del gioco fino alla fine.
- **Every-visit MC**: Aggiorna il punteggio per uno stato ogni volta che lo visiti in un episodio.
- **Nessun bootstrapping**: Monte Carlo non usa stime di valori futuri — aspetta il risultato reale!
- **Politica ε-soft** (ε = epsilon): Di solito esegue la migliore azione conosciuta, ma a volte esplora casualmente.

L'idea principale: **Monte Carlo impara giocando molte partite complete. È come imparare dall'esperienza — ricordi tutto quello che è successo e capisci cosa ha portato alla vittoria!**
