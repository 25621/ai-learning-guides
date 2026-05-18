# SARSA vs Q-Learning: Percorsi Sicuri vs Ottimali 🐢 vs 🐇

## Cos'è?

Due robot devono entrambi camminare lungo il **bordo di un precipizio** per raggiungere l'obiettivo. Entrambi i robot
stanno ancora *imparando* e a volte compiono mosse casuali (oops!).

- 🐢 **Robot SARSA**: "So che a volte ondeggio... quindi camminerò lontano dal precipizio
  per essere al sicuro, anche se ci vorrà più tempo."
- 🐇 **Robot Q-Learning**: "Il percorso più breve costeggia il precipizio — andiamo! (Cade
  a volte durante l'apprendimento, ma alla fine impara la strada migliore.)"

Entrambi i robot sono intelligenti, ma fanno un **compromesso diverso**: sicuro-ma-più-lento vs
ottimale-ma-rischioso-durante-l'apprendimento.

---

## La Differenza Chiave: Quale "Azione Successiva" Usi?

Quando aggiornano i punteggi dopo ogni passo, entrambi gli algoritmi chiedono:
> "Qual è il valore dello *stato successivo*?"

| Algoritmo | Usa l'azione successiva... | On-policy? |
|-----------|----------------------------|------------|
| **SARSA** | ...che *compirò effettivamente* (forse casuale!) | Sì |
| **Q-Learning** | ...che è *teoricamente la migliore* (sempre greedy) | No |

**Esempio di vita reale:** Due bambini che imparano ad andare in bicicletta.

- **Bambino SARSA**: Rimane vicino all'erba perché *sa* che a volte ondeggia casualmente.
  Sta pianificando per il suo vero sé "ondeggiante".
- **Bambino Q-Learning**: Si esercita nel mezzo del sentiero perché sta immaginando un sé futuro
  perfetto che non ondeggia mai. Cade a volte ora, ma impara il percorso migliore più velocemente.

Entrambi i bambini alla fine imparano — ma durante l'addestramento, il bambino SARSA cade meno!

---

## Cosa ha scoperto il nostro codice

Entrambi gli algoritmi sono stati eseguiti per **500 episodi** su Cliff Walking con ε=0.1 (ε = epsilon, "EP-si-lon"; qui significa una probabilità del 10% di compiere una mossa casuale):

| Metrica | SARSA | Q-Learning |
|---------|-------|------------|
| Ricompensa media durante l'addestramento (ultimi 50 ep) | **-19.7** | **-51.0** |
| Valutazione greedy (senza esplorazione) | -17 | **-13** |

- **Durante l'addestramento**: SARSA ottiene **ricompense molto migliori** perché evita il precipizio
  (tenendo conto delle proprie mosse casuali)
- **Dopo l'addestramento** (puro greedy): Q-Learning trova il **percorso ottimale più breve** (-13)!

Man mano che ε tende a 0, entrambi gli algoritmi convergono verso la stessa politica ottimale.

---

## Riepilogo Visivo

```
Percorso SARSA (durante l'addestramento):    Percorso Q-Learning (greedy, dopo l'addestramento):
[ ][→][→][→][→][→][→][→][→][→][→][↓]          [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[↑][→][→][→][→][→][→][→][→][→][→][↓]          [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]          [ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][↓]
[S][C][C][C][C][C][C][C][C][C][C][G]          [S][→][→][→][→][→][→][→][→][→][→][G]
    (deviazione sicura, righe superiori)             (ottimale, costeggia il precipizio)
```

---

## Esempi di vita reale

- **Chirurgo alle prime armi vs chirurgo esperto**: Il chirurgo alle prime armi (SARSA) si tiene lontano dalle
  tecniche rischiose mentre impara. Il chirurgo esperto (Q-Learning greedy) usa la tecnica più efficiente
  dopo averla padroneggiata.
- **Guida in città vs percorso autostradale**: Una pianificazione stile SARSA sceglie strade residenziali più sicure;
  il Q-Learning trova l'autostrada ottimale ma stretta.
- **Studente che studia**: Lo studente-SARSA si attiene ad argomenti ben compresi durante la pratica.
  Lo studente-Q-Learning si spinge verso i problemi più difficili (fallisce di più) ma impara la strategia ottimale.

---

## Parole Chiave da Ricordare

- **On-policy** (SARSA): Impara da ciò che *fai realmente*, inclusa l'esplorazione casuale
- **Off-policy** (Q-Learning): Impara dal *miglior comportamento possibile* separatamente da
  ciò che fai realmente
- **Percorso sicuro**: Percorso più lungo che evita il pericolo, usato quando l'esplorazione causa incidenti
- **Percorso ottimale**: Percorso più breve/con la ricompensa più alta, trovato quando non avviene esplorazione
- **Compromesso Esplorazione-Sfruttamento (Exploration-exploitation tradeoff)**: L'equilibrio tra provare cose nuove e usare
  ciò che si conosce

La grande idea: **SARSA è più sicuro durante l'addestramento (on-policy), il Q-Learning trova il percorso
ottimale più velocemente (off-policy). Quale sia il migliore dipende dal fatto che cadere dal precipizio sia importante o meno!**
