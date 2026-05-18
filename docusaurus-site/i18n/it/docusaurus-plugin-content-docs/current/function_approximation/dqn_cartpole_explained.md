# Deep Q-Network (DQN) da Zero 🧠

## Il Problema con il Metodo Lineare

Ricordi la nostra formula lineare di prima?

> Punteggio = w₁ × pos_carrello + w₂ × vel_carrello + w₃ × angolo_asta + w₄ × vel_asta

Funziona discretamente per CartPole, ma cosa succede con un videogioco dove vedi migliaia di
pixel? Non puoi scrivere una formula semplice per quello!

Abbiamo bisogno di qualcosa che possa osservare situazioni complicate e capire l'azione migliore.
Quel qualcosa è una **rete neurale**.

---

## Cos'è una Rete Neurale?

Pensa al tuo cervello. Milioni di piccole cellule chiamate neuroni comunicano tra loro. Quando
tocchi qualcosa di caldo, i neuroni inviano segnali: "SCOTTA! → Togli subito la mano!" Ogni neurone
passa l'informazione agli altri e insieme prendono una decisione intelligente.

Una **rete neurale su un computer** funziona allo stesso modo:

```
Strato di Input     Strato Nascosto 1   Strato Nascosto 2   Strato di Output
[pos carrello]  →   [128 neuroni]  →    [128 neuroni]  →    [punteggio sposta a SINISTRA]
[vel carrello]  →   [  ...      ]       [  ...      ]       [punteggio sposta a DESTRA]
[angolo asta]   →
[vel asta]      →
```

Ogni freccia ha un **peso (weight)** (quanto è forte quella connessione). Ci sono migliaia di
questi pesi — e la rete li impara TUTTI!

**Esempio di vita reale:** Uno chef in un ristorante assaggia il cibo e regola centinaia di
ingredienti contemporaneamente. Ogni papilla gustativa è come un neurone, e insieme dicono allo chef
"aggiungi più sale" o "meno pepe". Addestrare la rete è come lo chef che impara nel corso di
migliaia di pasti.

---

## DQN = Deep Q-Network

Il **DQN** (Deep Q-Network) è stato inventato da DeepMind nel 2013. Hanno preso la vecchia formula
del Q-learning e hanno sostituito la Q-table con una rete neurale!

Invece di:
> Q-table[stato][azione] = punteggio

Abbiamo:
> Q-network(stato) → [punteggio_per_sinistra, punteggio_per_destra]

La rete riceve lo stato come input e restituisce i valori Q per TUTTE le azioni contemporaneamente.
Questo è molto più efficiente che calcolarli separatamente!

---

## Questo Script: La Versione "Naive"

Questo script mostra il DQN **senza** alcun trucco speciale. Semplicemente:
1. Osserva lo stato.
2. Chiede alla rete "quanto è buono andare a sinistra? quanto a destra?".
3. Compi l'azione con il punteggio più alto.
4. Riceve una ricompensa, aggiorna la rete.

**Questa versione è intenzionalmente instabile!** Immagina uno studente che dimentica immediatamente
le lezioni precedenti ogni volta che impara qualcosa di nuovo. La rete si aggiorna dopo
ogni singolo passo, il che causa caos.

**Esempio di vita reale:** Immagina di imparare a cucinare cambiando l'intera ricetta dopo
ogni singolo assaggio. Potresti passare da "troppo salato" a "senza sale" a "decisamente troppo salato"
senza mai stabilirti sulla giusta quantità. È quello che succede qui!

---

## Cosa Vedrai

Quando esegui `dqn_cartpole.py`:
- I punteggi potrebbero saltare molto (apprendimento instabile).
- A volte l'agente diventa molto bravo, poi dimentica tutto.
- Il grafico della perdita (loss) mostra oscillazioni selvagge.

**Questo è previsto!** Mostra PERCHÉ abbiamo bisogno di miglioramenti — come l'experience replay e le
reti target. Questi arrivano dopo!

---

## Il Trucco ε-Greedy 🎲

Il robot non sceglie sempre l'azione migliore. A volte sceglie a caso!

Perché? Perché se scegliesse sempre quello che sembra il meglio, potrebbe non scoprire mai opzioni migliori.

> Con probabilità ε (epsilon): scegli un'azione CASUALE (esplora! - explore)
> Con probabilità 1-ε: scegli l'azione migliore conosciuta (sfrutta! - exploit)

Iniziamo con ε = 1.0 (100% casuale) e lo diminuiamo lentamente fino a ε = 0.01 (1% casuale).
In questo modo, il robot esplora molto all'inizio, poi si concentra su ciò che ha imparato.

**Esempio di vita reale:** Quando visiti una nuova città, all'inizio potresti provare ristoranti a
caso (esplorazione). Dopo un po', torni nei tuoi preferiti (sfruttamento). Ma ogni tanto
provi comunque qualcosa di nuovo, nel caso ci sia un tesoro nascosto!

---

## Vocabolario Chiave

| Parola | Significato |
|--------|-------------|
| **Rete Neurale (Neural Network)** | Strati di neuroni matematici connessi che imparano dai dati |
| **Deep (Profondo)** | Più di uno strato nascosto (da cui "deep learning") |
| **DQN** | Deep Q-Network — usa una rete neurale invece di una Q-table |
| **ε-Greedy** | Strategia: a volte esplora casualmente, altre volte sfrutta la migliore conoscenza |
| **Instabilità** | La rete continua a "dimenticare" perché gli aggiornamenti interferiscono tra loro |

---

## Cosa Manca (e Perché è Importante)

Questo DQN "naive" ha due grandi problemi:

1. **Aggiornamenti correlati**: Ogni esperienza arriva in ordine (passo 1, passo 2, passo 3...).
   Se il passo 5 è andato male, TUTTI gli aggiornamenti vicini vengono confusi insieme.
   
2. **Bersaglio mobile (Moving target)**: Dopo ogni aggiornamento, la rete cambia. Ma l'aggiornamento successivo usa
   la STESSA rete per calcolare quale dovrebbe essere il target. È come sparare a un bersaglio
   in movimento!

Questi problemi vengono risolti dall'**Experience Replay** e dalle **Target Networks** nei prossimi
script. Insieme, trasformano il DQN da un principiante traballante in un campione di videogiochi!
