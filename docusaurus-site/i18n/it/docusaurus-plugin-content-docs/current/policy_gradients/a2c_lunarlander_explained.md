# A2C: L'Attore e il Critico lavorano insieme

## L'Idea di Base

REINFORCE aspetta che la partita sia completamente finita prima di aggiornarsi. È come un allenatore che guarda un'intera partita di calcio in silenzio, per poi dare tutti i feedback alla fine.

**A2C (Advantage Actor-Critic)** fornisce feedback DURANTE la partita — ogni pochi passi, l'allenatore si ferma per dire: "Quel passaggio è stato fantastico! Quel tackle è stato pessimo!"

Questo è molto più veloce ed efficiente.

---

## Incontra i due Personaggi

> **Cos'è LunarLander?** In questo documento utilizziamo l'ambiente **LunarLander** — una simulazione fisica in cui controlli una piccola navicella spaziale e devi farla atterrare dolcemente su una piattaforma bersaglio sulla luna usando tre motori (sinistro, principale e destro). È un benchmark standard nell'apprendimento per rinforzo, disponibile nella libreria Gymnasium.

### L'Attore 🎭
L'**Attore** è la politica — decide quale azione intraprendere.

> "Sono in questo stato. Dovrei attivare il motore sinistro o quello destro?"

**Esempio di vita reale:** Il *conducente* di un'auto che gira il volante e preme i pedali.

### Il Critico 🎬
L'**Critico** stima quanto sia buona la situazione attuale — il valore V(s).

> "Trovarsi in QUESTO stato vale circa +150 punti di ricompensa futura totale."

**Esempio di vita reale:** Il *navigatore* seduto accanto al conducente, che dice: "Siamo su una buona strada — prevedo di arrivare in 30 minuti" o "Stiamo andando verso il traffico — sarà lenta".

### Condividono un Cervello
Nella nostra implementazione, entrambi utilizzano la **stessa struttura di rete neurale**:

```
          Stato (8 numeri per LunarLander)
                       ↓
          ┌─────────────────────────┐
          │  Strati Condivisi       │
          │  [256 neuroni] → ReLU   │
          │  [256 neuroni] → ReLU   │
          └────────┬────────┬───────┘
                   ↓        ↓
          Testa dell'Attore  Testa del Critico
          [4 output]         [1 output]
          (prob. azioni)     (V(s))
```

- **ReLU** (Rectified Linear Unit): una funzione di attivazione applicata dopo ogni strato — restituisce `max(0, x)`, mantenendo i valori positivi e azzerando i negativi. Ciò consente alla rete di apprendere schemi non lineari.
- **prob. azioni**: la probabilità di intraprendere ciascuna delle 4 azioni. L'Attore campiona da questa distribuzione per scegliere un'azione ad ogni passo.

**Esempio di vita reale:** Un solo cervello, due compiti — come un tassista che sia guida (attore) SIA sa se il percorso è buono (critico). Condividere il cervello significa imparare più velocemente!

---

## Il Vantaggio: È stato meglio del previsto?

Proprio come REINFORCE con baseline, A2C calcola il **Vantaggio**:

> A(s, a) = "Risultato effettivo" − "Cosa ci aspettavamo"

Ma qui, il "risultato effettivo" deriva dal **n-step bootstrap** del Critico (**bootstrapping** = usare la previsione stessa del Critico V(s) per approssimare il valore dei passi futuri, invece di aspettare che l'episodio finisca — come stimare il punteggio del tuo esame finale a metà semestre usando il tuo voto attuale):

```
Ritorno TD effettivo: r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γⁿ · V(s_{t+n})
Vantaggio A_t = Ritorno TD - V(s_t)
```

**Esempio di vita reale:** Ti aspetti di segnare 3 gol in questa partita (V(s)). Se segni 5 gol, il tuo vantaggio è +2. Se segni 1 gol, il tuo vantaggio è -2.

Vantaggio positivo → "quell'azione ha aiutato più del previsto → falla di più!"
Vantaggio negativo → "quell'azione ha aiutato meno del previsto → falla di meno!"

---

## Perché usare più ambienti paralleli?

Il nostro A2C utilizza **8 copie** di LunarLander in esecuzione contemporaneamente!

**Perché?** Perché le esperienze di un singolo ambiente sono **correlate** — un passo segue da vicino il passo precedente. Questa correlazione inganna la rete neurale facendole credere che gli schemi siano più affidabili di quanto non siano in realtà.

Con 8 ambienti, ogni passo fornisce 8 esperienze indipendenti da situazioni molto diverse. Questo rompe la correlazione e stabilizza drasticamente l'addestramento.

**Esempio di vita reale:** Per imparare il meteo, cosa è meglio:
- Osservare una città per 8 ore consecutive (correlato — se c'era il sole alle 14:00, probabilmente c'era il sole alle 15:00)
- Osservare 8 città simultaneamente (decorrelato — diversi modelli meteorologici, più informazioni!)

```
Ambiente 1: [atterrato sulla luna, motore sinistro, crash, reset...]
Ambiente 2: [caduta troppo veloce, entrambi i motori, stazionamento, atterraggio...]
Ambiente 3: [inclinazione a destra, motore destro, stabilizzazione, atterraggio...]
...
Ambiente 8: [deriva a sinistra, motore sinistro, stabile, ...]
```

Tutti e 8 aggiornano la rete simultaneamente — un'esperienza 8 volte più diversificata per aggiornamento!

---

## Aggiornamenti N-Step: Non aspettare che il gioco finisca

REINFORCE aspetta un episodio completo (potrebbero essere 1000 passi!).

A2C si aggiorna ogni **n_steps = 128 passi**:

```
Gioca 128 passi in 8 ambienti
    → Ottieni 128 × 8 = 1024 tuple di esperienza
    → Calcola vantaggi e ritorni
    → Aggiorna l'Attore e il Critico
    → Gioca altri 128 passi...
```

**Esempio di vita reale:** Uno studente che studia per un esame.
- Stile REINFORCE: Leggi l'intero libro di testo, POI fai i test di pratica.
- Stile A2C: Leggi 10 pagine, fai un quiz, leggi altre 10 pagine, fai un quiz...

Feedback più frequente = apprendimento più veloce!

---

## Tre Perdite Combinate

A2C si addestra con tre termini di perdita contemporaneamente:

Una **perdita** (loss) è il numero che l'ottimizzatore cerca di minimizzare. Una perdita minore significa che il comportamento attuale della rete è più vicino all'obiettivo dell'addestramento.

### 1. Perdita dell'Attore (Gradiente della Politica)
Rendi più probabili le azioni vantaggiose:
```
L_actor = -E[log π(a|s) · A(s,a)]
```
Se A > 0: aumenta la probabilità di quell'azione
Se A < 0: diminuisci la probabilità di quell'azione

### 2. Perdita del Critico (MSE della Funzione Valore)
Rendi più accurate le previsioni del valore (**MSE** = Mean Squared Error: eleva al quadrato l'errore di previsione e fai la media — elevare al quadrato penalizza gli errori grandi più pesantemente di quelli piccoli):
```
L_critic = E[(V(s) - ritorno)²]
```
Come addestrare qualsiasi modello di **regressione** (regressione = prevedere un numero continuo, qui il ritorno atteso V(s)) — minimizza l'errore di previsione.

### 3. Bonus di Entropia (Esplorazione)
Evita che la politica diventi troppo sicura di sé troppo velocemente:
```
L_entropy = -H[π(·|s)] = E[log π(a|s)]
```
Alta entropia = scelte d'azione diverse = esplorazione
Bassa entropia = scelte sicure e ristrette = sfruttamento

**Esempio di vita reale:** Il bonus di entropia è come un insegnante che dice: "Non limitarti a indovinare A in ogni domanda a scelta multipla! Prova diverse risposte così impari cosa funziona."

```
Perdita totale = L_actor + 0.5 × L_critic - 0.01 × entropia
```

---

## LunarLander: Una sfida più difficile

**LunarLander-v3** è un ambiente Gymnasium (precedentemente OpenAI Gym) — "v3" è il numero di versione che indica la terza revisione di questo ambiente. L'agente controlla una piccola navicella spaziale che deve atterrare in sicurezza su una piattaforma designata sulla luna. È molto più difficile di CartPole:
- Spazio degli stati a 8 dimensioni (posizione, velocità, angolo, contatto gambe, carburante)
- 4 azioni discrete (non fare nulla, motore sinistro, motore principale, motore destro)
- Ricompensa: +100 per l'atterraggio, -100 per il crash, piccole penalità per il carburante

La curva di apprendimento mostra un miglioramento graduale da ricompense altamente negative verso ricompense positive. A2C su LunarLander richiede un'esperienza significativa prima che il lander apprenda la stabilità di base.

---

## Equazioni Chiave

```
ritorno n-step:  G_t = r_t + γ·r_{t+1} + ... + γⁿ·V(s_{t+n})
Vantaggio:       A_t = G_t - V(s_t)
Aggiornamento Attore:   θ_π ← θ_π - α ∇ L_actor
Aggiornamento Critico:  θ_V ← θ_V - α ∇ L_critic
```

---

## Punti Chiave

| Concetto | Linguaggio Semplice |
|----------|---------------------|
| **Attore** | La politica — decide cosa fare |
| **Critico** | La funzione valore — giudica quanto sia buona la situazione |
| **Vantaggio** | "È stato meglio del previsto?" (effettivo - previsto) |
| **Ritorno n-step** | Guarda n passi nel futuro prima del bootstrapping con V(s) |
| **Ambienti paralleli** | Più ambienti per un'esperienza diversificata e decorrelata |
| **Bonus di entropia** | Incoraggia l'attore a continuare a provare nuove cose |

---

## Cosa viene dopo?

A2C è ottimo ma ha un punto debole principale: a volte aggiorna la politica in modo troppo aggressivo. Un singolo aggiornamento errato può distruggere tutto il buon apprendimento di un aggiornamento precedente.

**PPO (Proximal Policy Optimization)** risolve questo problema con un intelligente "safety clip" che impedisce a ogni singolo aggiornamento di cambiare troppo la politica.

Vedi `ppo_scratch.py` per l'implementazione PPO!
