# Giochi a Matrice: Il Mondo Multi-Agente più Semplice 🎲

## Cos'è un Gioco a Matrice?

Immagina che tu e un amico scegliate un segno con la mano — **sasso, carta o forbici** — *contemporaneamente*. Non vedete la scelta dell'altro. Il vincitore viene deciso da una piccola tabella:

|        | Sasso | Carta | Forbici |
|--------|:----:|:-----:|:--------:|
| Sasso    |  0,0  | -1,+1 | +1,-1 |
| Carta    | +1,-1 |  0,0  | -1,+1 |
| Forbici  | -1,+1 | +1,-1 |  0,0  |

Quella tabella è l'intero mondo del gioco. Nessun movimento, nessun tempo, nessuna mappa. Solo una decisione istantanea. Chiamiamo questo un **gioco a matrice** perché la matrice dei payoff rappresenta l'intero ambiente.

I giochi a matrice sono il posto più semplice dove studiare la **RL multi-agente**, perché l'unica cosa che può cambiare durante l'addestramento è la *politica* di ciascun giocatore — la probabilità di scegliere ogni azione.

---

## Perché è "Multi-Agente"

Nella RL a agente singolo l'ambiente è fisso: il vento soffia sempre nello stesso modo, il pavimento non si muove mai. L'agente migliora e alla fine vince.

In un gioco a matrice, il tuo "ambiente" è *un altro agente che impara*. Mentre l'altro diventa più intelligente, ciò che conta come una buona mossa per te *cambia*. Questo fenomeno è chiamato **non-stazionarietà** ed è il problema centrale della RL multi-agente.

> Se continui a giocare Sasso, il tuo avversario alla fine inizierà a giocare sempre Carta. Quindi tu passi a Forbici. Allora lui passa a Sasso. Tu passi a Carta... e così via. La "mossa migliore" non rimane mai fissa.

La soluzione classica sono le **strategie miste**: non scegliere un'azione in modo deterministico — randomizza in modo che l'avversario non possa sfruttarti.

---

## I Tre Giochi che Esploriamo

### 1) Sasso-Carta-Forbici (a somma zero)
- Il guadagno di un giocatore è la perdita dell'altro.
- L'**equilibrio di Nash** è: ogni giocatore sceglie ogni azione con probabilità ⅓. Qualsiasi deviazione è sfruttabile.
- Ci aspettiamo che i nostri due Q-learner oscillino intorno a ⅓-⅓-⅓ — mai perfettamente stabili, perché ogni volta che uno si sposta, l'altro reagisce.

### 2) Dilemma del Prigioniero (a somma generale)
Due sospettati vengono interrogati separatamente:

|           | Coopera | Defeziona |
|-----------|:---------:|:------:|
| Coopera   |   3, 3    |  0, 5  |
| Defeziona |   5, 0    |  1, 1  |

- "Defeziona" batte "Coopera" indipendentemente da ciò che fa l'altro — è una **strategia dominante**.
- Entrambi i giocatori sono razionali → entrambi defezionano → entrambi ottengono 1, anche se (Coopera, Coopera) avrebbe dato 3 a testa. La migliore risposta egoistica distrugge il benessere del gruppo.
- Ci aspettiamo che il Q-learning converga chiaramente verso (Defeziona, Defeziona).

### 3) Caccia al Cervo (coordinazione)
Due cacciatori possono abbattere insieme un cervo (grande premio) o accontentarsi di una lepre ciascuno (premio piccolo ma sicuro):

|       | Cervo | Lepre |
|-------|:----:|:----:|
| Cervo | 4, 4 | 0, 3 |
| Lepre | 3, 0 | 2, 2 |

- (Cervo, Cervo) è **dominante nel payoff** — il meglio per entrambi.
- (Lepre, Lepre) è **dominante nel rischio** — sicuro se non ti fidi del partner.
- Il risultato dipende dalle condizioni iniziali: i Q-learner indipendenti spesso finiscono nell'equilibrio *peggiore* (Lepre, Lepre) perché le lepri sono più sicure da imparare.

---

## Esempi di Vita Reale

- **Prezzi in un duopolio.** Due caffetterie sulla stessa strada scelgono un prezzo ogni mattina. La forma della matrice dei payoff decide se finiranno a un prezzo alto e "cooperativo" (buono per loro, cattivo per i clienti) o a un prezzo basso e agguerrito.
- **Protocolli di rete.** Router e mittenti scelgono strategie temporali; il risultato della congestione della rete è determinato dai payoff (stile gioco a matrice) tra il passare e il ritirarsi.
- **Offerte in un'asta.** Ogni offerente sceglie un'offerta senza conoscere quelle degli altri; i payoff dipendono dall'intero vettore. L'equilibrio di Nash è una *strategia di offerta*, non un singolo numero.

---

## Cosa Fa il Nostro Codice

Per ogni gioco:
1. Creiamo due Q-learner senza stato (Q è solo un numero per azione — non ci sono stati in un gioco istantaneo).
2. Eseguiamo un ciclo per 20.000 passi. Ad ogni passo: entrambi gli agenti scelgono un'azione ε-greedy simultaneamente, ottengono una ricompensa, aggiornano i loro valori Q.
3. Tracciamo la **frequenza empirica delle azioni** di ciascun agente in una finestra mobile di 500 passi. Invece di guardare solo le probabilità astratte, contiamo quali azioni hanno effettivamente scelto di recente (es. "negli ultimi 500 round, hanno giocato Sasso il 40% delle volte"). Questo ci dà un'immagine pratica e in tempo reale della loro strategia che cambia.
4. Tracciamo le frequenze nel tempo, salviamo in `outputs/<gioco>.png` e stampiamo i valori Q finali.

### Cosa dovresti vedere

| Gioco | Risultato atteso del grafico |
|------|------------------------------|
| **Sasso-Carta-Forbici** | Entrambi i giocatori gravitano intorno a ⅓-⅓-⅓ ma oscillano visibilmente. Le curve si inseguono — un classico comportamento ciclico. |
| **Dilemma del Prigioniero** | La frequenza di "Defeziona" di entrambi i giocatori sale rapidamente a ~1.0. "Coopera" viene schiacciato. |
| **Caccia al Cervo** | La maggior parte dei seed casuali si stabilizza su (Lepre, Lepre). Alcuni seed fortunati raggiungono (Cervo, Cervo) — prova a cambiare il seed nello script. |

---

## Dove l'Apprendimento Indipendente Fallisce

I nostri agenti sono *indipendenti* — vedono solo la propria ricompensa, mai l'azione o i valori Q dell'avversario. Questa è la baseline più semplice e ha dei limiti:

- **Non può garantire la convergenza** nei giochi a somma generale.
- Può rimanere bloccata in **equilibri negativi** (Caccia al Cervo).
- **Non può modellare l'avversario.**

Gli algoritmi multi-agente reali risolvono questo problema ragionando esplicitamente sull'altro agente. Ecco cosa fa ciascuno di essi:

| Algoritmo | Idea centrale | Analogia con la vita reale |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| **Fictitious play** | Tiene un conteggio aggiornato di quante volte l'avversario ha scelto ogni azione. Assume che domani farà ciò che ha sempre fatto e sceglie la propria risposta migliore a tale convinzione. | Osservare le abitudini di un avversario in molte partite di scacchi e regolare la propria apertura di conseguenza. |
| **CFR (Counterfactual Regret Minimisation)** | Dopo ogni round, chiede: *"Quanto mi sono pentito di non aver scelto le altre azioni?"* Sposta gradualmente la probabilità verso le azioni che si è pentito di non aver fatto. Usato nel poker perché gestisce giochi a **informazione imperfetta** (non vedi le carte dell'avversario). | Dopo una mano di poker, ripensarci e dire: *"Avrei dovuto puntare di più — lo farò la prossima volta."* |
| **LOLA (Learning with Opponent-Learning Awareness)** | Il passo del gradiente tiene conto del fatto che anche l'avversario sta facendo passi del gradiente. Ottimizzi il tuo aggiornamento anticipando quello dell'avversario — due passi avanti invece di uno. | Negoziare un accordo pensando: *"Se offro X, risponderanno con Y, quindi dovrei iniziare con Z."* |
| **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** | Il *critico* (stimatore del valore) di ogni agente è addestrato con una **visione globale**: vede le osservazioni e le azioni di tutti. L'*attore* (la politica distribuita) usa ancora solo informazioni locali — questo è lo schema CTDE (Centralized Training with Decentralized Execution). | Un allenatore di basket che osserva l'intero campo (critico centralizzato) ma insegna a ogni giocatore a reagire solo a ciò che può vedere (attore decentralizzato). |

Tuttavia, il Q-learning indipendente è il giusto primo passo. Vedi il problema della non-stazionarietà colpirti direttamente e le soluzioni acquistano senso successivamente.

---

## Parole Chiave da Ricordare

| Parola | Significato |
|------|---------|
| **Matrice dei payoff** | La tabella che definisce un gioco multi-agente istantaneo |
| **Equilibrio di Nash** | Un profilo di politiche in cui nessun agente può migliorare deviando da solo |
| **Strategia mista** | Una politica che randomizza su più azioni |
| **Non-stazionarietà** | L'ambiente (= gli altri agenti) continua a cambiare mentre impara |
| **Agente indipendente** | Un agente che ignora l'esistenza di altri agenti che imparano |
| **A somma zero** | Il guadagno di un agente è esattamente la perdita dell'altro |
| **A somma generale** | Entrambi gli agenti possono vincere, entrambi possono perdere, o qualsiasi cosa nel mezzo |

---

## Riassunto in Una Sola Frase

> **Nei giochi a matrice, l'"ambiente" è un altro agente che impara — quindi la mossa migliore continua a spostarsi.**

Questa è l'idea di base dietro ogni algoritmo multi-agente che incontrerai in seguito, dal self-play a MADDPG alla MARL con comunicazione.
