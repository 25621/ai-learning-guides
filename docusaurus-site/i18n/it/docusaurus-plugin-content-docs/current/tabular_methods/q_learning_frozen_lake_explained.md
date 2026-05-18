# Agente di Q-Learning per Frozen Lake 🧊

## Cos'è?

Immagina un laghetto ghiacciato con ghiaccio scivoloso. C'è un **quadrato di Inizio (Start)** e un **quadrato Obiettivo (Goal)**
con alcuni **Buchi (Holes)** nel mezzo. Se cadi in un buco, ricominci da capo!

Il ghiaccio è scivoloso, quindi anche se provi a camminare verso destra, potresti scivolare verso l'alto o verso il basso.
Un **agente di Q-Learning** è un robot che impara — provando e riprovando — come arrivare dall'Inizio all'Obiettivo senza cadere!

---

## Cosa significa la "Q" in Q-Learning?

La **"Q"** sta per **"Qualità" (Quality)** — nello specifico, la *qualità* di compiere una particolare
azione in una particolare situazione.

Pensalo come la valutazione di un ristorante: "Quanto è buono (qualità) ordinare la pizza in QUESTO
ristorante?" Q(s, a) chiede: "Quanto è buono compiere l'azione **a** quando mi trovo nello stato **s**?"

Un valore Q alto significa: "Ottima scelta! Questa azione porta a molte ricompense."
Un valore Q basso significa: "Pessima idea! Questa azione di solito porta a problemi."

**Esempio di vita reale:** Immagina di essere un bambino che decide se mangiare caramelle prima di cena.
Il tuo valore Q per "mangiare caramelle ora" potrebbe essere alto in questo momento (hanno un ottimo sapore!) ma basso nel complesso
(la mamma si arrabbia, ti senti male dopo). Il Q-learning impara a tenere conto di quelle conseguenze future — non solo della sensazione immediata!

---

## La Grande Idea: Una Tabella Magica di Punteggi

Il Q-Learning costruisce una grande tabella chiamata **Q-table**. Ogni riga è un quadrato sul ghiaccio,
e ogni colonna è un'azione (sinistra, destra, su, giù). I numeri all'interno sono **punteggi**:
"Quanto è buono compiere questa azione da questo quadrato?"

Ogni volta che il robot prova una mossa:
1. Riceve un feedback (è caduto? ha raggiunto l'obiettivo?)
2. Aggiorna il punteggio nella tabella usando questa formula:

> **Nuovo Punteggio = Vecchio Punteggio + Tasso di Apprendimento × (Cosa è successo realmente − Cosa mi aspettavo)**

Il robot si sta fondamentalmente chiedendo: "Questa mossa è stata migliore o peggiore di quanto pensassi?"

**Esempio di vita reale:** Pensa a un bambino che impara a camminare. Ogni volta che prova a fare un passo e cade,
impara che "quel passo era sbagliato". Ogni volta che ci riesce, ricorda che "ha funzionato!". Dopo
molti tentativi, capisce come camminare. Il Q-learning fa la stessa cosa, ma con una tabella!

---

## Cosa rende speciale il Q-Learning: È Off-Policy!

Ecco qualcosa di astuto: quando il Q-Learning aggiorna la sua tabella, *assume sempre che farà la
mossa perfetta la volta successiva*, anche se durante l'addestramento a volte esplora mosse casuali.

Questo rende il Q-Learning **off-policy**: la strategia che *impara* (scegli sempre la migliore
azione conosciuta) è separata dalla strategia che *segue* durante l'addestramento (a volte scegli un'azione casuale per esplorare). Concretamente, l'aggiornamento della Q-table usa il valore Q *massimo* dello stato successivo — il meglio teorico — anche quando la mossa successiva effettiva del robot sarà casuale.

In parole povere: il robot potrebbe vagare casualmente a sinistra per esplorare, ma il suo apprendimento calcola comunque come se dovesse compiere la *migliore* azione successiva. Questa separazione permette al Q-Learning di convergere verso la strategia ottimale indipendentemente da quanto esplora.

---

## Cosa ha scoperto il nostro codice

Abbiamo addestrato l'agente per **50.000 episodi** sul Frozen Lake 4×4 scivoloso:

| Metrica | Risultato |
|---------|-----------|
| Tasso di successo valutazione greedy | **73.1%** |
| Target milestone (>70%) | ✓ **SUPERATO** |

Il ghiaccio è molto scivoloso, quindi anche la migliore politica non può vincere il 100% delle volte!

La Q-table appresa mostra che l'agente ha capito: vai in basso e a destra evitando i buchi.

---

## Esempi di vita reale

- **Auto a guida autonoma**: Imparare quali corsie prendere agli incroci attraverso prove di guida.
- **Sistemi di raccomandazione**: Imparare quali film suggerire in base al fatto che gli utenti abbiano gradito i suggerimenti precedenti.
- **IA nei videogiochi**: Un personaggio che impara a navigare in un labirinto provando molti percorsi.

---

## Parole Chiave da Ricordare

- **Q-table**: La tabella di "quanto è buona ogni azione in ogni stato"
- **Q(s, a)**: Il punteggio per compiere l'azione a nello stato s
- **Reward (Ricompensa)**: Ciò che l'agente riceve dopo aver compiuto un'azione (+1 per il raggiungimento dell'obiettivo, 0 altrimenti)
- **Off-policy**: Impara la strategia ottimale anche mentre esplora casualmente
- **ε-greedy** (ε = epsilon): La maggior parte delle volte esegui la migliore azione conosciuta; a volte esplora casualmente
- **Fattore di sconto γ** (γ = gamma): Quanto valgono le ricompense future (come preferire i soldi subito rispetto a dopo)

La grande idea: **Il Q-Learning costruisce un "foglietto illustrativo" per ogni situazione e continua a migliorarlo finché non conosce la mossa migliore ovunque.**
