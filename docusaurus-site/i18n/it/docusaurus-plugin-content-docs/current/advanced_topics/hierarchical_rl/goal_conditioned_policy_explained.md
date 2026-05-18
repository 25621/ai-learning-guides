# Politica Condizionata dall'Obiettivo (Goal-Conditioned Policy)

## La grande idea: una politica per domarli tutti

Immagina di essere un corriere. Non hai bisogno di competenze completamente diverse per ogni indirizzo. Sai guidare, leggere una mappa e destreggiarti nel traffico — devi solo inserire la *destinazione di oggi* e andare.

Una **politica condizionata dall'obiettivo (goal-conditioned policy)** funziona allo stesso modo. Invece di addestrare un agente che può andare solo verso un obiettivo fisso, addestriamo un singolo agente che accetta qualsiasi obiettivo come input e capisce come arrivarci.

---

## Come differisce dall'RL standard

Nell'RL standard (trattato nelle fasi precedenti del corso), la funzione di ricompensa è predefinita: "raggiungi la cella (7, 7), ottieni +1". L'agente impara esattamente una cosa: come raggiungere *quella* cella.

Nell'RL condizionato dall'obiettivo, la ricompensa dipende dal fatto che l'agente raggiunga *qualunque obiettivo gli sia stato assegnato questa volta*. La politica impara:

> **"Dato dove mi trovo e dove voglio essere, cosa devo fare?"**

L'obiettivo viaggia *con* l'agente, come una destinazione digitata in un'app di navigazione.

---

## Il problema della ricompensa sparsa

Ecco l'inghippo: imparare da ricompense sparse (solo +1 all'obiettivo, 0 ovunque) è brutalmente difficile. La maggior parte dei tentativi fallisce — l'agente vaga a caso, non si imbatte mai nell'obiettivo e la rete non ottiene nulla di utile da cui imparare.

Immagina di cercare di imparare a lanciare un dardo bendato. Lanci mille volte e sbagli sempre. Dopo mille fallimenti, non hai ancora idea di cosa si provi a fare "un buon lancio".

È qui che entra in gioco l'**Hindsight Experience Replay (HER)**.

---

## Hindsight Experience Replay: fallire in avanti

Il trucco di HER è meravigliosamente semplice. Dopo un episodio fallito, HER chiede:

> *"Anche se non hai raggiunto il tuo obiettivo... dove sei finito effettivamente?"*

Quindi **riproduce lo stesso episodio**, ma finge che la posizione finale effettiva dell'agente **fosse** l'obiettivo fin dall'inizio. Improvvisamente, un episodio fallito diventa un successo — per un obiettivo diverso.

È come un giocatore di basket che continua a tirare a canestro e a sbagliare. HER direbbe: "Ok, hai colpito il muro a sinistra ogni volta. Congratulazioni — sei bravissimo a colpire il muro a sinistra! Registriamo quei tiri come tentativi riusciti di colpire il muro a sinistra". Con il tempo il giocatore sviluppa l'abilità di colpire *qualsiasi* bersaglio, e alla fine la trasferisce al canestro reale.

Questo trasforma migliaia di "fallimenti" in una ricca libreria di navigazioni *riuscite* verso molti punti diversi. L'agente impara a raggiungerli tutti, il che si generalizza all'obiettivo reale.

---

## Analogia con la vita reale: un bambino che impara a impilare i blocchi

Un bambino che cerca di mettere un blocco in un secchio sbaglia costantemente. Ma ogni "errore" fa atterrare il blocco da *qualche parte*. Se riproduci ogni errore come "stavi cercando di metterlo *proprio lì* — e l'hai fatto!", il bambino sviluppa abilità motorie fini su tutto il tavolo. Presto potrà posizionare un blocco ovunque — compreso il secchio.

---

## Cosa fa il nostro codice

Lo script `goal_conditioned_policy.py` viene eseguito in un **labirinto 7x7** con pareti. All'inizio di ogni episodio, viene scelta una cella obiettivo casuale. L'agente deve trovarla.

La politica riceve due input ad ogni passo:
1. Dove si trova attualmente l'agente
2. Dove vuole andare

Dopo ogni episodio (riuscito o meno), HER genera diversi "successi" sintetici aggiuntivi rietichettando le posizioni effettivamente visitate come obiettivi alternativi.

L'addestramento dura 3.000 episodi con un tasso di esplorazione decrescente — l'agente esplora di più all'inizio e poi si fida sempre più di ciò che ha imparato.

---

## Cosa mostrano i grafici

![Risultati della Politica Condizionata dall'Obiettivo](outputs/goal_conditioned_policy.png)

**Sinistra — Tasso di successo durante l'addestramento:** Ogni episodio è un successo (raggiunto l'obiettivo) o un fallimento. La curva sale costantemente man mano che l'abilità di navigazione universale dell'agente migliora. Alla fine, l'agente raggiunge quasi ogni volta qualsiasi obiettivo.

**Destra — Mappa di calore del tasso di successo degli obiettivi:** Dopo l'addestramento, testiamo l'agente su ogni possibile cella obiettivo e coloriamo ogni cella in base alla frequenza con cui l'agente la raggiunge. Il verde significa che l'agente raggiunge quel punto in modo affidabile; il rosso significa che ha ancora difficoltà. Un agente ben addestrato mostra principalmente verde in tutto il labirinto.

---

## Dove si applica nel mondo reale

| Applicazione | L'"obiettivo" |
|-------------|------------|
| Braccio robotico che raggiunge un punto | Posizione 3D target |
| Auto a guida autonoma | Coordinate GPS |
| Assistente basato su modello linguistico | Istruzione dell'utente |
| Personaggio non giocante di un videogioco | Qualsiasi punto di passaggio (waypoint) sulla mappa |

Le politiche condizionate dall'obiettivo sono uno dei mattoni fondamentali per HIRO (Hierarchical RL with subgoals) — il manager di alto livello sceglie un sotto-obiettivo e il lavoratore (worker) di basso livello è esattamente questo tipo di politica condizionata dall'obiettivo.

---

## Riassunto in una frase

> **Una politica condizionata dall'obiettivo è un agente in grado di navigare verso qualsiasi destinazione — e HER rende possibile imparare dai fallimenti fingendo che ogni colpo mancato fosse mirato esattamente al punto in cui è atterrato.**
