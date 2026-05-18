# Compiti a Lungo Orizzonte (Long-Horizon Tasks)

## La grande idea: quando la ricompensa è molto lontana {#the-big-idea-when-the-reward-is-very-far-away}

Immagina di essere uno chef che cerca di imparare una nuova ricetta basandosi esclusivamente sull'assaggio del piatto finale. Segui 40 passaggi — tagliare, saltare, condire, sobbollire, impiattare — ma ricevi un feedback solo alla fine: "Troppo salato". Quale dei 40 passaggi ha causato il problema? Non ne hai idea.

Questo è il **problema a lungo orizzonte (long-horizon problem)**: quando il segnale di ricompensa è separato dalle decisioni che lo hanno causato da dozzine (o centinaia) di passaggi, l'apprendimento diventa molto difficile.

---

## Perché gli agenti piatti hanno difficoltà {#why-flat-agents-struggle}

Un agente RL "piatto" (come gli agenti DQN della Fase 3) cerca di imparare il valore di ogni singolo passaggio tutto in una volta. In compiti brevi — bilanciare un'asta, evitare un muro — questo funziona bene. La ricompensa arriva velocemente e l'agente può collegare causa ed effetto.

Ma in un compito lungo — raccogliere una chiave, poi usarla per aprire una porta, quindi uscire dal labirinto — l'agente deve:

1. Imbattersi nella chiave (fortuna!)
2. Ricordare che raccogliere chiavi è utile
3. Imbattersi nella porta (di nuovo fortuna!)
4. Collegare l'intera sequenza alla singola ricompensa all'uscita

Con l'esplorazione casuale, la probabilità di completare accidentalmente questa intera sequenza si riduce esponenzialmente con ogni nuovo passaggio richiesto. Il DQN piatto ha essenzialmente bisogno di essere fortunato moltissime volte prima di vedere una singola ricompensa positiva da cui imparare.

---

## La soluzione gerarchica: divide et impera {#the-hierarchical-solution-divide-and-conquer}

L'RL gerarchico scompone il compito lungo in una **struttura a due livelli**:

| Livello | Nome | Compito |
|-------|--------|-----|
| Alto | **Manager** | Sceglie il prossimo sotto-obiettivo |
| Basso  | **Lavoratore (Worker)** | Naviga verso quel sotto-obiettivo |

Questo è esattamente il modo in cui gli esseri umani affrontano compiti complessi. Non pianifichi il tuo viaggio in auto svolta per svolta prima di partire. Invece:

- **Manager (tu, a casa):** "Prima tappa: il distributore di benzina. Prossima tappa: l'imbocco dell'autostrada. Poi: uscita 42."
- **Lavoratore (tu, alla guida):** Gestisce tutte le singole decisioni di sterzata per raggiungere ogni tappa.

Il manager pensa per *punti di controllo (checkpoints)*. Il lavoratore pensa per *volanti*.

---

## Perché questo supera l'apprendimento piatto nei compiti lunghi {#why-this-beats-flat-learning-on-long-tasks}

Il lavoratore deve solo raggiungere il *prossimo sotto-obiettivo* — un compito breve con una ricompensa chiara e vicina. Riceve feedback rapidamente e impara in modo efficiente.

Il manager deve solo decidere l'*ordine dei sotto-obiettivi* — un problema molto più semplice rispetto alla pianificazione di ogni singolo passaggio.

Insieme, i due livelli dividono il difficile problema a lungo orizzonte in due semplici problemi a breve orizzonte.

---

## L'esperimento della griglia Chiave-Porta {#the-key-door-grid-experiment}

Il nostro script testa entrambi gli approcci su una **griglia aperta 9x9** con due oggetti:

- Una **CHIAVE (KEY)** in un angolo (deve essere raccolta per prima).
- Una **PORTA (DOOR)** nell'angolo opposto (conta solo se hai la chiave).

L'unica vera ricompensa è +1 quando l'agente raggiunge la porta *dopo* aver raccolto la chiave. Quella singola ricompensa richiede che due sotto-compiti sequenziali siano concatenati correttamente.

Due agenti competono:

**DQN Piatto:** Deve imbattersi per caso in entrambi i sotto-compiti nell'ordine corretto, quindi retro-propagare un segnale attraverso entrambi. Poiché il successo richiede due scoperte fortunate in un unico episodio, il DQN raramente impara qualcosa di utile.

**Agente Gerarchico:**
- Regola del Manager: "Vai prima alla chiave, poi alla porta."
- Il lavoratore ottiene **+1 ogni volta che raggiunge un sotto-obiettivo** — che sia la chiave o la porta.
- Due compiti brevi separati, ognuno con una chiara ricompensa vicina.

---

## Cosa mostrano i grafici {#what-the-charts-show}

![Risultati del compito a lungo orizzonte](outputs/long_horizon_tasks.png)

**Sinistra — Tasso di successo nel tempo:** L'agente gerarchico (blu) impara a risolvere il labirinto molto prima del DQN piatto (rosso). L'agente piatto potrebbe alla fine imparare anch'esso — dati abbastanza episodi — ma l'agente gerarchico ci arriva più velocemente perché il suo segnale di apprendimento è denso e locale.

**Destra — Prestazioni finali:** Il grafico a barre mostra il tasso di successo medio negli ultimi 500 episodi. Il vantaggio dell'agente gerarchico è chiaro: scomporre il problema in sotto-obiettivi lo rende trattabile.

---

## Dove si applica il pensiero a lungo orizzonte {#where-long-horizon-thinking-shows-up}

| Dominio | Esempio di lungo orizzonte |
|--------|---------------------|
| Robotica | Assemblare un dispositivo con 30 parti in ordine |
| Giochi | Vincere una partita a scacchi (molte mosse, un solo vincitore) |
| Linguaggio | Scrivere un intero articolo di ricerca (molte decisioni di scrittura, un unico punteggio di qualità) |
| Scienza | Eseguire un esperimento di più mesi e valutarne i risultati |

Questo è esattamente il motivo per cui sono state inventate le Feudal Networks (un'architettura in cui i manager stabiliscono obiettivi direzionali per i lavoratori di livello inferiore) e HIRO (RL gerarchico con sotto-obiettivi) — poiché l'RL piatto ha incontrato ostacoli su questi problemi, la scomposizione gerarchica è diventata la strategia dominante.

---

## Il collegamento con le politiche condizionate dall'obiettivo {#the-connection-to-goal-conditioned-policies}

Nota che il **lavoratore** nel nostro agente gerarchico è essenzialmente una **politica condizionata dall'obiettivo (goal-conditioned policy)** — riceve un sotto-obiettivo e naviga verso di esso. Questo è il design standard in HIRO e negli articoli correlati: il manager stabilisce gli obiettivi, il lavoratore è una politica condizionata dall'obiettivo che li insegue.

Le due idee — politiche condizionate dall'obiettivo e struttura gerarchica — sono quindi due facce della stessa medaglia, motivo per cui appaiono insieme in questo modulo.

---

## Riassunto in una frase {#one-sentence-summary}

> **I compiti a lungo orizzonte sono difficili perché la ricompensa arriva troppo tardi per insegnare le singole decisioni — l'RL gerarchico risolve questo problema inserendo sotto-obiettivi vicini che permettono al lavoratore di imparare rapidamente, mentre il manager gestisce la sequenza generale.**
