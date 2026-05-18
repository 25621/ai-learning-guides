# Behavioral Cloning (BC) 🐒

## Cos'è?

Immagina di voler imparare a giocare a tennis. Guardi centinaia di ore di partite registrate di Wimbledon e semplicemente **copi ciò che fanno i giocatori**. Non pensi se il loro colpo sia stato il *migliore* possibile — ti limiti a far corrispondere la posizione del tuo corpo alla loro e a colpire nello stesso modo.

Questo è il behavioral cloning (clonazione comportamentale). **Nessuna ricompensa. Nessuna pianificazione. Solo imitazione.**

In termini di RL: prendi il dataset di coppie `(stato, azione)` e addestra una rete neurale a predire l'azione partendo dallo stato, esattamente come un modello di classificazione di immagini predice "cane o gatto". L' "etichetta" (label) è l'azione intrapresa da chi ha raccolto i dati.

---

## Come differisce dalla "vera" RL Offline

| Approccio | Usa le ricompense? | Può battere i dati? |
|----------|---------------|---------------------|
| **BC**   | ❌ no         | ❌ no — al massimo, eguaglia la qualità media dei dati |
| **CQL** (e simili) | ✅ sì | ✅ sì — può "cucire" insieme le buone transizioni da dati misti |

Il BC è la "visione dell'apprendimento supervisionato" della RL. È incredibilmente semplice, spesso sorprendentemente efficace ed è la baseline universale. **Se un algoritmo di RL offline non riesce a battere il BC sullo stesso dataset, non ha fatto nulla.**

---

## Esempi di Vita Reale

- **Imparare a guidare dai video delle dashcam.** Guarda la strada, predi l'angolo del volante usato dall'umano. Due esempi storici:
  - **ALVINN (1989)** — il primissimo guidatore a rete neurale; una minuscola rete a 3 strati addestrata su input di telecamera e laser per guidare un furgone sulle autostrade.
  - **NVIDIA PilotNet (2016)** — una moderna CNN profonda addestrata end-to-end su filmati di dashcam; ha imparato il mantenimento della corsia e lo sterzo di base puramente imitando i guidatori umani, senza regole ingegnerizzate a mano.
- **Apprendista che copia uno chef stellato.** "Qualunque cosa faccia lo chef, la faccio anch'io". Funziona benissimo se lo chef è bravo; produce un cattivo chef se lo chef è scarso.
- **GitHub Copilot.** Il completamento automatico è addestrato per predire "quale codice scriverebbe un umano dopo?" — pura imitazione dei log del codice sorgente.
- **Imitare il fratello maggiore.** I bambini lo fanno per anni prima di iniziare a ragionare sul *perché* il fratello maggiore faccia quello che fa.

---

## La Matematica (In una riga)

Per ogni `(s, a)` nel dataset, minimizza:

```
perdita = -log π(a | s)        (cross-entropia)
```

Tutto qui. La politica `π` è solo un MLP che emette i logit delle azioni; l'addestramento è identico a MNIST. Analizziamo i termini tecnici:
- **`π` (Pi):** Il simbolo standard per "politica" — la regola o rete neurale che decide cosa fare.
- **MLP (Multi-Layer Perceptron):** Una rete neurale di base standard.
- **Logit:** I punteggi grezzi e non normalizzati emessi dalla rete prima di essere trasformati in probabilità.
- **Cross-entropia:** La formula standard per penalizzare un modello quando assegna una bassa probabilità alla risposta corretta.
- **MNIST:** Il famoso dataset per principianti di cifre scritte a mano.

Addestrare un agente a giocare a un gioco tramite BC è letteralmente identico ad addestrare una rete a riconoscere cifre scritte a mano in MNIST. In MNIST, l'input è un'immagine e l'output è una cifra (0-9). Nel BC, l'input è lo stato del gioco e l'output è l'azione (es. "muovi a sinistra").

---

## Cosa Fa il Nostro Codice

Lo script `behavioral_cloning.py`:

1. **Carica tutti e quattro i dataset** creati da `d4rl_dataset.py` (`random`, `medium`, `expert`, `medium-replay`).
2. Per ogni dataset, **addestra una politica BC separata** per 10.000 passi di gradiente di cross-entropia. La colonna delle ricompense viene completamente ignorata.
3. Ogni 2.500 passi, **valuta** la politica corrente eseguendo un rollout greedy nell'ambiente CartPole-v1 reale (20 episodi, mediati).
4. Genera grafici:
   - Un grafico a barre: ritorno finale del BC per dataset.
   - Un grafico della curva di apprendimento: come ogni variante del BC sale durante l'addestramento.

---

## Cosa Dovresti Vedere

Un'esecuzione tipica stampa:

```
Ritorni della valutazione finale:
  BC su random          ->    ~20  ± pochi   (≈ gioco casuale)
  BC su medium          ->   ~150  ± errore  (≈ la politica medium)
  BC su expert          ->   ~480  ± errore  (≈ la politica expert)
  BC su medium-replay   ->    ~60  ± errore  (≈ la MEDIA dei dati misti)
```

Il grafico a barre rende evidente la storia: **il ritorno del BC segue il ritorno medio del dataset.** Non può superare quel limite massimo perché non ha modo di preferire le parti "buone" di un dataset misto rispetto a quelle "cattive" — entrambe sono bersagli di imitazione ugualmente validi.

Questa è la conclusione: **il BC eredita il limite massimo dei dati.**

---

## BC vs CQL — Il Confronto Più Chiaro

Sul dataset **medium-replay** (il caso più realistico, con qualità mista):

| Metodo | Ritorno finale approssimativo | Perché? |
|--------|------------------------------:|---------|
| BC     | ~60   | Imita la *media* tra i fallimenti iniziali e le buone esecuzioni successive |
| CQL    | ~400+ | Usa le ricompense per preferire transizioni con Q elevato; cuce una buona politica da dati misti |

Quindi il CQL **batte i dati**, il BC **eguaglia i dati**. Questo è l'intero motivo per cui la RL offline è un campo di ricerca e non solo "fai apprendimento per imitazione". Quando i dati sono di qualità mista (come lo sono sempre i log reali), i metodi sensibili alle ricompense recuperano di più.

Sui dati **expert** il confronto si inverte: il BC eguaglia l'esperto (~480). Potresti chiederti perché il CQL "pareggi" qui invece di perdere. Poiché il CQL è progettato per essere *conservativo* e penalizzare le azioni non viste nel dataset, finisce per fare esattamente ciò che ha fatto l'esperto. Non può battere l'esperto (perché il punteggio massimo possibile è già raggiunto), ma non rovina attivamente la strategia dell'esperto. Pareggia semplicemente con le prestazioni del BC.

Questo è il famoso compromesso tra "qualità dei dati e algoritmo":

```
                            Dati EXPERT  →  BC vince, CQL pareggia
   Sofisticatezza           ↑         
   dell'algoritmo           Dati MIXED   →  CQL batte chiaramente BC
                            
                            Dati RANDOM  →  Tutti falliscono; serve esplorazione
```

---

## Dove si Colloca il BC nella RL Moderna

- **Pre-addestramento per RL online.** Molti sistemi moderni (RT-1, Voyager, bot di gioco) iniziano con il BC su dimostrazioni, per poi affinare online con PPO/SAC.
- **RLHF.** Il passo 1 di InstructGPT è il fine-tuning supervisionato — puro BC su risposte scritte da umani. PPO e modello di ricompensa arrivano dopo.
- **DAgger (Ross et al., 2011).** Un'estensione intelligente per risolvere il problema dell'**errore cumulativo**.
  *Perché l'errore cumulativo è un problema se il BC clona perfettamente?* Anche se un modello BC è accurato al 99%, quell'1% di errore alla fine si verifica. Quando succede, l'agente entra in uno stato che non ha mai visto nel dataset guidato perfettamente. Poiché è confuso, commette un errore più grande, allontanandosi ancora di più dai dati noti, accumulando errori fino a un fallimento totale (come finire in un dirupo).
  *La soluzione:* Potremmo semplicemente chiedere all'esperto di guidare per sempre, ma il tempo dell'esperto è costoso. Invece, DAgger lascia guidare la politica BC. Quando la politica commette un errore e finisce in uno stato strano, ci fermiamo, chiediamo all'esperto "cosa faresti *proprio qui*?" e aggiungiamo l'azione al dataset. Interpelliamo l'esperto solo sugli stati visitati dalla politica BC perché abbiamo bisogno che l'esperto ci insegni come recuperare dai nostri specifici errori.
- **Decision Transformer (Chen et al., 2021).** Un BC "intelligente" che condiziona la previsione dell'azione su un *ritorno-da-ottenere* (return-to-go) desiderato, trasformando essenzialmente la RL offline in una previsione del token successivo.

---

## Parole Chiave da Ricordare

| Parola | Significato |
|------|---------|
| **Apprendimento per imitazione** | Termine ombrello per "copia il dimostratore"; il BC è il membro più semplice |
| **Errore cumulativo** | Un piccolo errore del BC ti porta in stati mai visti nel dataset, dove gli errori si accumulano |
| **Dati di dimostrazione** | Traiettorie prodotte da un esperto, usate come set di addestramento per il BC |
| **Limite massimo dei dati** | Il ritorno del BC è limitato dal ritorno medio presente nel dataset |
| **DAgger** | Una soluzione interattiva per l'errore cumulativo |

---

## Riassunto in Una Sola Frase

> **Il behavioral cloning è semplicemente apprendimento supervisionato su coppie (stato, azione) — efficace quando i dati sono buoni, impotente quando i dati sono misti.**
