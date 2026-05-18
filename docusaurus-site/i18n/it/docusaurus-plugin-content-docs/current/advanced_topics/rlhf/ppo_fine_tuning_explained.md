# Fine-Tuning PPO: Affinare un Modello Senza Romperlo

## L'Idea Centrale

Una volta che abbiamo un modello di ricompensa che valuta le risposte, vogliamo che il nostro modello linguistico produca risposte con punteggi più alti. PPO (Proximal Policy Optimization) fa esattamente questo, ma aggiunge una cintura di sicurezza in modo che il modello non insegua il punteggio dimenticando come scrivere un testo normale.

Pensalo come una fase di rifinitura. Il modello parla già in modo fluente; lo spingiamo solo a parlare in un modo che il modello di ricompensa premi, mantenendo la sua voce riconoscibile.

## Un'Analogia della Vita Reale

Immagina uno chef che cucina già bene ma che ora sta imparando a compiacere uno specifico critico gastronomico.

Dopo ogni piatto il critico dà un voto. Lo chef ha due pressioni:

1. **Ottenere un punteggio più alto.** Cucinare in un modo che piaccia al critico.
2. **Non diventare irriconoscibile.** Se lo chef abbandona completamente il proprio stile - aggiungendo sale a manciate solo per inseguire un punteggio - il cibo diventa strano. I clienti smettono di venire.

PPO cattura entrambe le pressioni:

- La parte della **ricompensa** spinge il modello verso risposte che piacciono al giudice.
- La parte della **penalità KL** spinge il modello a tornare al modo in cui parlava prima dell'inizio dell'addestramento. La KL è solo un modo per misurare "quanto è diverso il nuovo comportamento dal vecchio comportamento".

Insieme dicono: *migliora, ma resta te stesso*.

## Come Funziona l'Apprendimento (Solo Intuizione)

Ogni round di addestramento si presenta così:

1. Prendi alcuni prompt. Lascia che il modello attuale produca delle risposte.
2. Valuta le risposte con il modello di ricompensa.
3. Confronta con il **modello di riferimento** - una copia congelata del modello prima dell'addestramento. Se le nuove risposte sono selvaggiamente diverse, sottrai una penalità KL dalla ricompensa.
4. Spingi il modello verso le risposte che hanno ottenuto un buon punteggio.

Il termine "Proximal" in PPO significa *non fare grandi salti*. Ogni aggiornamento è un piccolo passo attento. I grandi salti nell'addestramento della politica causano crash, motivo per cui i metodi precedenti come il vanilla policy gradient erano così instabili.

## Cosa Mostra l'Esperimento

Iniziamo con una nuova politica e un modello di ricompensa addestrato. PPO viene eseguito per 150 iterazioni, campionando batch di risposte e aggiornando la politica.

![PPO training](outputs/ppo_fine_tuning.png)

- **Sinistra** - il punteggio medio del modello di ricompensa sale costantemente. La politica sta imparando a produrre risposte che piacciono al giudice.
- **Centro** - la divergenza KL dal modello di riferimento cresce. La politica si sta allontanando da dove è partita. Questo è previsto, ma se crescesse senza controllo, il modello andrebbe alla deriva nel non-senso.
- **Destra** - la ricompensa modellata (ricompensa grezza meno la penalità KL) segue da vicino la ricompensa grezza all'inizio, poi rimane indietro mentre la KL sale. La penalità sta facendo il suo lavoro: far "pagare" il modello per essersi allontanato troppo.

In un sistema RLHF reale, si sintonizza il coefficiente KL finché il punteggio continua a salire ma il modello rimane coerente. Una penalità troppo piccola e il modello "hackera" la ricompensa emettendo strane frasi ripetitive. Troppo grande e il modello non migliora mai.

## Dove si Colloca nella Pipeline RLHF

Questo è il secondo passaggio della ricetta classica RLHF:

1. Addestrare un modello di ricompensa dalle preferenze.
2. **Affinare il modello linguistico con PPO utilizzando quel modello di ricompensa.**
3. (Opzionale) Saltare il passaggio 2 con DPO se si desidera un percorso più semplice.

PPO è il cavallo di battaglia che aziende come OpenAI e Anthropic hanno utilizzato per la prima ondata di modelli allineati, inclusi InstructGPT e il ChatGPT originale.

## Perché Questo è Importante Fuori dal Laboratorio

Il modello "migliora, ma non andare alla deriva" si presenta ovunque:

- Un pianista che esercita un pezzo difficile non cambia tutta la sua tecnica per azzeccare un passaggio - ciò romperebbe il resto del concerto.
- Un'azienda che modifica un sito web per aumentare le iscrizioni deve comunque mantenere il marchio riconoscibile per gli utenti esistenti.
- Una fabbrica che regola una manopola in un processo mantiene le altre vicine alle impostazioni note come buone.

PPO è solo una versione attenta di questa idea universale, scritta in matematica.

## Riassunto in una Frase

**Il fine-tuning PPO spinge un modello verso una ricompensa più alta mentre una penalità KL lo mantiene vicino al suo comportamento originale: migliora, ma resta te stesso.**
