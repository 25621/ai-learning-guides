# Esplorare gli Ambienti PettingZoo 🦓

## Cos'è PettingZoo?

Se hai fatto RL ad agente singolo, probabilmente hai usato **Gymnasium** (il successore di OpenAI Gym). Ogni ambiente ha lo stesso aspetto: `env.reset()`, `env.step(azione) → obs, reward, done, info` — una nuova *osservazione* del mondo, un segnale di *ricompensa* scalare, un flag *done* che dice "partita finita" e un dizionario *info* per extra di debug. Quell'uniformità è ciò che fa funzionare le librerie di RL.

**PettingZoo** è esattamente la stessa idea ma per *agenti multipli*. È uno zoo di ambienti multi-agente — tutti dietro una API ben definita:
- **Problemi giocattolo classici**: ambienti semplici come Sasso-Carta-Forbici per testare algoritmi di base.
- **Mondi a griglia cooperativi**: agenti che navigano in una griglia per raggiungere un obiettivo condiviso.
- **Atari multiplayer**: classici giochi competitivi come Pong.
- **MPE (Multi-Particle Environment)**: ambienti fisici in spazio continuo per coordinazione e competizione complessa.

Se riesci a scrivere codice che funziona su un ambiente PettingZoo, puoi collegarti a qualsiasi altro con quasi zero modifiche.

---

## I Due Stili di API

Le impostazioni multi-agente sono più complesse di quelle a agente singolo perché due agenti possono agire contemporaneamente, a turni o in ordini arbitrari. PettingZoo risolve questo problema con due API parallele:

### 1) AEC (Agent-Environment-Cycle)

Agisce un agente alla volta. L'ambiente cicla tra gli agenti in un certo ordine e ognuno riceve:
- un'**osservazione** — ciò che vede *in questo momento*,
- una **ricompensa** — il payoff guadagnato dall'azione *congiunta* nell'ultimo round completo (ovvero, cosa è successo come risultato dell'azione di *tutti* gli agenti, non solo la tua; in una partita a scacchi, ad esempio, la tua ricompensa riflette lo stato della scacchiera dopo l'ultima mossa dell'avversario, non solo la tua),
- un **flag di terminazione** — `True` quando l'episodio finisce *naturalmente* (es. scacco matto, qualcuno vince),
- un **flag di troncamento** — `True` quando l'episodio viene *interrotto* da un limite di tempo prima di raggiungere una fine naturale.

Questo stile è naturale per **giochi a turni** come scacchi, Go, poker.

```python
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = mia_politica(obs, agent)
    env.step(action)
```

### 2) Parallelo (Parallel)

Tutti gli agenti osservano e agiscono simultaneamente ad ogni passo. `step()` accetta un *dizionario* di azioni e restituisce dizionari di osservazioni e ricompense.

Questo stile è naturale per **giochi in tempo reale** come MPE (Multi-Particle Environments, dove tutti gli agenti-punto si muovono simultaneamente) o mondi a griglia multi-agente.

```python
obs, info = env.reset()
while env.agents:
    actions = {a: mia_politica(obs[a]) for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
```

I due stili sono **isomorfi** — strutturalmente equivalenti e interconvertibili: qualsiasi ambiente AEC può essere automaticamente avvolto (wrapped) per apparire come uno Parallelo, e viceversa. PettingZoo include i wrapper di conversione in modo da dover scrivere il codice solo per uno stile.

---

## Analogia con la Vita Reale

- **AEC = una serata di giochi da tavolo.** "Tocca ad Alice. Ora a Bob. Ora a Carol. Di nuovo ad Alice". Chiunque muova dopo vede l'ultimo stato del tabellone.
- **Parallelo = un videogioco multiplayer.** Tutti e quattro i giocatori premono pulsanti contemporaneamente; il gioco aggiorna il mondo 60 volte al secondo.
- **Perché le API uniformi contano.** Immagina se ogni videogioco multiplayer avesse bisogno del proprio joystick. PettingZoo è il "joystick universale" della MARL.

---

## Cosa Fa il Nostro Codice

Costruiamo da zero un ambiente **in stile PettingZoo**: il **Gioco di Coordinazione Iterato**. Due agenti scelgono ripetutamente il canale `0` o `1`:

- Stessa scelta → entrambi ottengono +1
- Scelta diversa → entrambi ottengono -1

L'**osservazione** che ogni agente riceve è la precedente *azione congiunta* — ciò che entrambi gli agenti hanno scelto nel round scorso, impacchettato in un singolo intero.
Concretamente: l'ultimo stato di ogni agente è uno tra `{inizio, 0, 1}` (3 stati), quindi la coppia si codifica come `3 × stato_agente_1 + stato_agente_2`, producendo 9 possibili interi (0 – 8). L'intero 0 è lo stato di "inizio" — segnala che non è stata ancora compiuta alcuna azione (l'inizio assoluto di un episodio).
Un episodio dura 25 passi, quindi il ritorno totale massimo è +25 per agente e il minimo è −25. **Il gioco casuale ottiene un punteggio ≈ 0** perché ad ogni passo due agenti casuali indipendenti scelgono 0 o 1 con uguale probabilità: coincidono il 50% delle volte (+1) e differiscono il 50% delle volte (−1), dando una ricompensa attesa per passo di 0.5 × (+1) + 0.5 × (−1) = **0**. Sommando su 25 passi, il ritorno atteso dell'episodio è anch'esso 0.

Successivamente:

1. **Dimostriamo l'interfaccia AEC** con un rollout casuale — questo conferma il ciclo AEC di base: `agent_iter()` fornisce l'agente di turno, `last()` legge l'osservazione corrente e la ricompensa accumulata dell'agente, e `step()` consegna l'azione scelta all'ambiente.
2. **Addestriamo due Q-learner indipendenti tramite l'interfaccia Parallela**. Ogni agente mantiene la propria tabella Q indicizzata dall'**osservazione dell'azione congiunta** (il singolo intero che codifica ciò che *entrambi* gli agenti hanno fatto lo scorso round), in modo da poter imparare "quando entrambi abbiamo scelto 0 l'ultima volta, dovrei scegliere di nuovo 0".
3. **Proviamo a importare la vera libreria `pettingzoo`** ed eseguiamo un rollout di uno dei suoi ambienti integrati (Sasso-Carta-Forbici) con una politica casuale. Se PettingZoo non è installato, saltiamo questo passaggio con un messaggio informativo.

### Cosa dovresti vedere

| Fase | Risultato Atteso |
|-------|----------|
| Rollout casuale (AEC)            | Ritorno medio dell'episodio vicino a **0** — gli agenti casuali scelgono i canali indipendentemente. |
| Q-learner indipendenti (Parallelo) — primi 100 ep | Circa **0** — ancora prevalentemente casuale mentre gli agenti esplorano. |
| Q-learner indipendenti — ultimi 100 ep             | Fortemente positivo, **+20 a +25** — **la coordinazione è emersa**: entrambi gli agenti hanno imparato a scegliere affidabilmente lo stesso canale ogni round. |

Il grafico `outputs/pettingzoo_coordination.png` mostra i ritorni dei singoli episodi (grigio) e una curva della **Media** mobile (blu). La media smussa gli episodi rumorosi in modo da poter vedere la tendenza: gli agenti passano dal gioco casuale non coordinato vicino a ~0 verso una **coordinazione** stabile vicino a ~+25. La linea verde tratteggiata segna il limite della coordinazione perfetta.

Se `pettingzoo` è installato, lo script esegue anche `pettingzoo.classic.rps_v2` per dimostrare che lo script funziona con la vera libreria esattamente come con il nostro ambiente personalizzato. Per abilitare quella sezione:

```bash
source ../../venv/bin/activate
pip install "pettingzoo[classic]"
```

---

## Perché Costruire Prima un Ambiente Personalizzato?

Perché **l'API è la lezione.** (Capire come strutturare l'interazione tra più agenti e l'ambiente è più importante delle specifiche regole del gioco). La RL multi-agente ha molte varianti (a turni, in tempo reale, cooperativa, competitiva, mista) e tutte rientrano nel pattern AEC / Parallelo. Una volta implementati questi due cicli, ogni ambiente PettingZoo è solo questione di collegare un diverso costruttore di ambiente — il codice di addestramento rimane lo stesso.

Questo è esattamente il modo in cui Gymnasium ha cambiato la RL ad agente singolo: rendendo l'ambiente una scatola nera dietro un'interfaccia uniforme.

---

## Dove il Q-learning Indipendente Aiuta e dove Danneggia

I giochi di coordinazione sono *indulgenti* — gli agenti condividono il segno della ricompensa, quindi i loro interessi sono allineati. Gli agenti indipendenti possono risolvere questo problema felicemente perché qualsiasi miglioramento di un agente aiuta l'altro.

Nei giochi **avversari** (RPS) il Q-learning indipendente oscilla all'infinito (mentre un agente si adatta, l'altro cambia strategia per contrastarlo, portando a un inseguimento senza fine).
Nei giochi **parzialmente osservabili** non può imparare affatto perché l'"osservazione" è solo un pezzo dello stato (un agente potrebbe essere penalizzato per una buona azione solo perché non poteva vedere cosa stava facendo l'altro agente). PettingZoo include entrambi i tipi di ambiente in modo da poter vedere questi fallimenti con i tuoi occhi.

---

## Parole Chiave da Ricordare

| Parola | Significato |
|------|---------|
| **PettingZoo**     | Il Gymnasium della RL multi-agente — una libreria di ambienti MARL standardizzati |
| **AEC**            | Agent-Environment-Cycle: un agente agisce per passo (a turni) |
| **API Parallela**  | Tutti gli agenti agiscono simultaneamente ad ogni passo |
| **MPE**            | Multi-Particle Environment, un popolare banco di prova cooperativo/competitivo incluso in PettingZoo. |
| **CTDE**           | Centralised Training, Decentralised Execution — addestra con una visione globale, distribuisci solo con osservazioni locali. |
| **Q-learning Indipendente** | Ogni agente esegue il Q-learning standard, ignorando l'esistenza di altri agenti che imparano. |

---

## Riassunto in Una Sola Frase

> **PettingZoo dà a ogni ambiente multi-agente la stessa forma — così il codice che scrivi oggi funzionerà ancora domani su un gioco totalmente diverso.**

Una volta che i due stili di API diventano naturali, puoi passare a MADDPG (critico centralizzato per agenti a controllo continuo), QMIX (miscelazione dei valori per team cooperativi), MAPPO (PPO multi-agente) o qualsiasi altro algoritmo MARL moderno — il lato ambiente del tuo codice non dovrà mai cambiare.
