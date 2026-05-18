# Sensibilità agli Iperparametri di PPO: Cosa conta di più?

## Perché gli Iperparametri sono importanti

Immagina di preparare una torta al cioccolato. La ricetta richiede:
- 2 uova
- 200g di farina
- 1 cucchiaino di lievito
- 35 minuti a 180°C

Se usi 10 uova, la torta esplode. Se usi 0,1 cucchiaini di lievito, non lievita. Se la cuoci a 300°C per 10 minuti, brucia fuori ed è cruda dentro.

**Gli iperparametri in PPO sono come gli ingredienti e le impostazioni del forno.** La giusta combinazione funziona a meraviglia; impostazioni errate possono impedire del tutto l'apprendimento.

Questo script testa sistematicamente 3 iperparametri chiave modificandone solo UNO alla volta, eseguendo ogni impostazione con 3 diversi semi casuali (seeds) e confrontando i risultati.

---

## I tre Esperimenti

### Esperimento 1: Clip Epsilon (ε)

```
ε = 0.05   (molto conservativo — sono consentiti solo piccoli cambiamenti di politica)
ε = 0.2    (standard — equilibrio tra sicurezza e velocità)
ε = 0.4    (aggressivo — consente grandi cambiamenti di politica)
```

**Cosa controlla ε?**

ε è la dimensione della "finestra di sicurezza" attorno alla vecchia politica:
```
il rapporto deve rimanere in [1 - ε,  1 + ε]
ε=0.05: rapporto in [0.95, 1.05]  ← cambiamenti minuscoli
ε=0.2:  rapporto in [0.80, 1.20]  ← standard  
ε=0.4:  rapporto in [0.60, 1.40]  ← grandi cambiamenti
```

**Esempio di vita reale:** Pensa a ε come a "quanto ti è permesso girare il volante dell'auto in una singola mossa".
- ε=0.05: Come guidare sul ghiaccio — solo micro-regolazioni
- ε=0.2:  Guida normale — curve ragionevoli
- ε=0.4:  Pilota da corsa — sterzate aggressive, rischio di **testacoda** (perdere il controllo perché il cambiamento è troppo drastico, come un'auto che sbanda fuori strada)

**Risultati attesi:**
- ε=0.05: Apprendimento lento ma stabile (troppo cauto)
- ε=0.2:  Buon equilibrio (il **valore "Riccioli d'oro"** — né troppo piccolo, né troppo grande, giusto — dal nome della fiaba in cui Riccioli d'oro sceglie la zuppa che non è né troppo calda né troppo fredda)
- ε=0.4:  Può imparare velocemente ma potrebbe **andare oltre (overshoot) e oscillare** (overshoot = superare la politica ottimale; oscillare = rimbalzare avanti e indietro attorno ad essa senza stabilizzarsi, come un pendolo che oscilla troppo lontano in entrambe le direzioni)

---

### Esperimento 2: Tasso di Apprendimento (Learning Rate)

```
lr = 1e-4  (lento ma stabile)
lr = 3e-4  (standard)
lr = 1e-3  (veloce ma rischioso)
```

**Cosa controlla il tasso di apprendimento?**

Il tasso di apprendimento è come la "dimensione del passo" quando si scala una collina (ogni passo = un aggiornamento dei pesi della rete neurale, spostandola leggermente nella direzione che migliora la ricompensa):
- Troppo piccolo: Ci vuole un'eternità per raggiungere la cima (converge lentamente)
- Troppo grande: Superi la vetta e cadi dall'altra parte (**diverge** — la ricompensa dell'addestramento crolla o fluttua selvaggiamente invece di migliorare costantemente)
- Giusto: Progresso costante verso la vetta

**Esempio di vita reale:** Accordare una corda di chitarra.
- lr=1e-4: Rotazioni minuscole della **chiave** (la manopola che ruoti per stringere o allentare una corda) — richiede un'eternità ma è preciso
- lr=3e-4: Accordatura normale — trova l'intonazione giusta in pochi giri
- lr=1e-3: Grandi **strattoni** alla chiave — potrebbe **spezzare** la corda (romperla completamente, proprio come aggiornamenti troppo grandi possono interrompere l'addestramento in modo irreversibile)!

**Risultati attesi:**
- lr=1e-4: Alla fine buono ma molto lento
- lr=3e-4: Migliori prestazioni complessive
- lr=1e-3: Rapido progresso iniziale, poi instabilità

---

### Esperimento 3: Epoche di Aggiornamento (K)

```
K = 3   (conservativo — pochi passaggi attraverso ogni batch)
K = 10  (standard)
K = 20  (aggressivo — molti passaggi attraverso ogni batch)
```

**Cosa controllano le epoche di aggiornamento?**

Dopo aver raccolto un **rollout** (= giocare per un periodo di tempo per raccogliere nuova esperienza — come uno studente che fa una sessione di compiti prima di rivederli), PPO impacchetta quell'esperienza in un **batch** (= l'insieme completo di tuple stato, azione, ricompensa di quel rollout). Quindi esegue K **passaggi** (= scansioni complete attraverso il batch, ogni passaggio aggiorna la rete una volta) sugli stessi dati.
Più epoche = estrarre più apprendimento da ogni batch, ma con il rischio di **overfitting su dati obsoleti** (= memorizzare schemi che erano veri sotto la vecchia politica ma non sono più validi una volta che la politica è stata aggiornata, come uno studente che memorizza l'esame dell'anno scorso e fallisce quello nuovo).

**Esempio di vita reale:** Uno studente che si esercita con un set di 20 problemi di matematica.
- K=3:  Fai ogni problema 3 volte → stai ancora imparando, non fare overfitting sul set di pratica
- K=10: Fai ogni problema 10 volte → solida padronanza di questi problemi specifici
- K=20: Fai ogni problema 20 volte → **memorizzi le soluzioni senza capire davvero la matematica** (= il modello si adatta perfettamente al batch specifico ma perde la capacità di generalizzare)!

> ⚠️ **"Ma i risultati per K=20 sembrano buoni — perché dovrei preoccuparmi?"**
> Il trucco del clipping di PPO limita quanto la politica può cambiare per ogni passaggio, quindi K=20 non causerà un crollo improvviso. Tuttavia, l'agente si sta adattando silenziosamente troppo a dati che non riflettono più ciò che la politica attuale sperimenterebbe effettivamente. Questo **rallenta l'apprendimento a lungo termine**: ogni rollout insegna all'agente meno di quanto dovrebbe, perché i passaggi successivi riciclano informazioni sempre più obsolete. Il danno è graduale, non drammatico — che è esattamente il motivo per cui è facile trascurarlo in brevi esperimenti.

Il clipping previene l'overfitting catastrofico, ma troppe epoche possono comunque rallentare l'apprendimento complessivo.

**Risultati attesi:**
- K=3:  Meno efficiente (parte del potenziale di apprendimento viene sprecato per batch)
- K=10: Buon equilibrio
- K=20: Rischio che la politica diventi **troppo sicura di sé su dati obsoleti** (= gli aggiornamenti della rete sono guidati da esperienze che non corrispondono più a ciò che la politica attuale incontrerebbe, erodendo silenziosamente l'efficienza campionaria)

---

## Come leggere i risultati

Il grafico mostra tre diagrammi, ciascuno dei quali varia un iperparametro:

```
Grafico a sinistra:  Clip Epsilon — quale ε impara più velocemente?
Grafico al centro:   Learning Rate — quale lr è più stabile?
Grafico a destra:    Epoche di aggiornamento — quale K trova la politica migliore?
```

Ogni linea è la **ricompensa media su 3 semi (seeds)** (per ridurre la casualità).

**Cosa cercare:**
1. **Velocità di apprendimento:** Quale linea raggiunge più velocemente una ricompensa elevata?
2. **Prestazioni finali:** Quale linea ottiene la ricompensa finale più alta?
3. **Stabilità:** Quale linea ha meno oscillazioni?

Un buon iperparametro bilancia tutti e tre!

---

## Metodologia: Sperimentazione Scientifica

Questo esperimento utilizza un design di **studio di ablazione** (= un metodo in cui si rimuove o si varia un componente alla volta per misurarne l'impatto individuale — chiamato così dalla pratica scientifica di rimuovere selettivamente un tessuto per studiarne la funzione):
1. Scegli i valori predefiniti: ε=0.2, lr=3e-4, K=10
2. Cambia UN solo parametro alla volta
3. Mantieni tutto il resto fisso
4. Confronta i risultati

Questo ci dice l'effetto di OGNI parametro in isolamento.

**Esempio di vita reale:** Testare se un nuovo fertilizzante aiuta le piante:
- Cambia il fertilizzante, mantieni tutto il resto uguale (stesso terreno, acqua, luce solare)
- Se le piante crescono meglio → il fertilizzante ha aiutato!

---

## Risultati Comuni nella Pratica

| Iperparametro | Troppo piccolo | Punto ottimale | Troppo grande |
|----------------|-----------|------------|-----------|
| **ε (clip)** | Convergenza lenta | ε ≈ 0.2 | Instabilità |
| **lr** | Troppo lento | da 2.5e-4 a 3e-4 | Divergenza |
| **K (epoche)** | **Spreco di dati** (scarta il rollout prima di estrarre il segnale completo) | K = 4-10 | Overfitting su dati di rollout obsoleti |
| **n_steps** | Troppo rumoroso | 128-2048 | **Errori di memoria OOM** (usa troppa RAM) |
| **batch_size** | Troppo rumoroso | 32-256 | **Errori di memoria OOM** (usa troppa RAM) |

Questi "punti ottimali" possono variare a seconda dell'ambiente!

---

## L'intuizione chiave: PPO è relativamente robusto

Rispetto agli algoritmi precedenti (come DQN senza reti bersaglio), PPO è relativamente robusto rispetto alle scelte degli iperparametri. Il meccanismo di clipping fornisce una rete di sicurezza naturale.

**Esempio di vita reale:** Un'auto con freni **ABS** (Anti-lock Braking System — una funzione di sicurezza che impedisce il bloccaggio delle ruote durante frenate brusche, mantenendo il controllo del conducente) rispetto a una senza:
- Senza ABS (DQN): Una curva sbagliata (iperparametro errato) e vai in testacoda
- Con ABS (PPO): L'auto si corregge da sola — iperparametri ragionevoli funzionano tutti bene

Questa robustezza è uno dei motivi principali per cui PPO è l'algoritmo di RL più popolare nella pratica!

---

## Punti Chiave

| Concetto | Linguaggio Semplice |
|----------|---------------------|
| **Studio di ablazione** | Cambia una cosa alla volta per vederne l'effetto |
| **Clip epsilon ε** | Confine di sicurezza — 0.2 è solitamente il migliore |
| **Tasso di apprendimento** | **Dimensione del passo** — quanto vengono regolati i pesi della rete dopo ogni batch (pensalo come la dimensione di ogni passo quando cammini verso un obiettivo). **2.5e-4 a 3e-4** è la notazione scientifica per 0.00025 a 0.0003 — questi sono moltiplicatori adimensionali, non valori temporali |
| **Epoche di aggiornamento K** | Quante volte riutilizzare ogni batch — 4-10 è lo standard |
| **Semi casuali (Random Seeds)** | Ogni esperimento viene ripetuto con diversi **semi casuali** (= il numero iniziale fornito al generatore di numeri casuali, che controlla tutte le scelte casuali nell'addestramento). L'uso di più semi rivela se i risultati sono coerenti o se si è stati solo fortunati |

---

## Riepilogo: Metodi del Gradiente della Politica a colpo d'occhio

```
REINFORCE              A2C                    PPO
     │                  │                      │
Episodi completi  Aggiornamenti N-step   N-step + clipping
Semplice ma rumoroso Più veloce ma instabile Stabile + efficiente
Migliore per ambienti Ambienti a difficoltà  Ambienti difficili
semplici               media                  (standard industriale)
```

**Se impari solo UN algoritmo da questa fase, impara PPO.** È la base di:
- Addestramento di ChatGPT di OpenAI (RLHF usa PPO)
- Seguiti di AlphaGo di DeepMind
- Gran parte della moderna ricerca sulla robotica
- AI che giocano ai videogiochi
