# PPO per il Controllo Continuo: Far camminare BipedalWalker

## Azioni Discrete vs. Continue

Finora, ogni ambiente che abbiamo risolto aveva azioni **discrete**:
- CartPole: spingi a SINISTRA o a DESTRA (2 scelte)
- LunarLander: non fare nulla / sinistra / principale / destra (4 scelte)

Ma i robot nel mondo reale hanno bisogno di azioni **continue**:
- Un robot umanoide: "quanto forte spingere ogni articolazione" (qualsiasi valore da -1 a +1)
- Un'auto: "esattamente quanto girare il volante" (qualsiasi angolo da -30° a +30°)
- Un braccio: "applica esattamente 2,3 Newton in questa direzione"

**Esempio di vita reale:** Digitare su una tastiera = discreto (premi A, B, C...).
Scrivere con una matita = continuo (muovi la mano di 2,3 cm a destra, premi con una forza di 40g...).

---

## La Politica Gaussiana per Azioni Continue

Per le azioni continue, invece di una distribuzione Categoriale (scegli tra N categorie), utilizziamo una **distribuzione Normale (Gaussiana)**:

```
Azione ~ Normale(μ, σ)
```

Dove:
- **μ (mu, media)**: Il centro della distribuzione — il valore dell'azione a cui la rete "mira"
- **σ (sigma, deviazione standard)**: La dispersione — quanta casualità / esplorazione aggiungere

```
        Probabilità
             │
        0.4 ─┤      ██████
             │    ████████████
        0.2 ─┤  ██████████████████
             │████████████████████████
             └──────────────────────── Valore dell'azione
           -1  -0.5   0   0.5   1
                      ↑
                   media μ
```

**Esempio di vita reale:** Un arciere esperto mira al centro del bersaglio (μ). Le sue frecce non atterrano tutte esattamente nello stesso punto — c'è una certa dispersione (σ). Man mano che si esercita, diventa più preciso (σ diminuisce) pur rimanendo centrato sul centro del bersaglio (bullseye).

---

## La nostra Rete Attore-Critico Gaussiana

```
Stato (24 numeri) → [256 neuroni] → [256 neuroni] →
    ├── Attore: 4 valori medi (μ₁, μ₂, μ₃, μ₄)
    │           + 4 parametri log_std (condivisi tra tutti gli stati!)
    └── Critico: 1 valore (V(s))
```

Il `log_std` (logaritmo della **deviazione standard** — una misura di dispersione o incertezza) è un **parametro apprendibile** — non dipende dallo stato. Questo lo mantiene semplice pur consentendo all'esplorazione di cambiare durante l'addestramento.

**Perché log_std invece di std?** La deviazione standard deve essere positiva. L'uso di `log_std` consente alla rete di produrre in output qualsiasi numero reale (positivo o negativo), quindi applichiamo `exp(log_std)` — la funzione esponenziale, che è l'inversa del logaritmo — per recuperare una deviazione standard garantita come positiva. Ciò impedisce che la deviazione standard diventi mai negativa o nulla.

---

## Calcolo della Log-Probabilità per Azioni Continue

Per le azioni discrete: `log_prob = log(P(azione=SINISTRA))`

Per le azioni continue, la **distribuzione Normale** descrive una curva a campana regolare attorno alla media. Un singolo valore esatto ha probabilità zero nella matematica del continuo, quindi usiamo l'altezza della curva in quel valore, chiamata **pdf** (funzione di densità di probabilità):
```
log_prob = Σᵢ log[Normale(μᵢ, σᵢ).pdf(aᵢ)]
```

`log` indica il logaritmo naturale. Trasforma minuscoli valori di densità in numeri stabili che sono più facili da ottimizzare per le reti neurali. Sommiamo tra tutte le dimensioni dell'azione (4 per BipedalWalker), perché l'azione completa è un vettore di 4 numeri.

**Esempio di vita reale:** Qual è la probabilità che domani ci siano esattamente 5,732...°C? Per il meteo continuo, guarderesti la curva della distribuzione Normale e vedresti quanto è alta in quel punto esatto. Le temperature più probabili (vicino alla media) hanno una probabilità più alta.

---

## BipedalWalker: Una sfida di camminata

BipedalWalker-v3 è un robot 2D che deve imparare a camminare senza cadere:

```
          O (testa)
         /│\
        / │ \
       /  │  \
      S   │   D   ← due gambe, ognuna con un'articolazione del ginocchio
     / \  │  / \
    ●   ● │ ●   ●  ← 4 motori (anca/ginocchio per ogni gamba)
```

**Spazio degli stati (24 numeri):**
- Scafo: angolo, velocità angolare, velocità orizzontale, velocità verticale (4 numeri)
- Articolazioni: 4 motori (2 anche, 2 ginocchia) ciascuno dei quali fornisce angolo e velocità, più 2 sensori di contatto con il suolo (uno per ogni gamba) (10 numeri)
- 10 sensori di distanza LIDAR (letture di distanza che vedono il terreno davanti) (10 numeri)

**Spazio delle azioni (4 valori continui, ciascuno in [-1, 1]):**
I valori delle azioni controllano la **coppia** (torque - la forza rotazionale applicata dai motori) per esattamente 4 articolazioni (nessuna azione viene applicata direttamente allo scafo):
- Coppia anca Gamba 1, Coppia ginocchio Gamba 1, Coppia anca Gamba 2, Coppia ginocchio Gamba 2

**Ricompense:**
- +300 per il raggiungimento dell'obiettivo (lato destro)
- -100 per la caduta (toccando il suolo con il corpo)
- Piccola ricompensa per ogni passo di progresso in avanti
- Piccola penalità per ogni uso del motore (ricompensa l'efficienza)

**Risolto quando:** La ricompensa media è > 300 su 100 episodi

---

## Differenza chiave rispetto al PPO discreto

Tutto è uguale ECCETTO:

| | PPO Discreto | PPO Continuo |
|---|---|---|
| **Politica** | Categoriale(logits) | Normale(μ, σ) |
| **Campionamento** | azione = campiona da {0,1,...,N} | azione = μ + σ × rumore |
| **log_prob** | log P(azione=k) | Σ log Normale(μᵢ, σᵢ).pdf(aᵢ) |
| **Clamp** | Non necessario | Forza le azioni in [-1, 1] |

I **Logits** sono punteggi grezzi e non normalizzati per azioni discrete. Una politica categoriale li converte in probabilità con **softmax** — una funzione che prende qualsiasi insieme di numeri e li schiaccia in una distribuzione di probabilità valida (tutti i valori positivi, con somma pari a 1). Ad esempio, i logits [2.0, 1.0, 0.5] diventano probabilità [0.59, 0.24, 0.17]. Il PPO continuo **non** usa softmax per l'azione stessa, perché l'azione non viene scelta da un menu fisso. Invece, la politica produce in output la media e la deviazione standard di una distribuzione Normale, quindi campiona coppie (torques) a valori reali da essa.

**Clamp** significa forzare un valore in un intervallo valido. Il codice utilizza `action.clamp(-1, 1)` in modo che l'ambiente non riceva mai un comando motore al di fuori dei suoi limiti consentiti.

**Clip** in PPO significa qualcosa di diverso: PPO taglia (clipping) il rapporto di probabilità all'interno della perdita (loss), come spiegato nella [sezione sul clipping del PPO](./ppo_scratch_explained.md#the-clipping-trick). Il clamping dell'azione protegge l'interfaccia dell'ambiente; il clipping del PPO protegge l'aggiornamento della politica.

---

## Camminare da zero: Cosa impara l'agente

**Inizio addestramento (ricompense negative):** Il robot si dimena casualmente, cade immediatamente. Ogni episodio finisce in un crash in pochi secondi.

**Metà addestramento:** Il robot scopre che muovere le gambe alternativamente crea progresso in avanti. Inizia a fare piccoli e goffi passi — la ricompensa diventa meno negativa.

**Fine addestramento:** Emerge un'andatura (**gait**) fluida ed efficiente. Un'andatura è un modello di movimento ripetuto, come l'alternanza di passi sinistri e destri. Il robot si adatta dinamicamente a terreni irregolari utilizzando i suoi sensori LIDAR per adattare i suoi passi in tempo reale.

**Esempio di vita reale:** Un bambino che impara a camminare:
1. Cade immediatamente (ricompensa negativa)
2. Fa un passo, cade (leggermente meno negativa)
3. Fa alcuni passi (piccola ricompensa positiva)
4. Cammina per la stanza (grande ricompensa positiva!)

---

## Perché BipedalWalker ha bisogno di PPO (non di REINFORCE)

- Gli **episodi di BipedalWalker** possono durare fino a 1600 passi (molto più lunghi di CartPole!)
- Le **ricompense sono scarse** — le ricompense per il progresso in avanti sono minuscole per ogni passo
- **REINFORCE avrebbe bisogno** di migliaia di episodi completi per ottenere un segnale utile

Gli aggiornamenti n-step di PPO con [GAE (Generalized Advantage Estimation)](./ppo_scratch_explained.md#gae-smarter-advantage-estimates) consentono al robot di imparare da episodi incompleti:
> "Anche se sono caduto dopo 50 passi, quei passi hanno mostrato un CERTO progresso in avanti. Usiamo una stima del ritorno a 50 passi piuttosto che aspettare il completamento dell'episodio."

---

## Risultati

Dopo 500 aggiornamenti (≈ 1 milione di passi nell'ambiente):
- Il robot compie progressi visibili, passando dal dimenarsi casualmente verso un certo movimento in avanti
- Miglioramento costante nella curva di apprendimento
- La convergenza completa a una ricompensa > 300 richiede più addestramento (5-10 milioni di passi)

La curva di apprendimento mostra la caratteristica "curva a S" del controllo continuo:
1. Progressi iniziali lenti (apprendimento della stabilità)
2. Miglioramento rapido (scoperta dell'andatura)
3. Perfezionamento graduale (ottimizzazione dell'andatura)

---

## Punti Chiave

| Concetto | Linguaggio Semplice |
|----------|---------------------|
| **Politica Gaussiana** | Invece di scegliere da un menu, lancia una freccetta su un intervallo di valori |
| **μ (media)** | Dove la politica "mira" |
| **σ (std)** | Quanta casualità / esplorazione usa la politica |
| **log_std come parametro apprendibile** | Un tasso di esplorazione globale aggiornato dall'ottimizzazione basata sul gradiente (salita del gradiente sulla ricompensa, o equivalentemente discesa del gradiente sulla perdita PPO) — proprio come qualsiasi altro peso della rete |
| **Controllo continuo** | Controllo di output a valori reali (coppie, forze, angoli) |

---

## Cosa viene dopo?

PPO ha molti **iperparametri** — impostazioni che scegli prima che inizi l'addestramento (al contrario dei *parametri* come i pesi della rete, che vengono appresi automaticamente). Gli esempi includono `clip_eps`, il tasso di apprendimento (learning rate), il numero di epoche e la dimensione del batch.

Quanto è sensibile PPO a queste scelte? `ppo_hyperparams.py` esegue esperimenti variando sistematicamente ogni iperparametro e mostra l'effetto sulla velocità e sulla stabilità dell'apprendimento.
