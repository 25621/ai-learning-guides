# PPO: Aggiornamenti della Politica Sicuri e Costanti

## Il Problema con A2C

Immagina di imparare a tenere in equilibrio una scopa sul tuo dito. Dopo settimane di pratica, riesci a tenerla su per 30 secondi!

Ora il tuo allenatore ti dà un consiglio: "Inclina il polso leggermente più a sinistra."

**Buon consiglio → cambiamento attento → equilibrio mantenuto per 30 secondi ✓**

Ma cosa succederebbe se l'allenatore reagisse in modo eccessivo e dicesse: "INCLINA FORTEMENTE A SINISTRA IMMEDIATAMENTE!"
Correggeresti troppo → la scopa cadrebbe → avresti perso settimane di progressi.

Questo è il problema di A2C: **grandi aggiornamenti del gradiente possono distruggere una buona politica**.

**PPO (Proximal Policy Optimization)** è un sistema di sicurezza che impedisce questo.

---

## L'Idea Centrale: Rimani Vicino a Ciò che Funzionava

Il vincolo chiave di PPO:

> **"Non cambiare troppo la politica in un singolo aggiornamento."**

Prima di un aggiornamento, abbiamo la "vecchia" politica π_old.
Dopo l'aggiornamento, abbiamo la "nuova" politica π_new.

PPO misura quanto la politica è cambiata con il **rapporto di probabilità (probability ratio)**:

```
r(θ) = π_new(a|s) / π_old(a|s)
```

- r = 1.0: politica invariata
- r = 1.5: la nuova politica ha il 50% di probabilità in più di compiere quell'azione
- r = 0.5: la nuova politica ha il 50% di probabilità in meno di compiere quell'azione

**Esempio di vita reale:** Sei uno chef che sta aggiustando una ricetta.
- r = 1.0: stessa quantità di sale di prima
- r = 2.0: raddoppia il sale — troppo estremo!
- r = 0.9: 10% di sale in meno — un cambiamento piccolo e sicuro

---

## Il Trucco del Clipping

PPO taglia (clip) il rapporto per farlo rimanere all'interno di [1-ε, 1+ε] (tipicamente ε = 0.2):

```
L_CLIP = E[min(r(θ) · A,  clip(r(θ), 1-ε, 1+ε) · A)]
```

Analizziamolo:

**Caso 1: L'azione era BUONA (A > 0)**

Vogliamo compiere questa azione più spesso (r > 1). Ma limitiamo quanto aumentiamo:
```
se r > 1.2: taglia a 1.2, nessun ulteriore incentivo a spingere oltre
```
Questo ci impedisce di oscillare TROPPO in una direzione.

**Caso 2: L'azione era CATTIVA (A < 0)**

Vogliamo compiere questa azione meno spesso (r < 1). Ma di nuovo, limitiamo:
```
se r < 0.8: taglia a 0.8, nessuna ulteriore penalità per andare oltre
```

**Visualizzazione:**
```
ε = 0.2, quindi la finestra del rapporto sicuro è da 0.8 a 1.2.

Azione BUONA (A > 0): aumenta la probabilità dell'azione, ma smetti di ricompensarla dopo 1.2
rapporto r:       0.6      0.8      1.0      1.2      1.4
incentivo:         ↑        ↑        ↑        ↑        -
significato:   troppo basso  ok    vecchio   max    tagliato

Azione CATTIVA (A < 0): diminuisci la probabilità dell'azione, ma smetti di ricompensarla sotto 0.8
rapporto r:       0.6      0.8      1.0      1.2      1.4
incentivo:         -        ↓        ↓        ↓        ↓
significato:    tagliato   max    vecchio    ok   troppo alto
```

Il segno `-` indica la regione piatta tagliata. In quella regione, rendere il rapporto di probabilità ancora più estremo non migliora l'obiettivo, quindi PPO non ha alcun incentivo extra a spingere oltre.

**Esempio di vita reale:** Il limitatore di velocità di un'auto. Puoi accelerare, ma una volta raggiunti i 120 km/h, il limitatore entra in funzione e non ti permette di andare più veloce. Ti mantiene al sicuro senza impedirti di muoverti.

---

## Perché Questo Previene Aggiornamenti Catastrofici

Un **aggiornamento catastrofico** si verifica quando un grande cambiamento della politica distrugge completamente tutto ciò che l'agente ha imparato — ore di addestramento perse in un singolo passo di gradiente.

Senza clipping: un grande passo di gradiente potrebbe cambiare drasticamente la politica.
Con clipping: il gradiente è zero al di fuori di [1-ε, 1+ε], quindi la politica può muoversi solo un po' per ogni passo.

**Esempio di vita reale:** Un bravo chirurgo esegue tagli piccoli e precisi — non grandi e ampi. PPO è il "chirurgo attento" degli ottimizzatori per RL.

---

## GAE: Stime del Vantaggio Più Intelligenti

PPO utilizza la **Generalized Advantage Estimation (GAE)** per calcolare il vantaggio:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)          (errore TD)
A_t = δ_t + γλ · δ_{t+1} + (γλ)² · δ_{t+2} + ...
```

GAE ha un parametro λ (lambda):
- λ = 0: usa solo l'errore TD a un passo (bassa varianza, alto bias)
- λ = 1: usa i ritorni Monte Carlo completi (alta varianza, basso bias)
- λ = 0.95: un buon equilibrio tra i due!

**Esempio di vita reale:** Pianificare un viaggio in auto.
- λ=0: guarda solo le prossime 5 miglia (sicuro, ma potrebbe perdere una scorciatoia più avanti)
- λ=1: considera l'intero viaggio di 500 miglia (più informazioni, ma molta incertezza)
- λ=0.95: guarda lontano ma dà più peso alle strade vicine ← l'equilibrio migliore!

---

## Epoche Multiple: Riutilizzare i Dati in Modo Efficiente

Dopo aver raccolto un batch di esperienza (rollout), REINFORCE lo scarta dopo UN singolo aggiornamento.

PPO riutilizza ogni batch per **K epoche** (tipicamente 4-10 passaggi attraverso gli stessi dati):

```
Raccogli 512 passi × 4 ambienti = 2048 transizioni
Epoca 1: 32 minibatch × aggiorna ciascuno
Epoca 2: mescola, altri 32 minibatch × aggiorna ciascuno
Epoca 3: ...
Epoca 4: ...
```

**Cos'è un "minibatch"?** Aggiornare con tutte le 2048 transizioni contemporaneamente è lento e richiede molta memoria; aggiornare una transizione alla volta introduce rumore. Un **minibatch** è un piccolo blocco intermedio — qui, 2048 ÷ 32 = **64 transizioni per minibatch**. Calcoliamo un passo di gradiente per minibatch, quindi ogni epoca esegue 32 aggiornamenti piccoli e stabili invece di 1 enorme. (Questa è la stessa idea di minibatch usata ovunque nel deep learning — vedi [discesa del gradiente stocastica a minibatch](https://it.wikipedia.org/wiki/Discesa_del_gradiente_stocastica)).

Il clipping assicura che questi passaggi multipli non superino il limite — senza clipping, più epoche distruggerebbero la politica spingendola troppo oltre!

**Esempio di vita reale:** Uno studente ha 30 problemi di pratica.
- REINFORCE: fa ogni problema una volta, impara un po', li butta via.
- PPO: fa ogni problema 4 volte (angolazioni diverse ogni volta), limita i cambiamenti così da non memorizzare schemi errati.

---

## La Perdita (Loss) Completa di PPO

```
L = L_CLIP - c₁ · L_entropy + c₂ · L_critic

L_CLIP    = gradiente della politica tagliato
L_entropy = bonus di entropia (incoraggia l'esplorazione)  
L_critic  = MSE tra V(s) e i ritorni
```

Coefficienti tipici: c₁ = 0.01 (entropia), c₂ = 0.5 (critico)

**Due termini che vale la pena approfondire:**

- **Gradiente della politica** — la metà "attore" (actor) della perdita. Utilizza il segnale del gradiente per spingere la politica verso azioni con un vantaggio maggiore e lontano da azioni con un vantaggio minore. Questa è la stessa idea centrale introdotta in REINFORCE — vedi la [spiegazione di REINFORCE](./reinforce_cartpole_explained.md#the-old-way-vs-the-new-way) per l'intuizione. PPO aggiunge semplicemente l'involucro del clipping attorno ad essa.
- **MSE (Errore Quadratico Medio)** — la metà "critico" (critic) della perdita. Il critico V(s) predice il ritorno atteso da uno stato; confrontiamo la sua previsione con il ritorno effettivo e facciamo il quadrato della differenza: `MSE = mean((V(s) - ritorno)²)`. Elevare al quadrato punisce gli errori grandi più di quelli piccoli e fornisce un segnale fluido e differenziabile per l'addestramento. (Perdita di regressione standard — vedi [errore quadratico medio](https://it.wikipedia.org/wiki/Errore_quadratico_medio)).

---

## I Risultati

```
Aggiornamento 200 | Ricompensa media: ~120
Aggiornamento 400 | Ricompensa media: ~280
Aggiornamento 800 | Ricompensa media: ~280-300
```

PPO su CartPole mostra un miglioramento costante ma tende a stabilizzarsi (plateau) intorno a 280-300.
(Un **plateau** significa che la curva di apprendimento si appiattisce — la ricompensa smette di migliorare anche se l'addestramento continua. La politica ha trovato una strategia localmente buona ma non sta facendo ulteriori progressi).
In realtà questo è previsto — PPO è progettato per ambienti più difficili con episodi più lunghi.

Un'osservazione interessante: **REINFORCE ha risolto CartPole più velocemente!** (media 500 vs media 300)

Perché? Gli episodi di CartPole sono brevi (≤500 passi), quindi i ritorni esatti di REINFORCE sono molto accurati. Le stime bootstrapped di PPO aggiungono una complessità non necessaria. PPO brilla davvero in ambienti in cui aspettare episodi completi è impraticabile (come BipedalWalker).

**Cos'è "BipedalWalker"?** BipedalWalker (nello specifico `BipedalWalker-v3` in [Gymnasium](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)) è un classico ambiente di benchmark per RL: un robot a 2 gambe che deve imparare a camminare in avanti su un terreno irregolare senza cadere. A differenza delle due azioni discrete di CartPole (SINISTRA / DESTRA), BipedalWalker ha azioni **continue** — quattro valori di coppia, uno per ogni giunto della gamba, ognuno dei quali è un numero reale in [-1, 1]. Gli episodi possono durare migliaia di passi, che è esattamente il regime in cui l'efficienza dei dati e la stabilità di PPO danno i loro frutti.

---

## Equazioni Chiave

```
Rapporto:       r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
Perdita Clip:   L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
GAE:            A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

---

## Concetti Chiave

| Concetto | Linguaggio Semplice |
|---------|---------------|
| **Rapporto r(θ)** | Quanto è cambiata la politica su questa azione |
| **Clip ε** | Il confine di sicurezza — non cambiare la politica più di questo |
| **GAE** | Un modo intelligente per stimare i vantaggi guardando avanti per più passi |
| **Efficienza dei dati** | Ogni rollout viene raccolto da diversi ambienti paralleli (esperienza decorrelata e stabile) e poi riutilizzato per K epoche di aggiornamenti minibatch — il clipping mantiene sicuri questi passaggi ripetuti |

---

## Prossimi Passi

Finora, tutti i nostri ambienti hanno avuto azioni **discrete** (spingi a SINISTRA o a DESTRA).

I robot reali devono controllare azioni **continue** — come "applica esattamente 0.73 Newton di forza."

`ppo_continuous.py` estende PPO alle azioni continue utilizzando una **politica Gaussiana**, e lo testa sull'ambiente BipedalWalker-v3, molto più difficile!
