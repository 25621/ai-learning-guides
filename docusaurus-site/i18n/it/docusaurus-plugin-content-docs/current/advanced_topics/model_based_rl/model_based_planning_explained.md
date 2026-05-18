# Usare un Modello Appreso per la Pianificazione (MPC) 🔮

## L'Idea di Base {#the-big-idea}

Hai un **modello del mondo** (una rete neurale che predice il futuro). E adesso?

L'uso più diretto è la **pianificazione**: in ogni momento, chiedi al modello "cosa succederebbe se provassi *questo* piano? e *quello*? e *quell'altro*?". Poi scegli il piano che sembra migliore — ma esegui **solo il primissimo passo**.

Poiché il modello non è perfetto, eseguiamo solo un'azione, osserviamo l'effettivo nuovo stato dall'ambiente reale e poi ripianifichiamo da zero.

Questo trucco ha un nome: **Model Predictive Control** (MPC).

---

## Un'Analogia con la Vita Reale {#a-real-life-analogy}

Sei al ristorante e guardi il menu. Non ti impegni subito per un ordine di cinque portate — ordini la prima portata, vedi quanto sei sazio e poi decidi di nuovo per il dolce.

Oppure: stai guidando su una strada tortuosa. Non blocchi i comandi dello sterzo per i prossimi 30 secondi — guardi costantemente avanti, pianifichi per qualche secondo, compi la successiva azione di sterzata e poi ripianifichi.

Quel ciclo **pianifica-lontano / agisci-vicino / ripianifica** è l'MPC.

---

## Come Funziona il "Random Shooting" {#how-random-shooting-works}

Esistono pianificatori più sofisticati — ad esempio:
- **CEM** (Cross-Entropy Method): affina iterativamente una distribuzione sui piani mantenendo solo i migliori (top-k) ad ogni round.
- **MCTS** (Monte Carlo Tree Search): costruisce un albero di ricerca guidato dalle statistiche di simulazione, usato da AlphaGo e MuZero.
- **Pianificatori basati sul gradiente**: differenziano le previsioni del modello rispetto alle azioni e seguono direttamente il gradiente.

Noi usiamo il più semplice tra quelli efficaci: il **random shooting**.

```
Dato lo stato attuale s:
    1. Campiona N=200 sequenze di azioni casuali di lunghezza H=15.
    2. Per ogni sequenza, simula attraverso il modello del mondo partendo da s,
       sommando una ricompensa modellata (shaped reward) ad ogni passo.
       (200 "sogni" in parallelo — veloce!)
    3. Trova la sequenza con la ricompensa totale prevista più alta.
    4. Esegui la PRIMA azione di quella sequenza nell'ambiente reale.
    5. Osserva il reale stato successivo. Scarta il resto del piano.
    6. Vai al punto 1 — ripianifica da zero.
```

200 piani × 15 passi = 3.000 transizioni immaginate per ogni passo reale. Il modello del mondo le esegue tutte in un singolo passaggio batch della rete neurale — tipicamente in pochi millisecondi.

---

## Perché Ripianificare ad Ogni Passo? {#why-re-plan-every-step}

Perché il modello è imperfetto. Gli errori si accumulano durante un rollout (come visto nel grafico generato da `world_model.py`, salvato in `outputs/world_model.png`). Il piano al passo 0 è affidabile solo per le prime mosse; al passo 15 il modello sta avendo delle "allucinazioni". Quindi ci fidiamo solo della **prima mossa**, poi aggiorniamo il piano con l'ultimo stato reale.

Questo è lo stesso motivo per cui gli esseri umani non scrivono un piano di scacchi di 100 mosse per poi seguirlo pedissequamente — le circostanze cambiano e più vai avanti a indovinare, meno la tua previsione corrisponde alla realtà.

---

## Una Difficoltà: La Ricompensa Deve Dire Qualcosa al Pianificatore {#a-wrinkle-the-reward-has-to-tell-the-planner-something}

In CartPole la ricompensa reale è `+1` per ogni passo finché l'asta non cade. Il modello predirebbe fedelmente `+1, +1, +1, ...` per quasi ogni piano, perché i piani casuali raramente terminano rapidamente all'interno del modello — e quindi ogni piano ottiene lo stesso punteggio. Il pianificatore non ha modo di scegliere.

La soluzione: sostituire la ricompensa reale con una **funzione approssimata (smooth proxy)** durante la pianificazione:

```python
reward_proxy(stato) = 1
                    - |angolo_asta| / 0.21          # asta dritta? (1=sì)
                    - 0.1 * |posizione_carrello| / 2.4  # carrello centrato? (1=sì)
```

Ora i piani che *porterebbero* a far cadere l'asta ottengono punteggi visibilmente peggiori rispetto ai piani che la mantengono dritta. Il pianificatore può classificarli.

> **Lezione di vita reale.** Un segnale di ricompensa piatto — "sei sopravvissuto un altro secondo" — è inutile per la pianificazione a breve termine. I segnali densi e modellati (shaped signals) aiutano.

---

## Cosa Fa il Nostro Codice {#what-our-code-does}

`model_based_planning.py`:

1. **Carica** i pesi del modello del mondo salvati da `world_model.py`. (Se mancano, ne addestra uno al volo).
2. **Esegue 20 episodi** di MPC sul CartPole-v1 reale.
3. **Esegue anche 20 episodi** con una politica uniformemente casuale, come baseline.
4. **Traccia** entrambi i risultati fianco a fianco e stampa le medie.

### Cosa dovresti vedere quando lo esegui {#what-you-should-see-when-you-run-it}

| Politica | Ricompensa media (passi sopravvissuti) |
|----------|---------------------------------------:|
| Casuale  | ~22 (tipico per CartPole — l'asta cade subito) |
| MPC (nostro) | ~150–500 (varia in base al seed; molti episodi vicino a 500) |
| Massimo possibile | 500 |

Quel **miglioramento di 5–25 volte** viene ottenuto senza rete della politica, senza funzione valore e senza ulteriore addestramento. Solo un modello del mondo + 200 sogni per passo.

Il grafico `outputs/model_based_planning.png` mostra due barre colorate per episodio — l'MPC è quasi sempre più alto del Casuale, con molti episodi che raggiungono il limite di 500 passi.

---

## Punti di Forza della Pianificazione Basata su Modelli {#strengths-of-model-based-planning}

- **Efficienza dei campioni.** Tutto l'apprendimento è stato fatto da un batch di transizioni casuali. Non è stata necessaria alcuna ulteriore interazione con l'ambiente per derivare una politica utile.
- **Facile da riorientare.** Vuoi controllare l'agente in modo diverso? Cambia la funzione di ricompensa approssimata — non serve riaddestrare. (Prova a massimizzare la velocità del carrello per divertimento).
- **Interpretabile.** Puoi ispezionare i piani considerati dall'agente, le traiettorie previste e i punteggi.

## Debolezze (e Come Vengono Risolte) {#weaknesses-and-what-people-do-about-them}

- **Il random shooting è rudimentale.** Campiona i piani alla cieca. Per dimensioni più elevate, si passa a **CEM** (Cross-Entropy Method — vedi sopra) o **iLQR** (Iterative Linear-Quadratic Regulator, un classico metodo di controllo ottimo che approssima il modello come localmente lineare e lo risolve analiticamente) o a un pianificatore completo **basato sul gradiente** che migliora le azioni seguendo i gradienti attraverso un modello differenziabile.
- **Errore cumulativo del modello.** Gli orizzonti lunghi tendono alla deriva. Si usano gli **ensemble probabilistici** (diversi modelli addestrati sugli stessi dati, come in PETS, Chua et al. 2018) in modo che il pianificatore possa notare il disaccordo e penalizzare i piani su cui il modello è incerto.
- **La ricompensa reale è ciò che vogliamo, in ultima analisi.** Il reward shaping aiuta, ma per compiti più complessi si impara una **funzione valore** addestrata *all'interno* del modello del mondo — un critico appreso che stima il ritorno a lungo termine da qualsiasi stato senza richiedere una funzione approssimata scritta a mano. Sia **Dreamer** (che addestra un actor-critic interamente nell'immaginazione latente) che **MuZero** (che accoppia MCTS con una rete del valore appresa) utilizzano questa idea.

---

## Collegamento con i Sistemi Moderni {#how-this-connects-to-modern-systems}

L'esatta ricetta che hai appena eseguito — **dinamiche apprese + pianificazione** — è la spina dorsale di alcuni dei più forti sistemi di RL nella ricerca moderna sull'IA:

- **MuZero** (DeepMind): combina un modello del mondo appreso con la Monte Carlo Tree Search. Ha dominato Go, scacchi, shogi e Atari senza conoscere le regole in anticipo.
- **Dreamer / DreamerV3** (Hafner et al.): addestra una politica *all'interno* di un modello del mondo nello **spazio latente** (il che significa che il modello comprime immagini o stati grezzi in una rappresentazione compatta e astratta prima di predire il futuro). Raggiunge prestazioni allo stato dell'arte su oltre 100 benchmark.
- **PETS / PlaNet / TD-MPC**: sono famiglie di algoritmi che scalano esattamente questa idea a complessi compiti di controllo continuo come la robotica.

Hai costruito — in poche centinaia di righe — il membro più piccolo di quella famiglia.

---

## Parole Chiave {#key-words}

| Termine | Linguaggio Semplice |
|------|---------------|
| **MPC** | Model Predictive Control — pianifica avanti, agisci una volta, ripianifica |
| **Random shooting** | Valuta molti piani casuali, scegli il migliore |
| **Orizzonte (H)** | Quanti passi in avanti guarda il piano |
| **N campioni** | Quanti piani candidati consideriamo per ogni passo |
| **Orizzonte recedente** | Ripianificare ad ogni passo invece di impegnarsi in un unico piano |
| **Reward proxy / shaping** | Una ricompensa surrogata fluida che fornisce al pianificatore un segnale utile da ottimizzare |

---

## Riassunto in Una Sola Frase {#one-sentence-summary}

> **Una volta che hai un modello del mondo, la pianificazione consiste semplicemente nel "sognare cento futuri, scegliere il miglior primo passo e ripetere".**

Questo è l'intero segreto della RL basata su modelli.
