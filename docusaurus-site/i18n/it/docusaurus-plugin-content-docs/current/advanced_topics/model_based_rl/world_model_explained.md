# Addestrare un Modello del Mondo: Insegnare all'Agente a Sognare 🌍

## Cos'è un "Modello del Mondo"?

Un **modello del mondo** è la *copia interna dell'universo* dell'agente. Forniscigli uno stato e un'azione, e lui predirà cosa accadrà dopo:

```
(stato, azione)  ──►  Rete Neurale  ──►  (stato_successivo, ricompensa)
```

Non è il mondo reale — è un **simulatore che l'agente ha costruito per se stesso** osservando la realtà e imparando a imitarla.

Una volta addestrato, il modello permette all'agente di porsi domande ipotetiche ("what-if") senza compiere alcuna azione reale:

> *"Se spingo a sinistra ora e poi due volte a destra, dove andrò a finire? L'asta cadrà?"*

L'agente può riflettere su cento piani all'interno del suo modello nel tempo che impiegherebbe per compiere una sola mossa reale. Questo è il punto centrale.

---

## Un'Analogia con la Vita Reale

Pensa a come *tu* risolvi un puzzle. Non sposti fisicamente ogni pezzo in ogni fessura. **Immagini** cosa succede se il pezzo A va qui. Se quella simulazione mentale sembra sbagliata, la scarti prima ancora di muovere un dito.

Il tuo cervello ha un modello del mondo appreso — costruito in anni passati a vedere come si comportano gli oggetti — che ti permette di simulare i risultati prima di impegnarti.

Altri esempi:

- **Un giocatore di scacchi** immagina le mosse con diversi turni di anticipo.
- **Un conducente** pensa: "Se freno ora, l'auto dietro ha abbastanza spazio".
- **Un bambino** che impila blocchi: "Se metto quello grande sopra, la torre traballerà". (Ha imparato questo modello facendo cadere torri in precedenza).

In ogni caso, **un modello mentale + l'immaginazione = decisioni migliori con meno rischi**.

---

## Come Costruisce l'Agente il Suo Modello?

Semplicemente **osservando**. Nello specifico:

1. **Raccoglie dati.** Lascia che una politica (anche casuale) interagisca con l'ambiente reale per un po'. Salva ogni transizione:
   ```
   (stato, azione, ricompensa, stato_successivo)
   ```
2. **Addestra una rete neurale** a predire `stato_successivo` e `ricompensa` partendo da `(stato, azione)`. Questo è apprendimento supervisionato: ogni transizione salvata è un esempio etichettato dove l'input è "ciò che l'agente ha visto e fatto" e l'etichetta è "ciò che è realmente accaduto dopo".
3. **Convalida.** Tieni da parte il 10% dei dati e controlla le previsioni del modello rispetto a quelle reali. Un errore basso significa che il modello ha catturato la **dinamica** dell'ambiente: come cambiano gli stati dopo le azioni.

Il trucco che usiamo: invece di predire direttamente lo `stato_successivo`, prediciamo il **delta** `stato_successivo − stato`. La maggior parte della fisica è incrementale ("il carrello si è mosso di un pochino") e i target piccoli sono più facili da gestire per le reti neurali.

---

## La Nostra Configurazione

| Scelta | Valore | Perché |
|--------|-------|-----|
| Ambiente | `CartPole-v1` | Stato 4-D, 2 azioni — facile da modellare |
| Dati | 20.000 transizioni da una politica casuale | Ampia copertura dello spazio degli stati |
| Rete | MLP, 2 × 128 ReLU nascosti | MLP = Multi-Layer Perceptron (rete neurale standard). Due strati nascosti da 128 neuroni. Capacità sufficiente, veloce da addestrare. |
| Perdita (Loss) | MSE su `(delta_stato, ricompensa)` | MSE = Errore Quadratico Medio. Perdita di regressione standard. |
| Ottimizzatore | Adam, lr = 1e-3, 30 epoche | Ottimizzatore adattivo. Standard, non richiede una sintonizzazione speciale. |

L'intero addestramento termina in pochi secondi su CPU.

---

## Che Aspetto Ha un "Buon" Modello?

Contano due tipi di diagnostica:

### 1. Accuratezza del singolo passo (MSE di convalida)

Questo indica "quanto bene il modello predice UN passo nel futuro?". Dopo 30 epoche dovresti vedere un MSE di convalida nell'intervallo **1e-4 a 1e-3**. È un valore piccolissimo — gli angoli dell'asta e le posizioni del carrello sono accurati fino a diverse cifre decimali.

### 2. **Errore cumulativo** su rollout di k-passi

Questo è il *vero* test. Prendi uno stato, passalo al modello, poi prendi la sua previsione e passala di nuovo al modello — per `k` passi di fila. L'errore cresce perché ogni passo aggiunge un po' di rumore sopra la previsione precedente.

```
Passo  1:  errore L2 ≈ 0.01   (quasi perfetto)
Passo  5:  errore L2 ≈ 0.05
Passo 10:  errore L2 ≈ 0.15
Passo 20:  errore L2 ≈ 0.40   (visibile deriva)
```

*(Errore L2 = distanza Euclidea tra lo stato successivo previsto e quello reale — pensa ad esso come a "quanto è lontana la stima del modello nello spazio degli stati 4-D?")*

**Perché questo è importante.** Se pianifichiamo 15 passi avanti con il modello, lo stato *esatto* al passo 15 sarà sbagliato — ma se la classifica relativa dei "piani buoni rispetto ai piani cattivi" viene preservata, la pianificazione funziona ancora. (Questo è ciò che sfrutta `model_based_planning.py`).

Il grafico in `outputs/world_model.png` mostra entrambe le diagnostiche fianco a fianco: la curva della perdita di addestramento scende bene su una scala logaritmica, e la curva dell'errore di rollout sale costantemente.

---

## Perché Predire il *Delta*?

Confronta due modi di porre lo stesso problema alla rete:

| Target | Magnitudine tipica | Facile o difficile? |
|--------|------------------:|--------------|
| `stato_successivo` | 0–2.4 (pos carrello) | La rete deve riprodurre la posizione **e** il piccolo cambiamento |
| `stato_successivo - stato`| ~0.02 | La rete impara solo il piccolo cambiamento |

Predire il delta significa anche che: se la rete emette zeri (come spesso fa una rete non addestrata), la previsione è semplicemente "nulla si è mosso" — un default sensato e sicuro per un singolo passo temporale. Al contrario, predire direttamente lo `stato_successivo` assoluto emetterebbe inizialmente valori casuali senza senso, rendendo l'inizio dell'addestramento molto instabile.

---

## Cosa Ci Permette di Fare

Un modello del mondo addestrato è la base per:

- **Pianificazione** — ricerca su sequenze di azioni immaginate (vedi `model_based_planning.py`).
- **Aumento in stile Dyna** — addestrare una rete Q su dati immaginati per moltiplicare l'efficienza dei campioni.
- **Curiosità / esplorazione** — visitare stati che il modello non riesce a predire bene.
- **Paper Dreamer / World-Models** — addestrare una *politica* interamente all'interno del modello con zero interazioni col mondo reale oltre alla raccolta dati iniziale.

---

## Limiti e Cautele

- **Deriva fuori distribuzione (Out-of-distribution drift).** Il modello conosce solo la parte di mondo che ha visto. Pianifica in modo troppo aggressivo e finirai in regioni che il modello non ha mai visitato — lì le previsioni sono pura fantasia.
- **Errore cumulativo.** La pianificazione su lunghi **orizzonti** (molti passi nel futuro) non è affidabile a causa dell'accumulo di errori, come mostra il grafico. I sistemi moderni affrontano questo problema usando **ensemble probabilistici** (addestrando più modelli e controllando se concordano, come in PETS o Dreamer) in modo che il pianificatore sappia esattamente *quanto è incerto* il modello ad ogni passo e possa evitare percorsi rischiosi e sconosciuti.
- **Ambienti stocastici.** Un regressore deterministico standard predice solo il risultato *medio* e perde completamente la gamma dei risultati possibili. Gli ambienti complessi del mondo reale richiedono modelli probabilistici (come quelli con output Gaussiani, o **modelli stocastici latenti**) per rappresentare accuratamente l'incertezza e la casualità.

---

## Parole Chiave

| Termine | Linguaggio Semplice |
|------|---------------|
| **Modello del mondo** | Una rete neurale che imita l'ambiente |
| **Dinamica** | La funzione `(s, a) → s'` |
| **Modello di ricompensa** | La funzione `(s, a) → r` |
| **Previsione a un passo** | Cosa emette il modello da uno stato reale |
| **Rollout** | Previsioni a un passo ripetute, riutilizzando gli output come input |
| **Errore cumulativo** | Piccoli errori che crescono durante un rollout |

---

## Riassunto in Una Sola Frase

> **Un modello del mondo è una piccola copia neurale dell'universo che l'agente può consultare — e in cui può sognare — prima di rischiare un'azione reale.**

Prossimo: `model_based_planning.py` mette al lavoro questo modello per il processo decisionale effettivo.
