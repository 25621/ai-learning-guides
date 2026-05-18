# Rete Bersaglio (Target Network): Stabilizzare il Centro 🎯

## Il Problema dell'Obiettivo Mobile

Immagina di cercare di colpire il centro di un bersaglio con arco e freccia. Scocchi, guardi dove è atterrata la freccia e aggiusti la mira per la volta successiva. Semplice, vero?

Ora immagina che il bersaglio si MUOVA ogni volta che scocchi! Ogni freccia che lanci cambia leggermente la posizione in cui si troverà il bersaglio per il tiro successivo. Non miglioreresti mai — inseguiresti un obiettivo che scappa sempre.

Questo è esattamente il problema di DQN senza una rete bersaglio (target network)!

---

## Perché i Q-Target continuano a muoversi

In DQN, il bersaglio (target) per ogni aggiornamento è:
> target = ricompensa + γ × max(Q(stato_successivo))

Qui **γ (gamma)** è il **fattore di sconto** — un numero compreso tra 0 e 1 (tipicamente 0.99) che controlla quanto l'agente si preoccupi delle ricompense *future* rispetto a quelle *immediate*.

**Esempio di vita reale:** Immagina che qualcuno ti offra un biscotto ora, o due biscotti domani. Se vuoi davvero i biscotti ora, il tuo γ è basso (sconti pesantemente il futuro). Se sei paziente e felice di aspettare, il tuo γ è alto (le ricompense future contano quasi quanto quelle attuali). In RL, γ = 0.99 significa "una ricompensa al prossimo passo vale il 99% di una ricompensa proprio ora".

I valori Q sul lato destro derivano da... la stessa rete che stiamo addestrando!

Quindi ogni volta che aggiorniamo la rete (per migliorare i valori Q), cambiamo anche i bersagli. È un ciclo di feedback:

1. Aggiorna la rete → i valori Q cambiano
2. I valori Q cambiano → i bersagli cambiano
3. I bersagli cambiano → aggiorna la rete in modo diverso
4. Ripeti all'infinito — instabile!

**Esempio di vita reale:** Cercare di pesarsi su una bilancia che cambia le sue letture ogni volta che ci sali sopra. Non sapresti mai il tuo vero peso!

---

## La Soluzione: Congela il Bersaglio! ❄️

La **Rete Bersaglio** (Target Network) è una COPIA della rete Q principale che viene congelata.

- **Rete online** (`qnet`): Aggiornata ad ogni passo dell'addestramento — impara velocemente
- **Rete bersaglio** (`target_net`): Copia congelata — aggiornata solo ogni 100 passi

Usiamo la rete bersaglio CONGELATA per calcolare i bersagli:
> target = ricompensa + γ × max(Q_TARGET(stato_successivo))

Il bersaglio rimane fermo per 100 passi! Ciò fornisce alla rete online un obiettivo stabile a cui mirare. Quindi copiamo i pesi online nella rete bersaglio, congeliamo di nuovo e ripetiamo.

**Esempio di vita reale:** Pensa a uno studente e a un insegnante. L'insegnante assegna i compiti (il bersaglio). Lo studente impara e migliora. Dopo 100 lezioni, l'insegnante AGGIORNA i compiti per renderli più difficili. L'insegnante non cambia ogni singolo minuto — sarebbe troppo caotico!

---

## La Ricetta Completa di DQN 🍕

L'algoritmo DQN completo (experience replay + target network) è:

```
1. Inizializza la rete online Q e la rete bersaglio Q_target (stessi pesi)
2. Crea il buffer di replay (scatola della memoria)

Ogni passo nell'ambiente:
  a. Scegli l'azione usando ε-greedy con Q
  b. Memorizza (stato, azione, ricompensa, stato_successivo) nel buffer

Ogni 4 passi:
  c. Campiona un mini-batch casuale dal buffer
  d. Calcola i bersagli usando Q_TARGET (congelata!)
  e. Aggiorna Q per minimizzare la perdita

Ogni 100 passi:
  f. Copia i pesi di Q → Q_TARGET (sincronizza il bersaglio)
```

Questo è l'esatto algoritmo dell'articolo DQN di DeepMind (2015)!

---

## Cosa mostra il confronto

Quando esegui `dqn_target_network.py`, vedrai:

**Senza rete bersaglio (solo DQN + replay):**
- L'addestramento potrebbe essere "accettabile" ma con crolli periodici
- I valori Q possono divergere (esplodere o oscillare)
- L'apprendimento è meno prevedibile

**DQN completo (replay + rete bersaglio):**
- Apprendimento verso l'alto più costante
- I valori Q rimangono in un intervallo ragionevole
- Convergenza più rapida alla soglia di risoluzione (195+ su CartPole)

---

## La "Triade Mortale" ☠️

Nell'apprendimento per rinforzo, la combinazione di tre elementi crea instabilità:

1. **Approssimazione della funzione** (rete neurale invece di una tabella) ← la usiamo
2. **Bootstrapping** (usare valori Q per stimare valori Q) ← lo usiamo
3. **Apprendimento off-policy** (Q-learning usa il max, non la politica effettiva) ← lo usiamo

Tutti e tre insieme = la "triade mortale". DQN la doma con:
- Experience replay → rompe le correlazioni
- Rete bersaglio → rompe il ciclo di feedback

Non risolve completamente il problema, ma lo rende gestibile!

---

## Vocabolario Chiave

| Parola | Significato |
|--------|---------|
| **Rete Bersaglio (Target Network)** | Una copia congelata della rete Q usata solo per calcolare i bersagli |
| **Rete Online** | La rete Q che viene addestrata attivamente |
| **Sincronizzazione (Sync)** | Copiare i pesi della rete online nella rete bersaglio |
| **Ciclo di Feedback** | Quando l'output di un sistema torna indietro per cambiare l'input (può causare instabilità) |
| **Triade Mortale** | La combinazione di approssimazione della funzione + bootstrapping + off-policy che causa instabilità |

---

## Impatto nel Mondo Reale

Nel 2015, DeepMind pubblicò il suo articolo su DQN mostrando un'IA in grado di giocare a 49 giochi Atari a livello sovrumano — usando SOLO questi due trucchi (replay + rete bersaglio).

Prima di questo, si pensava che non fosse possibile addestrare reti neurali con RL a causa dell'instabilità. DeepMind ha dimostrato il contrario, dando il via alla rivoluzione del deep RL!

Successivamente, applicheremo questa ricetta DQN completa ad Atari Pong — un vero videogioco con pixel grezzi come input!
