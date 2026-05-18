# Double DQN: Risolvere il Problema dell'Eccesso di Fiducia 🤔

## Il Problema: DQN Pensa di Essere Migliore di quanto Sia in Realtà

Immagina che ti venga chiesto: "Qual è il miglior ristorante in città?"

Potresti rispondere: "Pizza Palace è incredibile — è sicuramente da 10/10!" Ma ci sei stato solo
due volte. Non sai davvero se sia *veramente* da 10/10. Potresti sopravvalutarlo perché sei stato
fortunato a trovare dell'ottima pizza in quelle due visite.

Lo stesso problema accade con il DQN: l'agente **sopravvaluta i valori Q**.

---

## Perché il DQN Sopravvaluta?

Quando il DQN calcola il target (l'obiettivo), esegue:
> target = reward + γ × **max** Q(next_state)

Il `max` è il problema! Quando scegli il massimo tra diverse stime "rumorose",
scegli quasi sempre quella con l'errore casuale più grande (bias verso l'alto).

**Esempio di vita reale:** Chiedi a 5 amici di indovinare l'altezza di un edificio. Le loro
stime sono: 40m, 38m, 45m (un tiro fortunato!), 39m, 41m. L'altezza reale è 40m.
Se usi `max(stime)` = 45m, sei fuori strada! Il massimo di stime imprecise
è quasi sempre una sopravvalutazione.

Dopo migliaia di aggiornamenti, il DQN continua ad addestrarsi verso questi target gonfiati,
imparando che le cose vanno meglio di quanto siano realmente. Questo può rallentare l'apprendimento
o spingere l'agente a prendere decisioni sbagliate dettate dall'eccesso di fiducia.

---

## La Soluzione: Double DQN

Il **Double DQN** (Hasselt et al., 2016) divide il `max` in due passaggi:

**Passaggio 1 — Quale azione?** Usa la **rete online** per scegliere l'azione migliore:
> best_action = argmax Q_online(next_state)

**Passaggio 2 — Qual è il suo valore?** Usa la **rete target** per valutare quell'azione:
> target = reward + γ × Q_target(next_state, best_action)

```
DQN Classico:  target = r + γ × max_a Q_target(s', a)
                                 ↑ la stessa rete sceglie E valuta → bias

Double DQN:    best_a = argmax_a Q_online(s', a)      ← la rete online sceglie
               target = r + γ × Q_target(s', best_a)  ← la rete target valuta
                                 ↑ reti diverse → meno bias
```

**Esempio di vita reale:** In un colloquio di lavoro, non permetti al candidato di dare il voto
al proprio test (questo è il problema del DQN classico!). Invece, il candidato
*indica* il suo lavoro migliore e un esaminatore *separato* lo valuta.
Due persone diverse = valutazione più equa!

---

## Perché la Separazione Aiuta?

Le due reti (online e target) hanno pesi diversi perché la rete target viene
aggiornata meno frequentemente. Hanno "opinioni" diverse su quale sia l'azione migliore.

Quando non sono d'accordo:
- La rete online dice: "L'azione A sembra fantastica!"
- La rete target dice: "In realtà, l'azione A è solo discreta — vale circa 7, non 10"

Usando la stima del VALORE della rete target per l'azione SCELTA dalla rete online,
otteniamo un numero più onesto e meno gonfiato.

---

## Differenza nel Codice: Solo una Riga!

L'unico cambiamento nel codice dal DQN classico al Double DQN riguarda il calcolo del target:

```python
# DQN Classico:
q_next = target_net(s_next).max(dim=1).values

# Double DQN:
best_actions = online_net(s_next).argmax(dim=1, keepdim=True)   # scelta con online
q_next = target_net(s_next).gather(1, best_actions)              # valutazione con target
```

Cambiano solo due righe — ma l'impatto sulla stabilità e sull'accuratezza è significativo!

---

## Cosa Mostra il Confronto

Quando esegui `double_dqn_cartpole.py`, vedrai due grafici:

**Grafico 1: Curve di Apprendimento**
- Sia il DQN classico che il Double DQN dovrebbero risolvere CartPole.
- Il Double DQN spesso converge più velocemente e in modo più stabile.
- CartPole è abbastanza semplice che la differenza è modesta; è più evidente su Atari.

**Grafico 2: Stime dei Valori Q**
- DQN Classico: i valori Q tendono a salire nel tempo (sopravvalutazione).
- Double DQN: i valori Q rimangono più modesti e accurati.

Il grafico della sopravvalutazione dei valori Q è l'intuizione chiave — mostra come il DQN classico impari
valori gonfiati che alla fine danneggiano le prestazioni.

---

## Quanto è Migliore il Double DQN?

| Metrica | DQN Classico | Double DQN |
|---------|--------------|------------|
| Accuratezza valore Q | Sopravvaluta | Più accurato |
| Stabilità apprendimento | Più varianza | Meno varianza |
| Prestazioni CartPole | Buone | Leggermente migliori |
| Prestazioni Atari (50 giochi) | Base | +2.6× più giochi vicini al livello umano |

Sui giochi Atari complessi, il Double DQN ha fatto una differenza molto più grande rispetto a CartPole
(perché Atari ha stime dei valori Q molto più rumorose).

---

## La Famiglia dei Miglioramenti DQN

Il Double DQN è solo uno dei molti miglioramenti al DQN classico. Il paper "Rainbow"
(2017) ha combinato 6 miglioramenti:

1. **Double DQN** (corregge la sopravvalutazione) ← questo script!
2. **Prioritized Replay** (impara di più dalle esperienze sorprendenti)
3. **Dueling Networks** (separa "quanto è buono questo stato?" da "qual è l'azione migliore?")
4. **Multi-step returns** (guarda più avanti nel futuro)
5. **Distributional RL** (impara l'intera distribuzione dei ritorni, non solo la media)
6. **NoisyNets** (esplorazione appresa invece di [ε-greedy](../foundations/multi_armed_bandit_explained.md#the-epsilon-greedy-strategy))

Rainbow li ha combinati TUTTI e ha ottenuto le migliori prestazioni su Atari del suo tempo!

---

## Vocabolario Chiave

| Parola | Significato |
|--------|-------------|
| **Sopravvalutazione (Overestimation)** | I valori Q sono più alti dei valori reali (troppo ottimistici) |
| **Double DQN** | Usa la rete online per la selezione dell'azione e la rete target per la valutazione |
| **Disaccoppiamento (Decoupling)** | Separare due compiti che venivano eseguiti dalla stessa rete |
| **Bias** | Un errore sistematico in una direzione (sempre troppo alto o sempre troppo basso) |
| **Rainbow** | Una variante del DQN che combina 6 miglioramenti per le massime prestazioni |

---

## Riepilogo: Il Viaggio della Fase 3

Hai ora completato l'intera progressione della Fase 3:

| Algoritmo | Cosa aggiunge | Perché aiuta |
|-----------|---------------|--------------|
| Linear Q | Rete neurale → formula semplice | Gestisce stati continui |
| DQN Naive | Rete neurale completa | Impara pattern complessi |
| + Replay buffer | Campionamento casuale della memoria | Rompe le correlazioni |
| + Target network | Copia congelata per i target | Stabilizza il "bersaglio" |
| Atari DQN | CNN + frame stacking | Impara dai pixel! |
| Double DQN | Selezione/valutazione separate | Riduce la sopravvalutazione |

Ogni passaggio ha risolto un problema specifico. È così che funziona la vera ricerca — un miglioramento
attento alla volta!
