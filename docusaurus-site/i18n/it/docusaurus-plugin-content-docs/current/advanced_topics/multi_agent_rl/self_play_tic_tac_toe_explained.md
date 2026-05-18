# Self-Play: Insegnare a un Agente Facendolo Giocare contro se Stesso ♟️

## Cos'è il Self-Play?

Immagina un bambino che vuole diventare bravissimo a scacchi ma non ha nessuno con cui giocare. Quindi gioca contro se stesso. Mano sinistra contro mano destra. In ogni partita, *entrambe* le parti cercano di vincere. In ogni partita, *entrambe* le parti imparano cosa ha funzionato.

Questo è il **self-play**: un singolo agente agisce come entrambi i giocatori e ogni mossa diventa una lezione per chiunque debba muovere successivamente. Nessun insegnante, nessun avversario esperto. Solo un allievo che è anche la sua stessa scala per salire.

Il self-play sembra un trucco — sicuramente serve un vero avversario? — ma è il motore dietro i traguardi più famosi della RL dell'ultimo decennio: **AlphaGo Zero**, **AlphaZero**, **MuZero**, **OpenAI Five**. Tutti usano il self-play. Il motivo è semplice: man mano che migliori, il tuo avversario migliora della stessa misura. La sfida corrisponde sempre alle tue abilità.

---

## Perché Funziona

Tre cose rendono speciale il self-play:

1. **Avversari infiniti.** Non finisci mai le partite. L'avversario è sempre presente e gratuito.
2. **Curriculum che cresce con te.** Un principiante può giocare solo contro altri principianti. Man mano che migliori, migliora anche la tua ombra — automaticamente.
3. **Simmetria.** In un gioco a somma zero (il guadagno di un giocatore è la perdita dell'altro), un unico insieme di valori Q descrive entrambe le parti; devi solo invertire il segno quando tocca all'altro giocatore. Quindi una *singola* tabella Q può insegnare a se stessa.

Il tris (Tic-tac-toe) è il banco di prova perfetto: abbastanza piccolo da stare in un dizionario, ma abbastanza complesso da far sì che scegliere mosse a caso porti quasi sempre alla sconfitta contro un giocatore strategico.

---

## Analogia con la Vita Reale

- **Praticare tennis contro un muro.** Non puoi perdere contro un muro, ma puoi esercitarti nei servizi. Il self-play è farlo da entrambi i lati — tu sei il muro *e* il giocatore, e passi continuamente da un ruolo all'altro.
- **Un club di dibattito che sostiene entrambe le parti.** I debuttanti migliori emergono difendendo sempre la visione opposta a quella in cui credono personalmente. Ogni argomento allena sia l'attacco che la difesa.
- **AlphaGo Zero.** Ha imparato partendo da zero partite umane. Iniziando con mosse casuali, ha giocato milioni di partite contro se stesso; in pochi giorni è diventato più forte di ogni precedente programma di Go, incluso quello che aveva battuto Lee Sedol.

---

## Cosa Fa il Nostro Codice

Impariamo una singola tabella Q per il *giocatore di turno*:

```
Q[(scacchiera, giocatore_di_turno)][azione] = ritorno atteso per quel giocatore
```

Il ciclo di addestramento è:

1. Inizia con una scacchiera vuota. `giocatore = X`.
2. Entrambi i giocatori agiscono con lo **stesso agente**, usando ε-greedy.
3. Dopo ogni partita, torna indietro attraverso ogni tripla (scacchiera, giocatore, azione) nella storia e applica l'aggiornamento del Q-learning.
4. La ricompensa inverte il segno tra i turni: se X vince, ogni mossa fatta da X riceve +1 (o eredita il valore da uno stato vincente futuro); ogni mossa fatta da O riceve -1.
5. Diminuiamo lentamente il nostro tasso di esplorazione (ε) da 0.2 a 0.02, così l'agente si impegna nel suo gioco migliore verso la fine dell'addestramento invece di provare mosse casuali.

Ogni 2.500 episodi valutiamo l'agente contro un **avversario casuale** (congeliamo il processo di apprendimento e facciamo giocare entrambe le parti in modo greedy). L'agente dovrebbe vincere o pareggiare il ~100% di quelle partite dopo abbastanza self-play.

### Cosa dovresti vedere

Dopo 50.000 episodi di self-play:

| Partita | Risultato atteso |
|----------|-----------------|
| Agente addestrato vs Avversario casuale (1000 partite) | **~95-99% vittorie o pareggi**, virtualmente 0% sconfitte |
| Agente addestrato vs Se stesso (200 partite greedy) | **Tutti 200 pareggi**. Il tris è un gioco che finisce sempre in pareggio se entrambi i giocatori giocano perfettamente. Il fatto che il self-play porti al pareggio in ogni partita è un segno di convergenza. |

Il grafico `outputs/self_play_tic_tac_toe.png` mostra le frazioni di vittorie/pareggi/sconfitte dell'agente contro un avversario casuale nel tempo:
- La percentuale di vittorie inizia al ~60% (quando entrambi i giocatori giocano a caso, il primo giocatore ha un vantaggio intrinseco perché può piazzare più simboli sulla scacchiera).
- Sale a >90%.
- La percentuale di sconfitte scende quasi a 0%.

Lo script stampa anche una partita di esempio mossa dopo mossa alla fine, così puoi vedere l'agente giocare.

---

## Attenzione a Queste Sottigliezze

- **L'inversione del segno conta.** Un bug comune: dimenticare che "l'avversario che massimizza il proprio valore" significa *minimizzare il nostro* nel target di bootstrap. L'aggiornamento nel nostro codice usa `target = reward - gamma * max(Q[successivo, avversario])`.
- **La simmetria non è sfruttata qui.** Un'implementazione reale normalizzerebbe le scacchiere (ruotando o riflettendo ogni stato in una "forma normale" standard) per condividere i valori Q tra 8 simmetrie. Noi lo saltiamo — lo spazio degli stati è abbastanza piccolo da poter usare la forza bruta.
- **La tabella Q cresce.** Dopo 50k partite di self-play vedrai alcune migliaia di chiavi stato-giocatore. In questo caso va bene; per gli scacchi o il Go servirebbe invece una rete neurale, motivo per cui **AlphaZero sostituisce la tabella con una CNN + MCTS**.

---

## Dove il Self-Play Fallisce

- **Giochi non a somma zero.** "Entrambe le parti felici" è incompatibile con il gioco simmetrico; non puoi semplicemente invertire un segno.
- **Ruoli asimmetrici.** Se "attaccante" e "difensore" hanno spazi di azione diversi, servono due reti separate.
- **Cicli di strategia.** Il self-play puro può rimanere bloccato in cicli stile sasso-carta-forbici. AlphaStar ha risolto questo problema mantenendo un ampio *pool* (o "lega") di versioni passate dell'agente e scegliendo gli avversari da quel pool a caso, così l'agente impara a battere molti stili diversi invece di quello attuale.
- **Reward hacking.** Il self-play rende entrambe le parti più intelligenti, ma solo nel gioco *per come lo hai definito*. Se il tuo sistema di ricompense ha delle scorciatoie non intenzionali, entrambe le parti le sfrutteranno a vicenda, portando a comportamenti bizzarri invece di padroneggiare il gioco reale.

---

## Parole Chiave da Ricordare

| Parola | Significato |
|------|---------|
| **Self-play**      | Lo stesso agente gioca su entrambi i lati di una partita |
| **A somma zero**   | Guadagno di un giocatore = perdita dell'altro |
| **Simmetria**      | Una tabella Q può servire entrambi i lati se si invertono i segni |
| **Population play**| Self-play con *molte* versioni passate di se stessi come avversari (AlphaStar) |
| **Curriculum**     | Una progressione naturale della difficoltà — il self-play la ottiene gratuitamente |
| **MCTS**           | Monte-Carlo Tree Search — l'algoritmo di pianificazione che AlphaZero accoppia con il self-play |

---

## Riassunto in Una Sola Frase

> **Il self-play trasforma il miglioramento nella sua stessa scala: ogni volta che diventi più bravo, lo diventa anche il tuo avversario — automaticamente.**

Questa idea, scalata con le **reti neurali** e la ricerca ad albero, ha battuto i migliori umani a Go, scacchi, shogi, Dota 2 e StarCraft.
