# Architettura Option-Critic

## La grande idea: lavorare per capitoli, non parola per parola

Immagina di scrivere un romanzo. Non pianifichi ogni singola parola prima di iniziare. Invece, pensi per **capitoli**: "Il capitolo 1 introduce l'eroe. Il capitolo 2 è la ricerca. Il capitolo 3 è lo scontro finale." All'interno di ogni capitolo, definisci i dettagli man mano che procedi.

Questo è esattamente il modo in cui l'architettura Option-Critic pensa alle decisioni.

---

## Cos'è un agente "piatto"?

Un normale agente RL (come quelli delle Fasi 3 e 4 del corso) decide un'azione alla volta, a ogni singolo passo. È come un GPS che ricalcola l'intero percorso da zero ogni volta che ti sposti di un metro. Funziona, ma è faticoso e lento da apprendere.

---

## Cos'è un'"opzione"?

Un'**opzione (option)** è una **competenza con un nome** — una mini-politica che l'agente può eseguire per diversi passaggi di fila prima di restituire il controllo.

Pensala come un manager che delega a degli specialisti:

| Chi | Cosa fa |
|-----|-------------|
| **Manager (meta-politica)** | Decide *quale* specialista inviare per un lavoro |
| **Specialista A** | Esperto nel navigare nella stanza in alto a sinistra |
| **Specialista B** | Esperto nell'attraversare le porte |
| **Specialista C** | Esperto nel dirigersi velocemente verso l'obiettivo |
| **Specialista D** | Generalista di supporto |

Il manager sceglie uno specialista. Lo specialista lavora finché non decide di aver finito (questo si chiama **terminazione**). Poi il manager sceglie di nuovo.

---

## Le tre parti in movimento

Ogni opzione ha tre componenti — pensale come la **descrizione del lavoro** dello specialista:

1. **Iniziazione (Initiation)**: Quando può essere chiamato questo specialista? *(es. "Lo specialista A si attiva solo vicino alla stanza in alto a sinistra.")*
2. **Politica intra-opzione (Intra-option policy)**: Cosa fa lo specialista mentre lavora? *(es. "Cammina verso l'angolo in alto a sinistra.")*
3. **Terminazione (Termination)**: Quando lo specialista restituisce il controllo? *(es. "Fermati quando hai raggiunto una porta.")*

Il bello di Option-Critic è che tutti e tre questi elementi vengono **appresi automaticamente** — non sei tu a creare manualmente gli specialisti. L'algoritmo capisce da solo che è utile avere un'opzione per ogni stanza, o una per correre verso l'obiettivo.

---

## Una giornata tipo di un agente Option-Critic

1. L'agente entra in una nuova stanza (stato).
2. Il **Manager** osserva la stanza e sceglie un'opzione — ad esempio, l'Opzione 2.
3. Lo **specialista dell'Opzione 2** prende il controllo: cammina verso la porta, passo dopo passo.
4. A un certo punto, l'Opzione 2 dice "Ho finito qui" (terminazione).
5. Il **Manager** si risveglia e sceglie una nuova opzione per la nuova situazione.
6. Il ciclo si ripete.

Confrontalo con l'agente piatto: l'agente piatto si tormenta per ogni singolo passo. Option-Critic delega interi tratti di comportamento, permettendo a ogni specialista di diventare bravo nel suo compito specifico.

---

## Perché è d'aiuto?

In un labirinto, l'agente deve raggiungere un obiettivo che può trovarsi a 30–50 passi di distanza. Con l'apprendimento piatto, ogni passo sul percorso è ugualmente "invisibile" finché la ricompensa non arriva finalmente alla fine — quel segnale deve viaggiare all'indietro per dozzine di passi.

Con le opzioni, il percorso si divide in **sotto-compiti**. Ogni sotto-compito riceve il proprio mini-segnale di ricompensa (raggiungere la porta, entrare nella stanza successiva). L'apprendimento si propaga attraverso segmenti più brevi. **L'agente impara più velocemente su problemi che richiedono molti passaggi.**

Questa è l'idea centrale dietro a tutto l'RL gerarchico — e l'architettura Option-Critic è una delle sue implementazioni più eleganti.

---

## Cosa fa il nostro codice

Lo script `option_critic.py` inserisce un agente Option-Critic in un **mondo a griglia 7x7** con un obiettivo fisso. L'agente inizia in un punto qualsiasi della griglia e deve navigare fino alla cella obiettivo.

L'agente ha quattro opzioni e deve imparare simultaneamente:

- Una politica per ogni opzione (dove camminare)
- Quando terminare ogni opzione (condizione di terminazione)
- Una meta-politica per scegliere tra le opzioni

La ricompensa utilizza la **modellazione basata sul potenziale (potential-based shaping)** — l'agente riceve un piccolo bonus a ogni passo in cui si avvicina all'obiettivo, oltre al +1 per averlo raggiunto. Questo feedback denso rende l'apprendimento abbastanza stabile da vedere le opzioni in funzione entro 2.500 episodi.

Nessun essere umano dice mai all'agente cosa debba fare ogni opzione. L'algoritmo scopre in quali aree della griglia ogni opzione si specializza.

---

## Cosa mostrano i grafici

![Curve di apprendimento Option-Critic](outputs/option_critic.png)

**Sinistra — Ritorno con modellazione (Shaped Return):** Un ritorno più alto significa che l'agente sta raggiungendo l'obiettivo in modo più affidabile *e* percorrendo percorsi più brevi (la modellazione fornisce un bonus per ogni passo di avvicinamento). La curva che sale e poi si stabilizza mostra le opzioni che imparano a coordinarsi.

**Destra — Passi verso l'obiettivo:** Meno passi significano che l'agente ha trovato un percorso più efficiente. La tendenza al ribasso mostra le opzioni che maturano in competenze coerenti che guidano l'agente più direttamente verso l'obiettivo.

Le curve smussate mostrano la tendenza generale su finestre di 100 episodi — un po' di rumore è normale nell'RL, specialmente quando più componenti (opzioni, terminazione, meta-politica) stanno imparando contemporaneamente.

---

## Riassunto in una frase

> **Option-Critic insegna a un agente a lavorare per competenze piuttosto che per singoli passi — un manager sceglie quale specialista attivare, ogni specialista fa il suo lavoro e l'intero sistema impara insieme dallo stesso segnale di ricompensa.**
