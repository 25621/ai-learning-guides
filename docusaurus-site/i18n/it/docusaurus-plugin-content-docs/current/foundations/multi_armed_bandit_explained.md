# Il Problema del Bandito Multi-braccio 🎰

## Cos'è?

Immagina di essere a una festa di compleanno e ci sono **10 diversi barattoli di caramelle**. Ogni barattolo contiene delle caramelle, ma alcuni barattoli hanno caramelle *buonissime* e altri hanno caramelle *meno buone*. Non sai quale sia il barattolo migliore — devi provarli!

Ogni volta che metti la mano in un barattolo, ottieni una caramella. Il tuo compito è:

> **Ottenere quante più caramelle buonissime possibile!**

Questo è il problema del Bandito Multi-braccio (Multi-Armed Bandit)! Invece di barattoli di caramelle, gli scienziati li chiamano "bracci" (come i bracci di una slot machine). Ogni braccio ti dà un premio, ma i premi sono diversi ogni volta.

---

## La Grande Domanda: Dovrei provare nuovi barattoli o restare con il mio preferito?

Questa è la parte più difficile! Supponiamo che tu abbia provato il Barattolo #3 e fosse piuttosto buono. Ora hai una scelta:

- **Sfruttamento (Exploit)**: Continua a scegliere il Barattolo #3 perché sai già che è buono.
- **Esplorazione (Explore)**: Prova un nuovo barattolo — forse il Barattolo #7 è ancora *meglio*!

Se scegli sempre e solo il primo barattolo che ti è piaciuto, potresti perderti il barattolo con le caramelle super-buone. Ma se provi *sempre* nuovi barattoli, non userai mai quello che hai già imparato!

**Esempio di vita reale:** Pensa al tuo ristorante preferito. Ordini sempre le crocchette di pollo (sfruttamento!), ma forse la pizza è ancora più buona. Se non provi mai nulla di nuovo, non lo saprai mai!

---

## La Strategia Epsilon-Greedy {#the-epsilon-greedy-strategy}

Un modo intelligente per risolvere questo problema è chiamato **epsilon-greedy** (epsilon è semplicemente la lettera greca ε, pronunciata "èp-si-lon"):

1. **La maggior parte delle volte (diciamo il 90%)**: Scegli il barattolo che *pensi* sia il migliore.
2. **A volte (diciamo il 10%)**: Scegli un barattolo *a caso* per esplorare!

Quel 10% di viaggi esplorativi ti aiuta a scoprire barattoli migliori. Il 90% di viaggi di sfruttamento ti permette di usare ciò che hai già imparato.

---

## Cosa ha scoperto il nostro codice

Abbiamo testato 10 bracci (barattoli di caramelle) con 200 bambini diversi, 1.000 scelte ciascuno:

| Strategia | % di tempo in cui è stato scelto il barattolo migliore |
|----------|----------------------------------|
| **Mai esplorare (ε=0)** | 14,5% — è rimasto bloccato all'inizio, non ha mai trovato il migliore! |
| **Esplorare l'1% delle volte (ε=0.01)** | 37,6% — ha trovato lentamente il barattolo migliore |
| **Esplorare il 10% delle volte (ε=0.10)** | **74,2%** — ha imparato velocemente, ha scelto il migliore la maggior parte delle volte! |

**Lezione**: Un po' di esplorazione fa molta strada!

---

## Esempi di Vita Reale

- **Raccomandazioni di Netflix**: Netflix dovrebbe mostrarti un film che probabilmente ti piacerà (sfruttamento) o suggerirti qualcosa di nuovo (esplorazione)?
- **Un medico che sceglie un trattamento**: Usare il trattamento che di solito funziona (sfruttamento) o provarne uno nuovo che potrebbe essere ancora migliore (esplorazione)?
- **Un'ape che cerca i fiori**: Dovrebbe continuare a visitare i fiori che sa avere il nettare, o volare in un nuovo campo?

---

## Parole Chiave da Ricordare

- **Braccio (Arm)**: Una delle scelte (come un barattolo di caramelle).
- **Ricompensa (Reward)**: Ciò che ottieni quando scegli un braccio (come la caramella).
- **Sfruttamento (Exploit)**: Usare ciò che sai già essere buono.
- **Esplorazione (Explore)**: Provare qualcosa di nuovo per imparare di più.
- **Epsilon (ε)**: La probabilità con cui esplori invece di sfruttare.

L'idea principale: **Devi bilanciare il provare cose nuove con l'uso di ciò che già conosci!**
