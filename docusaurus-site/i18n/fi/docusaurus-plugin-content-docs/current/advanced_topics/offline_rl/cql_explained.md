# Konservatiivinen Q-Learning (CQL) 🛡️

## Mikä se on?

Kuvittele, että opit sijoittamaan rahaa lukemalla jättiläiskirjaa menneisyydestä
muiden ihmisten tekemät osakekaupat. Kirjasto ostaa, myy ja pitää -
mutta **ei kirjaa mistään kaupasta, jota kukaan ei todellisuudessa tehnyt**.

Kuvittele nyt, että liian itsevarma opiskelija katsoo kirjanpitoa ja sanoo:
*"Entä jos joku olisi ostanut arpajaiset joka maanantai? Se olisi ostanut
oli upea kauppa!"*

Ongelma: **kirjassa ei ole tietoja maanantai-arpajaisista**, joten
opiskelija vain hallusinoi. Silti tuo hallusinoitu kauppa näyttää hyvältä
paperia, joten opiskelijan "politiikka" haluaa jatkuvasti tehdä sen.

Tuo hallusinaatio-ongelma on **jakauman muutos**: offline-oppija
rakastaa toimintoja, joita tietojoukko ei ole koskaan testannut, koska siinä ei ole tietoja
ristiriidassa optimismin kanssa. CQL on parannuskeino.

---

## Kuinka Q-Learning menee pieleen offline-tilassa

Normaali Q-learningin tavoite on:

```
target(s, a) = r + γ · max_{a'} Q(s', a')
```

Tuo `max_{a'}` on vaara. Kun tietojoukko ei koskaan tallentanut toimintaa `a'`
tilassa `s'`, verkko vain *arvaa* Q-arvon — ja hermoverkot
yleensä **yliarvioida** Q näkymättömille tuloille. Kohde perii
yliarvioida, verkko oppii ennustamaan, että suurempi luku, ja edelleen
Seuraavassa vaiheessa **ekstrapoloimme** (projektoimme vielä pidemmälle
data tukee) vielä korkeampi. Politiikka jahtaa haamua.

Jos voisit jatkaa tietojen keräämistä, tämä korjaisi itsensä (
haamutoiminta osoittautuu todellisuudessa huonoksi). Mutta **offline RL sinua
ei voi kerätä lisää tietoja.** Haamu on ikuinen.

---

## CQL:n temppu

CQL (Kumar et al., 2020) lisää tappioon sakon:

```
cql_loss(s)  =  log Σ_a exp Q(s, a)   -   Q(s, a_dataset)
```

Kaksi kappaletta:

1. **`log Σ_a exp Q(s, a)`** (lue: *"log-sum-exp kaikissa toimissa"*) on a
   **pehmeä maksimi** kaikissa toimissa – tasainen, erottuva
   likiarvo `max` joka harkitsee jokaista toimintoa kerralla eikä
   vaikea valita yksi voittaja. Sen rankaiseminen pienentää Q-arvoja
   **yleisesti** (työntää kaikkia ennusteita
   alas tasaisesti) — erityisesti toimille, joilla on *korkein* Q, mikä on täsmälleen
   missä hallusinaatiot elävät.
2. **`- Q(s, a_dataset)`** palkitsee korkean Q:n tietojoukon toiminnasta
   todella tallennettu — suojaa jakelun Q-arvoja
   kutistuminen yllä.

Nettovaikutus: **Q vedetään alas näkymättömissä toimissa, vedetään ylös nähdyissä toimissa
toiminnot.** Opitusta Q:sta tulee todellisen Q:n *alaraja*
**`argmax`** käytäntö (sääntö, joka yksinkertaisesti valitsee toiminnon, jolla on korkein Q)
lopettaa haamujen jahtaamisen.

Täysi tappio:

```
L  =  Bellman_MSE   +   α · cql_loss
```

(missä**`Bellman_MSE`** on normaali virhe normaalista Q-oppimisesta,
mittaamalla kuinka paljon verkon nykyinen arvaus poikkeaa omasta
tulevaisuuden arvaus).

`α` on konservatiivisuuden nuppi. Liian pieni → jakelun siirtymä hiipii taaksepäin
in. Liian suuri → agentti on niin konservatiivinen, ettei se koskaan parane
tiedot.

---

## Esimerkkejä tosielämästä

- **Konservatiivinen shakkivalmentaja.** Voit jo oppia vain peleistä
  pelannut. Huolimaton valmentaja sanoo: "Tämä hypoteettinen liike ilman
  ennakkotapaus voi olla loistava!" CQL on valmentaja, joka sanoo "meillä ei ole
  Tietoja siitä - pysytään todellisten pelaajien kokeiluissa liikkeissä."
- **Ravintolamenuvaihtoehdot.** Yelp-arvostelut eivät koskaan kata
  valikon ulkopuolisia kohteita. Naiivi politiikka suosittelisi valikon ulkopuolisia kohteita
  perustuu hallusinoituihin viiden tähden luokitukseen. CQL suosittelee vain sitä, mikä on
  tilattu tarpeeksi monta kertaa luottaakseen.
- **Robotti tarttuu tukista.** Robottissa on video, jossa tarrataan kuppeihin,
  pulloja ja kirjoja – mutta ei koskaan veistä. CQL kieltäytyy luottavaisesti
  suosittelen "tarttua veitseen terästä".

---

## Mitä koodimme tekee

Käsikirjoitus `cql.py`:

1. **Lataa neljä tietojoukkoa**, jotka on rakentanut `d4rl_dataset.py`.
2. ** Poimintoja `medium-replay`** harjoitussarjana - se on realistisin
   (sekoitettu laatu) ja kaikkein haitallisimmat naiiveille menetelmille.
3. **Kouluttaa kolme agenttia puhtaasti offline-tilassa**, samoissa olosuhteissa paitsi
   varten `α`:
   - `α = 0`   → naiivi offline-DQN (ei rangaistusta – rikkinäinen perusviiva)
   - `α = 1.0` → lievä CQL
   - `α = 5.0` → vahva CQL
4. **Arvioi jokaisen 2500 kaltevuusaskeleen** vierimällä ahneesti
   todellisessa ympäristössä (10 jaksoa). Tämä on *ainoa* env-yhteystieto;
   koulutus itsessään ei koskaan näe env.
5. **Oppimiskäyrät** piirtää `outputs/cql.png`.

---

## Mitä sinun pitäisi nähdä

Tyypillinen ajo tulostaa jotain tällaista:

```
Final evaluation returns (avg over 10 episodes, greedy):
  naive offline DQN (alpha=0)         ->  ~30-150  (unstable; often crashes)
  CQL (alpha=1.0)                     ->  ~300-450
  CQL (alpha=5.0)                     ->  ~450-500
```

Oppimiskäyräkaaviossa:

- **punainen käyrä** (`α = 0`) kiipeää aikaisin, sitten usein **putoaa kalliolta**
  kerran leviämismuutoksen hallusinaatiot tartuttavat **Bellmanin kohteen**
  (numero, jota käytämme "oikeana vastauksena" kun harjoittelemme Q-verkkoa:
  `r + γ · max Q(s', ·)`). Kun haamu-Q-arvot saastuttavat kohteen,
  jokainen kaltevuusaskel pahentaa asioita. **Bellmanin tappio** (MSE
  Q-verkon ennusteen ja Bellman-kohteen välillä) näyttää hyvältä -
  se on ongelman **petos**: verkko on täydellinen
  omien väärien uskomustensa mukainen, joten menetys ei anna varoitusta.
- **oranssi käyrä** (`α = 1.0`) kiipeää hitaammin, mutta **pysyy ylhäällä**.
- **vihreä käyrä** (`α = 5.0`) on vakain ja yleensä paras.

Bellmanin menetyspaneeli näyttää toisen merkkipalun: naiivi DQN:n menetys voi jäädä
pieni, kun taas sen politiikka on kauheaa, koska verkko on sisäinen
sopusoinnussa sen omien hallusinaatioiden kanssa.

---

## Missä CQL sijaitsee kentällä

CQL oli *iso*, koska se antoi periaatteellisen ja yksinkertaisen korjauksen
jakelun muutos. Sukulinja:

```
DQN (online)
   │
   ▼
Naive offline DQN  ── breaks because of distribution shift
   │
   ▼
CQL (Kumar 2020)    ── add a conservative penalty: Q is a lower bound
   │
   ▼
IQL (Kostrikov 2021)  ── never query Q on un-seen actions in the first place
   │
   ▼
Decision Transformer (Chen 2021)  ── skip Q entirely, treat RL as sequence modelling
                                      (predict the *next action* given past states and
                                       a desired total return, exactly like an LLM
                                       predicts the next word)
```

Jokainen vaihe tässä sukulinjassa on erilainen vastaus samaan kysymykseen:
**miten voin välttää kysymästä Q-verkostoltani asioita, joita se ei ole koskaan nähnyt?**

---

## Avainsanat muistaa

| sana | Merkitys |
|------|---------|
| **Jakelumuutos** | Koulutettu politiikka haluaa toimia tietojen ulkopuolella |
| **Pois jakelusta (OOD)** | (s, a) -pari, jota ei koskaan tallennettu tietojoukkoon |
| **Tosi Q** | Todellinen odotettu tuleva tuotto toimille `a` tilassa `s`, jos voisimme mitata sen täydellisesti |
| **Konservatiivinen Q** | Opittu Q-funktio, joka yrittää pysyä todellisen Q:n alapuolella ylilupaamisen sijaan |
| **Logsumexp** | Tasainen, differentioituva approksimaatio `max` |
| **Alfa (α)** | CQL:n konservatiivisuusnuppi – kuinka vaikeaa painaa Q alas OOD-toimissa |

---

## Yhden lauseen yhteenveto

> **CQL lisää "pessimismirangaistuksen", joka rankaisee korkeita Q-arvoja toimissa
> tietojoukkoa ei koskaan kokeiltu – joten käytäntöön ei voi rakastua
> hallusinaatioita.**
