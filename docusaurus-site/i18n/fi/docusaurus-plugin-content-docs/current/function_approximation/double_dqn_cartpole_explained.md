# Double DQN: Yliluottamusongelman korjaaminen 🤔

## Ongelma: DQN ajattelee, että se on parempi kuin se on

Kuvittele, että sinulta kysytään: "Mikä on kaupungin paras ravintola?"

Saatat sanoa: "Pizza Palace on uskomaton – se on ehdottomasti 10/10!" Mutta sinulla on vain
ollut siellä kahdesti. Et todellakaan tiedä, onko se *todella* 10/10. Saatat olla
yliarvioi, koska olet ollut onnekas hyvän pizzan kanssa näillä kahdella vierailulla.

Sama ongelma tapahtuu DQN:n kanssa: agentti **yliarvioi Q-arvot**.

---

## Miksi DQN yliarvioi?

Kun DQN laskee tavoitteen, se tekee:
> tavoite = palkkio + γ × **max** Q(seuraava_tila)

The `max` on ongelma! Kun valitset useista meluisista arvioista suurimman osan, sinä
valitse melkein aina se, jossa on suurin satunnainen virhe (ylössuuntainen harha).

**Tosielämän esimerkki:** Sinulla on 5 ystävää, jotka arvaavat rakennuksen korkeuden. Heidän
arvaukset ovat: 40m, 38m, 45m (onnekas arvaus!), 39m, 41m. Todellinen korkeus on 40m.
Jos käytät `max(guesses)` = 45 m, olet kaukana! Meluisten arvausten maksimi
on lähes aina yliarvioitu.

Yli tuhansien päivitysten aikana DQN harjoittelee jatkuvasti kohti näitä ylipaisutettuja tavoitteita,
oppia, että asiat ovat paremmin kuin ne todellisuudessa ovat. Tämä voi hidastaa oppimista tai aiheuttaa
agentti tekemään liian itsevarmoja, huonoja päätöksiä.

---

## Double DQN Fix

**Double DQN** (Hasselt et al., 2016) jakaa `max` kahteen vaiheeseen:

**Vaihe 1 – Mikä toimenpide?** Käytä **verkkoverkostoa** valitaksesi parhaan toiminnon:
> paras_toiminta = argmax Q_online(seuraava_tila)

**Vaihe 2 – Mikä on sen arvo?** Käytä **kohdeverkkoa** toimenpiteen arvioimiseen:
> tavoite = palkkio + γ × Q_tavoite(seuraava_tila, paras_toiminta)

```
Vanilla DQN:   target = r + γ × max_a Q_target(s', a)
                                 ↑ same network picks AND evaluates → biased

Double DQN:    best_a = argmax_a Q_online(s', a)   ← online picks
               target = r + γ × Q_target(s', best_a) ← target evaluates
                                 ↑ different networks → less biased
```

**Tosielämän esimerkki:** Työhaastattelussa et anna työnhakijan arvostella itseään
oma suorituskykytesti (se on vanilja DQN-ongelma!). Sen sijaan ehdokas
*nimeää* parhaan työnsä, ja *erillinen* tutkija arvioi sen.
Kaksi eri ihmistä = oikeudenmukaisempi arviointi!

---

## Miksi eroaminen auttaa?

Näillä kahdella verkostolla (online ja kohde) on eri painoarvot, koska kohde on
päivitetään harvemmin. Heillä on erilaisia ​​"mielipiteitä" siitä, mikä toiminta on paras.

Kun he ovat eri mieltä:
- Online sanoo: "Toiminto A näyttää hyvältä!"
- Target sanoo: "Itse asiassa Action A on vain okei - arvoltaan noin 7, ei 10"

Käyttämällä kohdeverkoston ARVO-arviota online-verkoston VALITTU toiminnolle,
saamme rehellisemmän, vähemmän paisutetun luvun.

---

## Koodiero: vain yksi rivi!

Ainoa koodimuutos vaniljasta kaksois-DQN:ään on tavoitelaskelmassa:

```python
# Vanilla DQN:
q_next = target_net(s_next).max(dim=1).values

# Double DQN:
best_actions = online_net(s_next).argmax(dim=1, keepdim=True)   # pick with online
q_next = target_net(s_next).gather(1, best_actions)              # evaluate with target
```

Vain kaksi riviä vaihtuu – mutta vaikutus vakauteen ja tarkkuuteen on merkittävä!

---

## Mitä vertailu osoittaa

Kun juokset `double_dqn_cartpole.py`, näet kaksi kaaviota:

**Juoni 1: Oppimiskäyrät**
- Sekä vaniljan että kaksois-DQN:n pitäisi ratkaista CartPole
- Double DQN konvergoi usein nopeammin ja vakaammin
- CartPole on tarpeeksi yksinkertainen, joten ero on vaatimaton; se on dramaattisempaa Atarissa

**Kaavio 2: Q-arvon arviot**
- Vanilla DQN: Q-arvot ajautuvat ylöspäin ajan myötä (yliarviointi)
- Double DQN: Q-arvot pysyvät vaatimattomina ja tarkempina

Q-arvon yliarviointikaavio on keskeinen näkemys – se näyttää vanilja DQN-oppimisen
paisutetut arvot, jotka lopulta heikentävät suorituskykyä.

---

## Kuinka paljon parempi on Double DQN?

| Metrinen | Vanilja DQN | Double DQN |
|--------|------------|------------|
| Q-arvon tarkkuus | Yliarvioinnit | Tarkempi |
| Oppimisen vakaus | Lisää varianssia | Vähemmän varianssia |
| CartPolen suorituskyky | Hyvä | Hieman parempi |
| Atari suorituskyky (50 peliä) | Perustaso | +2,6× enemmän pelejä lähellä ihmisen tasoa |

Monimutkaisissa Atari-peleissä Double DQN teki paljon suuremman eron kuin CartPolessa
(koska Atarilla on paljon meluisemmat Q-arvoarviot).

---

## DQN-parannusten perhe

Double DQN on vain yksi monista vanilja-DQN:n parannuksista. "Rainbow"-paperi
(2017) yhdistää kuusi parannusta:

1. **Double DQN** (korjaa yliarviointi) ← tämä kirjoitus!
2. **Priorisoitu toisto** (lue lisää yllättävistä kokemuksista)
3. **Dueling Networks** (erota "kuinka hyvä tämä tila on?" ja "mikä on paras toiminta?")
4. **Monivaiheinen palautus** (katso pidemmälle tulevaisuuteen)
5. **Distributional RL** (lue koko tuottojakauma, ei vain keskiarvo)
6. **NoisyNets** (oppinut tutkimisen sijaan [ε-ahne](../foundations/multi_armed_bandit_explained.md#the-epsilon-greedy-strategy))

Rainbow yhdisti ne KAIKKI ja saavutti aikansa parhaan Atari-suorituksen!

---

## Avainsanasto

| sana | Merkitys |
|------|---------|
| **Yliarviointi** | Q-arvot ovat korkeampia kuin todelliset arvot (liian optimistisia) |
| **Double DQN** | Käyttää online-verkkoa toimintojen valintaan, kohdeverkkoa arviointiin |
| **irrottaminen** | Erotetaan kaksi tehtävää, jotka tehtiin samassa verkossa |
| **Vinouma (bias)** | Järjestelmällinen virhe yhteen suuntaan (aina liian korkea tai aina liian matala) |
| **Rainbow** | DQN-versio, joka yhdistää 6 parannusta maksimaalisen suorituskyvyn saavuttamiseksi |

---

## Yhteenveto: Vaiheen 3 matka

Olet nyt suorittanut koko vaiheen 3 etenemisen:

| Algoritmi | Mitä se lisää | Miksi se auttaa |
|-----------|-------------|-------------|
| Lineaarinen Q | Hermoverkko → yksinkertainen kaava | Käsittelee jatkuvia tiloja |
| Naiivi DQN | Täysi neuroverkko | Oppii monimutkaisia kuvioita |
| + Toistopuskuri | Satunnainen muistinäytteenotto | Katkaisee korrelaatiot |
| + Kohdeverkko | Jäädytetty kopio kohteille | Vakauttaa "bullseye" |
| Atari DQN | CNN + kehysten pinoaminen | Oppii pikseleistä! |
| Double DQN | Erillinen valinta/arviointi | Vähentää yliarviointia |

Jokainen vaihe ratkaisi tietyn ongelman. Näin todellinen tutkimus toimii – varovainen
parannus kerrallaan!
