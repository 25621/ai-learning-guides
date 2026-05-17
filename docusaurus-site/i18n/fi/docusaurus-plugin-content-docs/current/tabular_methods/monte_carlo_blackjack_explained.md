# Monte Carlo Control Blackjackille 🃏

## Mikä se on?

Oletko koskaan pelannut korttipeliä, jossa sinun on päätettävä: **"Otanko toisen kortin,
vai olenko tyytyväinen siihen mitä minulla on?"**

**Blackjack** (kutsutaan myös "21") on juuri sitä! Haluat korttisi olevan mahdollisimman lähellä
21 asti kuin mahdollista ylittämättä. Jos ylität 21-vuotiaana, "hävität" ja häviät!

**Monte Carlo -ohjaus** on kuinka robotti oppii pelaamaan blackjackia – pelaamalla *tuhansia
täydet pelit* ja muistaa mikä toimi ja mikä ei.

---

## Suuri idea: Opi kokonaisista tarinoista

Sana "Monte Carlo" tulee kuuluisalta Monacon kasinolta. Matematiikassa se tarkoittaa:
**käytä satunnaisia ​​kokeita oppiaksesi jotain**.

Näin se toimii:

1. **Pelaa koko peli** (kokonainen jakso) käyttämällä mitä tahansa strategiaa
2. **Katso mitä tapahtui**: Voititko? Menettää? Piirrä?
3. **Työskentele taaksepäin**: Oliko lyöminen (kortin ottaminen) 17-vuotiaana hyvä idea? Entä klo 14?
4. **Päivitä muistisi**: Muista, johtiko jokainen päätös voittoon vai tappioon

Tee tämä **500 000 pelille** ja saat erittäin hyvää!

**Tosielämän esimerkki:** Kuvittele, että opit kokkaamaan tekemällä 500 000 ateriaa. Joka kerta sinä
muista tarkalleen mitä teit – ja maistuiko ateria hyvältä. Riittävän yrityksen jälkeen sinä
tietää: "Liika suolan lisääminen tässä vaiheessa teki siitä aina huonon." Monte Carlo toimii samalla tavalla!

---

## Keskeinen ero SARSAsta ja Q-Learningistä

SARSA ja Q-Learning päivittävät tietonsa **jokaisen vaiheen jälkeen** (jopa jakson puolivälissä).
Monte Carlo odottaa, kunnes **koko jakso on valmis**, ja katsoo sitten kaikkea.

| menetelmä | Päivitykset milloin? | Tarvitseeko täydellisen jakson? |
|--------|---------------|------------------------|
| **TD (SARSA, Q-Learning)** | Jokaisen askeleen jälkeen | Ei |
| **Monte Carlo** | Jokaisen täyden jakson jälkeen | Kyllä |

Tämä tekee Monte Carlosta helpommin ymmärrettävän, mutta se ei voi oppia ennen kuin jokainen jakso päättyy.

---

## Blackjackin osavaltio

Robotti katsoo kolmea asiaa joka käänteessä:
1. **Kortin kokonaissumma** (12-21)
2. **Mitä korttia jakaja näyttää?** (Ässä - 10)
3. **Onko minulla käyttökelpoinen ässä?** (Ässä voi olla 1 tai 11)

Näistä kolmesta tiedosta se päättää: **Lyö (ota kortti) vai Stick (pysähdy)**?

---

## Mitä koodimme löysi

**500 000 Blackjack-pelin** jälkeen:

| Tulos | Prosenttiosuus |
|---------|------------|
| **Voitot** | **43.1%** |
| **Arvonta** | 8.9% |
| **Tappiot** | 48.0% |

Tämä on lähellä matemaattisesti optimaalista "perusstrategiaa" (noin 42-43 % voittoja)!
Robotti oppi milloin lyödä ja milloin tikkua – vain pelaamalla ja muistamalla.

Opittu käytäntö osoittaa:
- **Lyö** (ota kortti), kun kokonaissumma on pieni (et todennäköisesti putoa)
- **Pidä kiinni**, kun kokonaissumma on korkea (saatat kaatua, jos otat toisen kortin)
- **Käytettävissä oleva Ace** antaa sinun olla aggressiivisempi (se voi vaihtaa 11:stä 1:een tarvittaessa)

---

## Esimerkkejä tosielämästä

- **Sään ennustaminen**: Monte Carlo -simulaatiot suorittavat tuhansia "mitä jos" -skenaarioita
  ennustamaan huomisen säätä.
- **Pörssimallinnus**: Analyytikot simuloivat tuhansia mahdollisia futuureja arvioidakseen
  riskiä.
- **Oppii pelaamaan shakkia**: Pelaaja arvioi kokonaisia pelejä (ei vain yksittäisiä liikkeitä).
  ymmärtää, mikä strategia johti voittoon.

---

## Avainsanat muistaa

- **Episode**: Yksi täydellinen peli alusta loppuun
- **Palautus (G)**: Kokonaispalkinto, joka on kerätty pelin pisteestä loppuun asti
- **Jokainen käynti MC:ssä**: Päivitä osavaltion pisteet aina, kun vierailet siinä jaksossa
- **Ei bootstrappingia**: Monte Carlo ei käytä arvioita tulevista arvoista – se odottaa
  todellista lopputulosta!
- **ε-pehmeä käytäntö** (ε = epsilon): Tee yleensä tunnetuin toiminto, mutta joskus tutkii satunnaisesti

Suuri idea: **Monte Carlo oppii pelaamalla monia kokonaisia pelejä. Se on kuin oppimista
kokemus - muistat kaiken tapahtuneen ja ymmärrät, mikä johti voittoon!**
