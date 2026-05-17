# PPO-hyperparametrin herkkyys: mikä on tärkeintä?

## Miksi hyperparametrit ovat tärkeitä

Kuvittele leipomassa suklaakakkua. Resepti vaatii:
- 2 munaa
- 200 g jauhoja
- 1 tl leivinjauhetta
- 35 minuuttia 180 °C:ssa

Jos käytät 10 munaa, kakku räjähtää. Jos käytät 0,1 tl leivinjauhetta, se ei nouse.
Jos paistat 300°C:ssa 10 minuuttia, se palaa ulkoa ja on sisältä raakaa.

**PPO:n hyperparametrit ovat kuin ainesosia ja uunin asetuksia.** Oikea yhdistelmä toimii
kauniisti; väärät asetukset voivat estää oppimisen kokonaan.

Tämä komentosarja testaa systemaattisesti 3 keskeistä hyperparametria muuttamalla vain YHDEN kerrallaan,
kunkin asetuksen suorittaminen 3 eri satunnaisella siemenellä ja tulosten vertailu.

---

## Kolme koetta

### Koe 1: Leikkaa Epsilon (ε)

```
ε = 0.05   (very conservative — only tiny policy changes allowed)
ε = 0.2    (standard — balanced safety and speed)
ε = 0.4    (aggressive — allows large policy changes)
```

**Mitä ε ohjaa?**

ε on vanhan käytännön ympärillä olevan "turvaikkunan" koko:
```
ratio must stay in [1 - ε,  1 + ε]
ε=0.05: ratio in [0.95, 1.05]  ← tiny changes
ε=0.2:  ratio in [0.80, 1.20]  ← standard  
ε=0.4:  ratio in [0.60, 1.40]  ← large changes
```

**Tosielämän esimerkki:** Ajattele ε:tä "kuinka pitkälle voit ohjata autoa yhdellä liikkeellä".
- ε=0,05: Kuten jäällä ajettaessa – vain pieniä säätöjä
- ε=0,2: Normaali ajo – kohtuulliset käännökset
- ε=0,4: Kilpa-kuljettaja – aggressiivinen ohjaus, riski **loikkaamisesta** (hallinnan menettäminen, koska muutos on liian jyrkkä, kuten auto luistaa tieltä)

**Odotetut tulokset:**
- ε=0,05: Hidas mutta vakaa oppiminen (liian varovainen)
- ε=0,2: Hyvä tasapaino (**"Kultakutri"-arvo** - ei liian pieni, ei liian suuri, juuri sopiva - nimetty sadun mukaan, jossa Kultakutri poimii puuron, joka ei ole liian kuuma eikä liian kylmä)
- ε=0,4: Voi oppia nopeasti, mutta saattaa **ylittää ja värähdellä** (ylitys = ylittää optimaalisen linjan; värähtelee = pomppii edestakaisin sen ympäri asettumatta, kuten heiluri, joka heiluu liian pitkälle molempiin suuntiin)

---

### Koe 2: Oppimisnopeus

```
lr = 1e-4  (slow but stable)
lr = 3e-4  (standard)
lr = 1e-3  (fast but risky)
```

**Mitä oppimisnopeus ohjaa?**

Oppimisnopeus on kuin "askelkoko" mäkeä noustessa (jokainen askel = yksi päivitys hermoverkon painoihin, liikuttamalla sitä hieman palkkiota parantavaan suuntaan):
- Liian pieni: Huipulle pääseminen kestää ikuisuuden (konvergoi hitaasti)
- Liian suuri: Ylität huipun ja putoat alas toiselle puolelle (**hajoaa** – harjoituspalkkio romahtaa tai vaihtelee hurjasti sen sijaan, että paranisi tasaisesti)
- Aivan oikein: Tasaista edistymistä kohti huippua

**Tosielämän esimerkki:** Kitaran kielen viritys.
- lr=1e-4: Pienet käännökset virityksessä **tappi** (nuppi, jota kierrät kierteen kiristämiseksi tai löysäämiseksi) – kestää ikuisuuden, mutta tarkasti
- lr=3e-4: Normaali viritys – löydä oikea sävelkorkeus muutamalla kierroksella
- lr=1e-3: Suuret **nykikset** (äkilliset kovat vedot) tapissa — voivat **katkaista** narun (katkoa sen kokonaan, aivan kuten liian suuret päivitykset voivat katkaista harjoittelun peruuttamattomasti)!

**Odotetut tulokset:**
- lr=1e-4: Lopulta hyvä, mutta erittäin hidas
- lr=3e-4: Paras suorituskyky kokonaisuudessaan
- lr=1e-3: Nopea alkukehitys, sitten epävakaus

---

### Koe 3: Päivitä aikakaudet (K)

```
K = 3   (conservative — few passes through each batch)
K = 10  (standard)
K = 20  (aggressive — many passes through each batch)
```

**Mitä päivityskaudet hallitsevat?**

Kun olet kerännyt **julkaisun** (= pelannut peliä jonkin verran uuden kokemuksen hankkimiseksi – kuten opiskelija tekee kotitehtävän ennen sen tarkistamista), PPO paketoi, jotka kokevat **erän** (= koko sarja tilan, toiminnon, palkkion sarjat kyseisestä julkaisusta). Sitten se suorittaa K **passia** (= täydet pyyhkäisysarjan läpi, jokainen kierros päivittää verkon kerran) samoilla tiedoilla.
Lisää aikakausia = purista enemmän oppimista jokaisesta erästä, mutta on vaara, että **sovitetaan liikaa vanhentuneisiin tietoihin** (= muistiin kuviot, jotka olivat totta vanhan käytännön mukaan mutta eivät ole enää voimassa, kun käytäntö on päivitetty, kuten opiskelija, joka muistaa viime vuoden kokeen ja epäonnistuu uudessa).

**Tosielämän esimerkki:** Opiskelija harjoittelee 20 matemaattisen tehtävän kanssa.
- K=3: Tee jokainen tehtävä 3 kertaa → opettele edelleen, älä sovita harjoitussarjaa liikaa
- K=10: Tee jokainen tehtävä 10 kertaa → hallitse nämä erityistehtävät
- K=20: Tee jokainen tehtävä 20 kertaa → **muistiin ratkaisut ymmärtämättä todella matematiikkaa** (= malli sopii täydellisesti tiettyyn joukkoon, mutta menettää yleistyskyvyn)!

> ⚠️ **"Mutta tulokset K=20 näyttävät hyvältä – miksi minun pitäisi välittää?"**
> PPO:n leikkaustemppu rajoittaa, kuinka paljon käytäntö voi muuttua läpimenoa kohden, joten K=20 ei aiheuta äkillistä romahdusta.
> Agentti sopeutuu kuitenkin edelleen hiljaa liikaa tietoihin, jotka eivät enää heijasta sitä, mitä nykyinen käytäntö todellisuudessa kokisi.
> Tämä **hidastaa pitkän aikavälin oppimista**: jokainen käyttöönotto opettaa agentille vähemmän kuin sen pitäisi, koska myöhemmät siirrot kierrättävät yhä vanhentuneempaa tietoa.
> Vahinko on asteittainen, ei dramaattinen – juuri siksi se on helppo jättää huomiotta lyhyissä kokeissa.

Leikkaus estää katastrofaalisen ylisovituksen, mutta liian monet aikakaudet voivat silti hidastaa yleistä oppimista.

**Odotetut tulokset:**
- K=3: Tehokkaampi (jokin oppimispotentiaalia menee hukkaan erää kohden)
- K=10: Hyvä tasapaino
- K = 20: Riski siitä, että käytäntö muuttuu **liian luottavaiseksi vanhentuneiden tietojen suhteen** (= verkon päivitykset perustuvat kokemuksiin, jotka eivät enää vastaa nykyisen käytännön kohdata, ja heikentävät hiljaa näytteen tehokkuutta)

---

## Kuinka lukea tuloksia

Kaaviossa on kolme kuvaajaa, joista jokainen vaihtelee yhdellä hyperparametrilla:

```
Left graph:    Clip Epsilon — which ε learns fastest?
Middle graph:  Learning Rate — which lr is most stable?
Right graph:   Update Epochs — which K finds the best policy?
```

Jokainen rivi on **keskimääräinen palkkio 3 siemenestä** (satunnaisuuden vähentämiseksi).

**Mitä etsiä:**
1. **Oppimisnopeus:** Mikä linja saavuttaa korkean palkinnon nopeimmin?
2. **Lopullinen suorituskyky:** Mikä linja saavuttaa suurimman lopullisen palkinnon?
3. **Vakaus:** Millä linjalla on vähiten värähtelyä?

Hyvä hyperparametri tasapainottaa kaikki kolme!

---

## Metodologia: Tieteellinen kokeilu

Tässä kokeessa käytetään **ablaatiotutkimusta** (= menetelmä, jossa yksi komponentti poistetaan tai vaihdellaan kerrallaan sen yksilöllisen vaikutuksen mittaamiseksi – nimetty tieteellisen käytännön mukaan, jossa kudosta poistetaan valikoivasti sen toiminnan tutkimiseksi):
1. Valitse oletusarvot: ε=0,2, lr=3e-4, K=10
2. Muuta YKSI parametri kerrallaan
3. Pidä kaikki muu kunnossa
4. Vertaa tuloksia

Tämä kertoo meille kunkin parametrin vaikutuksen erikseen.

**Tosielämän esimerkki:** Testaa, auttaako uusi lannoite kasveja:
- Vaihda lannoite, pidä kaikki muu ennallaan (sama maaperä, vesi, auringonvalo)
- Jos kasvit kasvavat paremmin → lannoite auttoi!

---

## Yleisiä havaintoja käytännössä

| Hyperparametri | Liian pieni | Sweet Spot | Liian suuri |
|----------------|-----------|------------|-----------|
| **ε (leike)** | Hidas lähentyminen | ε ≈ 0,2 | Epävakaus |
| **lr** | Liian hidas | 2,5e-4 - 3e-4 | Eroaminen |
| **K (aikakaudet)** | **Tietoja hukkaan** (hylkää käyttöönotto ennen täyden signaalin poistamista) | K = 4-10 | Ylisovitus vanhentuneisiin käyttöönottotietoihin |
| **n_vaihetta** | Liian meluisa | 128-2048 | **OOM-muistivirheet** (käyttää liikaa RAM-muistia) |
| **erän_koko** | Liian meluisa | 32-256 | **OOM-muistivirheet** (käyttää liikaa RAM-muistia) |

Nämä "suloiset kohdat" voivat muuttua ympäristön mukaan!

---

## Keskeinen näkemys: PPO on suhteellisen kestävä

Verrattuna aikaisempiin algoritmeihin (kuten DQN ilman kohdeverkkoja), PPO on suhteellinen
vankka hyperparametrivalinnoissa. Leikkausmekanismi tarjoaa luonnollisen turvaverkon.

**Esimerkki tosielämästä:** Auto, jossa on **ABS** (lukkiutumaton jarrujärjestelmä – turvaominaisuus, joka estää pyöriä lukkiutumasta kovan jarrutuksen aikana ja pitää kuljettajan hallinnassa) jarrut verrattuna:
- Ilman ABS:ää (DQN): Yksi väärä käännös (huono hyperparametri) ja pyörit ulos
- ABS:llä (PPO): Auto korjaa itsensä – kaikki kohtuulliset hyperparametrit toimivat

Tämä kestävyys on tärkein syy, miksi PPO on käytännössä suosituin RL-algoritmi!

---

## Avaimet takeawayt

| käsite | Pelkkää englantia |
|---------|---------------|
| **Ablaatiotutkimus** | Muuta yksi asia kerrallaan nähdäksesi sen vaikutuksen |
| **Clip epsilon ε** | Turvaraja - 0,2 on yleensä paras |
| **Oppimisaste** | **Askelkoko** — kuinka paljon verkon painoja säädetään jokaisen erän jälkeen (ajattele sitä jokaisen askeleen kokona, kun kävelet kohti tavoitetta). **2,5e-4 - 3e-4** on tieteellinen merkintä arvoille 0,00025 - 0,0003 - nämä ovat dimensioimattomia kertoimia, eivät aika-arvoja |
| **Päivitä aikakaudet K** | Kuinka monta kertaa kutakin erää käytetään uudelleen - 4-10 on vakiona |
| **Satunnaiset siemenet** | Jokainen koe toistetaan eri **satunnaisilla siemenillä** (= satunnaislukugeneraattoriin syötetty aloitusnumero, joka ohjaa kaikkia harjoittelun satunnaisia valintoja). Useiden siementen käyttäminen paljastaa, ovatko tulokset johdonmukaisia vai onko se vain onnekas |

---

## Yhteenveto: Käytännön gradienttimenetelmät yhdellä silmäyksellä

```
REINFORCE              A2C                    PPO
     │                  │                      │
Full episodes     N-step updates         N-step + clipping
Simple but noisy  Faster but unstable    Stable + efficient
Best for easy     Medium difficulty      Hard environments
environments      environments           (industry standard)
```

**Jos opit vain YHDEN algoritmin tästä vaiheesta, opi PPO.** Se on perusta:
- OpenAI:n ChatGPT-koulutus (RLHF käyttää PPO:ta)
- DeepMindin AlphaGo-seuranta
- Nykyaikaisin robotiikkatutkimus
- Videopelien pelaaminen tekoälyllä
